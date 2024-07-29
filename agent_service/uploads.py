import base64
import csv
import enum
import logging
from io import StringIO
from typing import Any, Generator, List, Optional, Set, Tuple

from fastapi import UploadFile
from pa_portfolio_service_proto_v1.well_known_types_pb2 import StockHolding
from pydantic import BaseModel

from agent_service.external.dal_svc_client import (
    ParsePortfolioWorkspaceResponse,
    get_dal_client,
)
from agent_service.external.grpc_utils import datetime_to_timestamp
from agent_service.external.pa_svc_client import (
    create_ts_workspace,
    modify_workspace_historical_holdings,
    recalc_strategies,
)
from agent_service.types import Message
from agent_service.utils.agent_event_utils import send_chat_message
from agent_service.utils.async_db import AsyncDB
from agent_service.utils.async_utils import gather_with_concurrency
from agent_service.utils.constants import SUPPORTED_FILE_TYPES
from agent_service.utils.logs import async_perf_logger

logger = logging.getLogger(__name__)


class UploadType(str, enum.Enum):
    PORTFOLIO = "portfolio"
    WATCHLIST = "watchlist"


class UploadResult(BaseModel):
    success: bool = True
    message: str = ""


def chunks(input_list: List[Any], chunk_size: int) -> Generator[List[Any], None, None]:
    for i in range(0, len(input_list), chunk_size):
        yield input_list[i : i + chunk_size]


@async_perf_logger
async def create_workspace_from_bytes(
    data: bytes, name: str, user_id: str, content_type: str
) -> Tuple[Optional[str], Optional[str], int]:
    # encode file data into base 64
    b64data = base64.b64encode(data).decode()

    # call DAL to map CSV data to gbi_id, weight and date
    dal_client = get_dal_client()
    parsed_response: ParsePortfolioWorkspaceResponse = await dal_client.parse_file(
        b64data=b64data,
        content_type=content_type,
    )

    if not parsed_response.securities:
        logger.info(f"No securities were found while parsing '{name}' for {user_id=}")
        return None, None, 0

    # get latest holding date
    latest_date = max(security.date for security in parsed_response.securities if security.date)

    # populate proto holdings (and latest holdings)
    holdings = []
    latest_holdings = []

    for security in parsed_response.securities:
        proto_holding = StockHolding(
            gbi_id=security.gbi_id,
            date=(datetime_to_timestamp(security.date) if security.date else None),
            weight=security.weight,
        )
        holdings.append(proto_holding)
        if security.date == latest_date:
            latest_holdings.append(proto_holding)

    # create the workspace with the latest holdings
    workspace_id, strategy_id = await create_ts_workspace(
        user_id=user_id, holdings=latest_holdings, workspace_name=name
    )

    # insert history in chunks of 5000

    chunk_size = 5000
    chunked_holdings = chunks(holdings, chunk_size)

    await gather_with_concurrency(
        [
            modify_workspace_historical_holdings(
                user_id=user_id, workspace_id=workspace_id, holdings=chunk
            )
            for chunk in chunked_holdings
        ]
    )

    logger.info(f"Created workspace for {user_id=}, {workspace_id=} {strategy_id=}")

    # kick off a recalc
    await recalc_strategies(user_id=user_id, strategy_ids=[strategy_id])

    return workspace_id, strategy_id, len(latest_holdings)


class UploadHandler:
    def __init__(
        self,
        user_id: str,
        upload: UploadFile,
        db: AsyncDB,
        agent_id: Optional[str] = None,
        send_chat_updates: bool = False,
    ) -> None:
        self.user_id = user_id
        self.upload = upload
        self.agent_id = agent_id
        self.send_chat_updates = send_chat_updates
        self.db = db

    async def _read_upload_bytes(self) -> bytes:
        raw_bytes = await self.upload.read(self.upload.size if self.upload.size is not None else 0)
        await self.upload.seek(0)
        return raw_bytes

    @async_perf_logger
    async def identify_upload_type(self) -> Tuple[Optional[UploadType], Optional[str]]:

        # TODO: use a better way to identify file types
        # just support basic portfolio / watchlist by parsing header column
        try:
            data_string = (await self._read_upload_bytes()).decode()
            reader = csv.reader(StringIO(data_string), delimiter=",")
            header_row = next(reader)
            columns: Set[str] = {column_name.lower() for column_name in header_row}

            # simple validation to check for one of these fields, it would be either portfolio or watchlist
            if columns.intersection({"isin", "symbol"}):
                if "weight" in columns:
                    return (
                        UploadType.PORTFOLIO,
                        "This looks to be a portfolio holdings file, let me import that...",
                    )
                # TODO WC-731: return watchlist type here
        except Exception:
            logger.exception(
                f"Error while parsing upload '{self.upload.filename}' for {self.user_id=}"
            )

        return None, None

    @async_perf_logger
    async def process_upload(self, upload_type: UploadType) -> UploadResult:
        if upload_type == UploadType.PORTFOLIO:

            data = await self._read_upload_bytes()

            try:
                workspace_id, strategy_id, latest_holding_count = await create_workspace_from_bytes(
                    data=data,
                    name=self.upload.filename,
                    user_id=self.user_id,
                    content_type=(
                        self.upload.content_type if self.upload.content_type else "text/csv"
                    ),
                )
            except Exception:
                logger.exception(f"Failed to create portfolio for {self.user_id}")
                return UploadResult(
                    message="Sorry, I ran into some issues while importing this portfolio."
                )

            if not workspace_id or not strategy_id:
                return UploadResult()

            return UploadResult(
                message="Based on this file, I've created a portfolio for you"
                + f" named '{self.upload.filename}' with {latest_holding_count}"
                + f" holding{'s' if latest_holding_count != 1 else ''}."
            )

        return UploadResult()

    @async_perf_logger
    async def handle_upload(self) -> UploadResult:
        """
        Process an uploaded file, return True if processing was successful.
        """
        if self.agent_id and self.send_chat_updates:
            await send_chat_message(
                Message(
                    agent_id=self.agent_id,
                    is_user_message=False,
                    message="I'm processing your upload, this may take a few moments.",
                    visible_to_llm=False,
                ),
                self.db,
                insert_message_into_db=True,
                send_notification=False,
            )
        logger.info(
            f"Starting identification for upload '{self.upload.filename}' for {self.user_id=}"
        )

        if self.agent_id and self.send_chat_updates:
            await send_chat_message(
                Message(
                    agent_id=self.agent_id,
                    is_user_message=False,
                    message="Upload complete, analyzing file contents...",
                    visible_to_llm=False,
                ),
                self.db,
                insert_message_into_db=True,
                send_notification=False,
            )

        upload_type, identify_message = await self.identify_upload_type()

        # if we could not identify the type, skip processing
        if not upload_type:
            if self.agent_id and self.send_chat_updates:
                await send_chat_message(
                    Message(
                        agent_id=self.agent_id,
                        is_user_message=False,
                        message=f"Sorry, I couldn't understand this file."
                        f" Currently I support file uploads like: {' '.join(SUPPORTED_FILE_TYPES)}",
                        visible_to_llm=False,
                    ),
                    self.db,
                    insert_message_into_db=True,
                    send_notification=False,
                )
            return UploadResult()

        if self.agent_id and self.send_chat_updates and identify_message:
            await send_chat_message(
                Message(
                    agent_id=self.agent_id,
                    is_user_message=False,
                    message=identify_message,
                    visible_to_llm=False,
                ),
                self.db,
                insert_message_into_db=True,
                send_notification=False,
            )

        logger.info(
            (
                f"Upload '{self.upload.filename}' for {self.user_id=} "
                f"is of type {upload_type}, starting processing..."
            )
        )
        result = await self.process_upload(upload_type)
        logger.info(f"Finished processing '{self.upload.filename}' for {self.user_id=}")
        if self.agent_id and self.send_chat_updates:
            await send_chat_message(
                Message(
                    agent_id=self.agent_id,
                    is_user_message=False,
                    message=result.message if result.message else "Upload processing finished!",
                    visible_to_llm=False,
                ),
                self.db,
                insert_message_into_db=True,
                send_notification=False,
            )
        return result
