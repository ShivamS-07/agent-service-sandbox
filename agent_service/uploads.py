import base64
import enum
import logging
from io import BytesIO
from typing import Any, Generator, List, Optional, Tuple

import pandas as pd
from fastapi import HTTPException, UploadFile
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
from agent_service.utils.logs import async_perf_logger

logger = logging.getLogger(__name__)


class FileType(enum.Enum):
    CSV = "csv"
    EXCEL = "excel"


class UploadType(enum.StrEnum):
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


@async_perf_logger
async def create_watchlist_from_bytes(
    data: bytes, name: str, content_type: str, user_id: str, jwt: Optional[str]
) -> Optional[str]:
    # encode file data into base 64
    b64data = base64.b64encode(data).decode()

    # call DAL to create workspace from file
    dal_client = get_dal_client()
    watchlist_id = await dal_client.create_watchlist_from_file(
        b64data=b64data, content_type=content_type, name=name, user_id=user_id, jwt=jwt
    )

    return watchlist_id


class UploadHandler:
    def __init__(
        self,
        user_id: str,
        upload: UploadFile,
        db: AsyncDB,
        agent_id: Optional[str] = None,
        send_chat_updates: bool = False,
        jwt: Optional[str] = None,
    ) -> None:
        self.user_id = user_id
        self.upload = upload
        self.agent_id = agent_id
        self.send_chat_updates = send_chat_updates
        self.db = db
        self.jwt = jwt

    async def read_upload_bytes(self) -> bytes:
        raw_bytes = await self.upload.read(self.upload.size if self.upload.size is not None else 0)
        return raw_bytes

    def identify_file_type(self) -> FileType:
        content_type = self.upload.headers.get("content-type")
        if content_type == "text/csv":
            return FileType.CSV
        elif content_type in (
            "application/vnd.ms-excel",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ):
            return FileType.EXCEL

        raise HTTPException(
            status_code=400, detail="Unsupported file type. Please upload a CSV or Excel file"
        )

    def identify_upload_type(self, file_type: FileType, raw_data: bytes) -> Tuple[UploadType, str]:
        # just support basic portfolio / watchlist by parsing header column
        bytes_io = BytesIO(raw_data)
        if file_type == FileType.CSV:
            first_row = pd.read_csv(bytes_io, nrows=1)
        elif file_type == FileType.EXCEL:
            first_row = pd.read_excel(bytes_io, nrows=1)
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {file_type}")

        columns = {col.lower() for col in first_row.columns}
        # simple validation to check for one of these fields, it would be either portfolio or watchlist
        if columns.intersection({"isin", "symbol"}):
            if "weight" in columns:
                return (
                    UploadType.PORTFOLIO,
                    "This looks to be a portfolio holdings file, let me import that...",
                )
            return (
                UploadType.WATCHLIST,
                "This looks to be a watchlist file, let me import that...",
            )

        raise HTTPException(
            status_code=400,
            detail="Unsupported upload type. Please upload a portfolio or watchlist",
        )

    @async_perf_logger
    async def process_upload(self, upload_type: UploadType, raw_data: bytes) -> UploadResult:
        if upload_type == UploadType.PORTFOLIO:
            try:
                workspace_id, strategy_id, latest_holding_count = await create_workspace_from_bytes(
                    data=raw_data,
                    name=self.upload.filename,  # type: ignore
                    user_id=self.user_id,
                    content_type=(
                        self.upload.content_type if self.upload.content_type else "text/csv"
                    ),
                )

                if not workspace_id or not strategy_id:
                    raise Exception(
                        f"Invalid workspace or strategy ID {workspace_id=} {strategy_id=}"
                    )
            except Exception:
                logger.exception(f"Failed to create portfolio for {self.user_id}")
                return UploadResult(
                    message="Sorry, I ran into some issues while importing this portfolio."
                )

            return UploadResult(
                message="Based on this file, I've created a portfolio for you"
                + f" named '{self.upload.filename}' with {latest_holding_count}"
                + f" holding{'s' if latest_holding_count != 1 else ''}."
            )

        if upload_type == UploadType.WATCHLIST:
            try:
                watchlist_id = await create_watchlist_from_bytes(
                    raw_data,
                    self.upload.filename,  # type: ignore
                    content_type=(
                        self.upload.content_type if self.upload.content_type else "text/csv"
                    ),
                    user_id=self.user_id,
                    jwt=self.jwt,
                )

                if not watchlist_id:
                    raise Exception("Invalid watchlist ID")

                logger.info(f"Created watchlist '{self.upload.filename}' for {self.user_id=}")

            except Exception:
                logger.exception(f"Failed to create watchlist for {self.user_id}")
                return UploadResult(
                    message="Sorry, I ran into issues while importing this watchlist."
                )

            return UploadResult(
                message=f"Based on this file, I've created a watchlist for you named '{self.upload.filename}'."
            )

    @async_perf_logger
    async def handle_upload(self) -> UploadResult:
        """
        Process an uploaded file, return True if processing was successful.
        """
        if not self.upload.filename:
            raise HTTPException(status_code=400, detail="Empty filename")

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

        raw_data = await self.read_upload_bytes()
        file_type = self.identify_file_type()
        upload_type, identify_message = self.identify_upload_type(file_type, raw_data)

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
        result = await self.process_upload(upload_type, raw_data)
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
