import base64
import enum
import logging
from typing import Optional, Tuple

from fastapi import UploadFile
from pa_portfolio_service_proto_v1.well_known_types_pb2 import StockHolding
from pydantic import BaseModel

from agent_service.external.dal_svc_client import (
    ParsePortfolioWorkspaceResponse,
    get_dal_client,
)
from agent_service.external.grpc_utils import datetime_to_timestamp
from agent_service.external.pa_svc_client import create_ts_workspace, recalc_strategies
from agent_service.types import Message
from agent_service.utils.agent_event_utils import send_chat_message
from agent_service.utils.async_db import AsyncDB
from agent_service.utils.logs import async_perf_logger

logger = logging.getLogger(__name__)


class UploadType(str, enum.Enum):
    PORTFOLIO = "portfolio"
    WATCHLIST = "watchlist"


class UploadResult(BaseModel):
    success: bool = True
    message: str = ""


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

    holdings = [
        StockHolding(
            gbi_id=security.gbi_id,
            date=(datetime_to_timestamp(security.date) if security.date else None),
            weight=security.weight,
        )
        for security in parsed_response.securities
    ]

    # get latest holding count
    latest_date = max(security.date for security in parsed_response.securities if security.date)
    latest_holding_count = len(
        [security for security in parsed_response.securities if security.date == latest_date]
    )

    # create the workspace with the holdings
    workspace_id, strategy_id = await create_ts_workspace(
        user_id=user_id, holdings=holdings, workspace_name=name
    )

    logger.info(f"Created workspace for {user_id=}, {workspace_id=} {strategy_id=}")

    # kick off a recalc
    await recalc_strategies(user_id=user_id, strategy_ids=[strategy_id])

    return workspace_id, strategy_id, latest_holding_count


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
    async def identify_upload_type(self) -> Tuple[UploadType, Optional[str]]:
        return (
            UploadType.PORTFOLIO,
            "This looks to be a portfolio holdings file, let me import that...",
        )

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
