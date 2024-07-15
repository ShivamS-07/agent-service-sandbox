import enum
import logging
from typing import Optional

from fastapi import UploadFile
from pydantic import BaseModel

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

    @async_perf_logger
    async def identify_upload_type(self) -> UploadType:
        return UploadType.PORTFOLIO

    @async_perf_logger
    async def process_upload(self, upload_type: UploadType) -> UploadResult:
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
        upload_type = await self.identify_upload_type()
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
                    message="Upload processing finished!",
                    visible_to_llm=False,
                ),
                self.db,
                insert_message_into_db=True,
                send_notification=False,
            )
        return result
