import datetime
from typing import List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field

from agent_service.io_type_utils import IOType
from agent_service.utils.date_utils import get_now_utc

GPT_USER_TAG = "Client"
GPT_AGENT_TAG = "You"


class Message(BaseModel):
    agent_id: str = Field(default_factory=lambda: str(uuid4()))  # default is for testing only
    message_id: str = Field(default_factory=lambda: str(uuid4()))
    message: IOType
    is_user_message: bool
    message_time: datetime.datetime = Field(default_factory=get_now_utc)

    def get_gpt_input(self) -> str:
        tag = GPT_USER_TAG if self.is_user_message else GPT_AGENT_TAG
        return f"{tag}: {self.message}"


class Notification(BaseModel):
    notification_id: str = Field(default_factory=lambda: str(uuid4()))
    agent_id: str
    message_id: Optional[str] = None
    summary: Optional[str] = None
    unread: bool = True
    created_at: datetime.datetime = Field(default_factory=get_now_utc)


class ChatContext(BaseModel):
    messages: List[Message]

    def get_gpt_input(self) -> str:
        return "\n".join([message.get_gpt_input() for message in self.messages])


class PlanRunContext(BaseModel):
    # TODO contains all necessary ID's, as well as chat context
    agent_id: str
    plan_id: str
    user_id: str
    plan_run_id: str

    # Can be filled in by whoever needs to avoid passing around large data
    chat: Optional[ChatContext] = None

    # Only populated before each task run
    task_id: Optional[str] = None

    # Useful for testing, etc.
    skip_db_commit: bool = False
    skip_task_cache: bool = False
    run_tasks_without_prefect: bool = False

    @staticmethod
    def get_dummy() -> "PlanRunContext":
        return PlanRunContext(
            agent_id=str(uuid4()),
            plan_id=str(uuid4()),
            user_id=str(uuid4()),
            task_id=str(uuid4()),
            plan_run_id=str(uuid4()),
            skip_db_commit=True,
            skip_task_cache=True,
            run_tasks_without_prefect=True,
        )
