import datetime
from typing import List, Optional
from uuid import uuid4

from pydantic import BaseModel

from agent_service.io_type_utils import IOType

GPT_USER_TAG = "Client"
GPT_AGENT_TAG = "You"


class Message(BaseModel):
    content: IOType
    is_user: bool
    timestamp: datetime.datetime

    def get_gpt_input(self) -> str:
        tag = GPT_USER_TAG if self.is_user else GPT_AGENT_TAG
        return f"{tag}: {self.content}"


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
        )
