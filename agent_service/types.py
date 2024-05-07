import datetime
from typing import List

from pydantic import BaseModel

from agent_service.tools.io_type_utils import IOType

GPT_USER_TAG = "User"
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
    task_id: str

    chat: ChatContext


class ExecutionPlan(BaseModel):
    pass
