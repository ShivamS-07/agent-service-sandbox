from typing import Optional

from pydantic import BaseModel

from agent_service.types import ChatContext


class QAContext(BaseModel):
    agent_id: str
    plan_run_id: str
    user_id: Optional[str]
    chat_context: ChatContext
    gpt_context: Optional[dict[str, str]] = None
