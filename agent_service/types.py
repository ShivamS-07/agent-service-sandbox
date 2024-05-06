import datetime

from pydantic import BaseModel


class ChatContext(BaseModel):
    pass


class PlanRunContext(BaseModel):
    # TODO contains all necessary ID's, as well as chat context
    agent_id: str
    plan_id: str
    user_id: str
    plan_run_id: str
    task_id: str

    chat: ChatContext
    current_date: datetime.date
