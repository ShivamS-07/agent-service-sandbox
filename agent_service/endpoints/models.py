import datetime
from typing import List, Optional

from pydantic import BaseModel


####################################################################################################
# CreateAgent
####################################################################################################
class CreateAgentRequest(BaseModel):
    first_prompt: str


class CreateAgentResponse(BaseModel):
    agent_id: str


####################################################################################################
# DeleteAgent
####################################################################################################
class DeleteAgentRequest(BaseModel):
    agent_id: str


class DeleteAgentResponse(BaseModel):
    success: bool


####################################################################################################
# GetAllAgents
####################################################################################################
class AgentMetadata(BaseModel):
    agent_id: str
    user_id: Optional[str]
    agent_name: str
    created_at: datetime.datetime
    last_updated: datetime.datetime


class GetAllAgentsResponse(BaseModel):
    agents: List[AgentMetadata]
