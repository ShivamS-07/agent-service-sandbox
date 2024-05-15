import datetime
from typing import List, Optional

from pydantic import BaseModel

from agent_service.types import Message


####################################################################################################
# CreateAgent
####################################################################################################
class CreateAgentRequest(BaseModel):
    first_prompt: str


class CreateAgentResponse(BaseModel):
    success: bool
    allow_retry: bool
    agent_id: Optional[str] = None  # only set if success is True


####################################################################################################
# DeleteAgent
####################################################################################################


class DeleteAgentResponse(BaseModel):
    success: bool


####################################################################################################
# UpdateAgent
####################################################################################################
class UpdateAgentRequest(BaseModel):
    agent_name: str


class UpdateAgentResponse(BaseModel):
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


####################################################################################################
# ChatWithAgent
####################################################################################################
class ChatWithAgentRequest(BaseModel):
    agent_id: str
    prompt: str


class ChatWithAgentResponse(BaseModel):
    success: bool
    allow_retry: bool


####################################################################################################
# GetChatHistory
####################################################################################################
class GetChatHistoryResponse(BaseModel):
    messages: List[Message]  # sorted by message_time ASC