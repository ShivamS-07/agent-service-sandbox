import datetime
from typing import List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field

from agent_service.utils.date_utils import get_now_utc


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
# UpdateAgent
####################################################################################################
class UpdateAgentRequest(BaseModel):
    agent_id: str
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


####################################################################################################
# GetChatHistory
####################################################################################################
class GetChatHistoryRequest(BaseModel):
    agent_id: str
    # time window for the chat history
    start: Optional[datetime.datetime] = None  # if None, start from the beginning
    end: Optional[datetime.datetime] = None  # if None, end at the current time


class ChatMessage(BaseModel):
    agent_id: str
    message_id: str = Field(default_factory=lambda: str(uuid4()))
    message: str
    is_user_message: bool
    message_time: datetime.datetime = Field(default_factory=get_now_utc)


class GetChatHistoryResponse(BaseModel):
    messages: List[ChatMessage]  # sorted by message_time ASC
