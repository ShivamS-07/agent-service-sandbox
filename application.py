import argparse
import datetime
import logging
from typing import Optional
from uuid import uuid4

import uvicorn
from fastapi import Depends, FastAPI, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from agent_service.chatbot.chatbot import Chatbot
from agent_service.endpoints.authz_helper import (
    User,
    get_keyid_to_key_map,
    parse_header,
    validate_user_agent_access,
)
from agent_service.endpoints.models import (
    AgentMetadata,
    ChatWithAgentRequest,
    ChatWithAgentResponse,
    CreateAgentRequest,
    CreateAgentResponse,
    DeleteAgentResponse,
    GetAllAgentsResponse,
    GetChatHistoryResponse,
    UpdateAgentRequest,
    UpdateAgentResponse,
)
from agent_service.types import ChatContext, Message
from agent_service.utils.date_utils import get_now_utc
from agent_service.utils.logs import init_stdout_logging
from agent_service.utils.postgres import DEFAULT_AGENT_NAME, get_psql

DEFAULT_IP = "0.0.0.0"
DEFAULT_DAL_PORT = 8000
SERVICE_NAME = "AgentService"

logger = logging.getLogger(__name__)

application = FastAPI(title="Agent Service")
application.add_middleware(
    CORSMiddleware,  # Add CORS middleware
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


####################################################################################################
# Test endpoints
####################################################################################################
@application.get("/", response_class=HTMLResponse, status_code=200)
def confirm_working() -> str:
    return "<html>Agent Service is online</html>"


@application.get("/health")
def health() -> str:
    return "OK"


####################################################################################################
# Agent endpoints
####################################################################################################
@application.post(
    "/agent/create-agent", response_model=CreateAgentResponse, status_code=status.HTTP_201_CREATED
)
async def create_agent(
    req: CreateAgentRequest, user: User = Depends(parse_header)
) -> CreateAgentResponse:
    now = get_now_utc()

    try:
        logger.info("Generating initial response from GPT...")
        agent = AgentMetadata(
            agent_id=str(uuid4()),
            user_id=user.user_id,
            agent_name=DEFAULT_AGENT_NAME,
            created_at=now,
            last_updated=now,
        )
        user_msg = Message(
            agent_id=agent.agent_id,
            message=req.first_prompt,
            is_user_message=True,
            message_time=now,
        )
        chatbot = Chatbot(agent.agent_id)
        gpt_resp = await chatbot.generate_initial_preplan_response(
            chat_context=ChatContext(messages=[user_msg])
        )
        gpt_msg = Message(
            agent_id=agent.agent_id,
            message=gpt_resp,
            is_user_message=False,
        )
    except Exception as e:
        # FE should retry if this fails
        logger.exception(f"Failed to generate initial response from GPT with exception: {e}")
        return CreateAgentResponse(success=False, allow_retry=True)

    try:
        logger.info(f"Inserting agent and messages into DB for agent {agent.agent_id}...")
        get_psql().insert_agent_and_messages(agent_metadata=agent, messages=[user_msg, gpt_msg])
    except Exception as e:
        logger.exception(f"Failed to insert agent and messages into DB with exception: {e}")
        return CreateAgentResponse(success=False, allow_retry=True)

    # TODO: kick off Prefect job -> retry is forbidden

    return CreateAgentResponse(success=True, allow_retry=False, agent_id=agent.agent_id)


@application.delete(
    "/agent/delete-agent/{agent_id}",
    response_model=DeleteAgentResponse,
    status_code=status.HTTP_200_OK,
)
def delete_agent(agent_id: str, user: User = Depends(parse_header)) -> DeleteAgentResponse:
    validate_user_agent_access(user.user_id, agent_id)

    get_psql().delete_agent_by_id(agent_id)
    return DeleteAgentResponse(success=True)


@application.put(
    "/agent/update-agent/{agent_id}",
    response_model=UpdateAgentResponse,
    status_code=status.HTTP_200_OK,
)
def update_agent(
    agent_id: str, req: UpdateAgentRequest, user: User = Depends(parse_header)
) -> UpdateAgentResponse:
    # NOTE: currently only allow updating agent name
    validate_user_agent_access(user.user_id, agent_id)

    get_psql().update_agent_name(agent_id, req.agent_name)
    return UpdateAgentResponse(success=True)


@application.get(
    "/agent/get-all-agents", response_model=GetAllAgentsResponse, status_code=status.HTTP_200_OK
)
def get_all_agents(user: User = Depends(parse_header)) -> GetAllAgentsResponse:
    return GetAllAgentsResponse(agents=get_psql().get_user_all_agents(user.user_id))


@application.post(
    "/agent/chat-with-agent", response_model=ChatWithAgentResponse, status_code=status.HTTP_200_OK
)
async def chat_with_agent(
    req: ChatWithAgentRequest, user: User = Depends(parse_header)
) -> ChatWithAgentResponse:
    now = get_now_utc()

    logger.info(f"Validating if user {user.user_id} has access to agent {req.agent_id}.")
    validate_user_agent_access(user.user_id, req.agent_id)

    try:
        logger.info("Generating initial response from GPT...")
        user_msg = Message(
            agent_id=req.agent_id,
            message=req.prompt,
            is_user_message=True,
            message_time=now,
        )
        chatbot = Chatbot(req.agent_id)
        gpt_resp = await chatbot.generate_initial_preplan_response(
            chat_context=ChatContext(messages=[user_msg])
        )
        gpt_msg = Message(
            agent_id=req.agent_id,
            message=gpt_resp,
            is_user_message=False,
        )
    except Exception as e:
        logger.exception(f"Failed to generate initial response from GPT with exception: {e}")
        return ChatWithAgentResponse(success=False, allow_retry=True)

    try:
        logger.info(f"Inserting user message and GPT response into DB for agent {req.agent_id}...")
        get_psql().insert_chat_messages(messages=[user_msg, gpt_msg])
    except Exception as e:
        logger.exception(f"Failed to insert messages into DB with exception: {e}")
        return ChatWithAgentResponse(success=False, allow_retry=True)

    # TODO: kick off Prefect job -> retry is forbidden

    return ChatWithAgentResponse(success=True, allow_retry=False)


@application.get(
    "/agent/get-chat-history/{agent_id}",
    response_model=GetChatHistoryResponse,
    status_code=status.HTTP_200_OK,
)
def get_chat_history(
    agent_id: str,
    start: Optional[datetime.datetime] = None,
    end: Optional[datetime.datetime] = None,
    user: User = Depends(parse_header),
) -> GetChatHistoryResponse:
    validate_user_agent_access(user.user_id, agent_id)
    chat_context = get_psql().get_chats_history_for_agent(agent_id, start, end)
    return GetChatHistoryResponse(messages=chat_context.messages)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--port",
        default=8000,
        type=int,
        required=False,
        help="Port to run the Agent Service on.",
    )
    parser.add_argument(
        "-a",
        "--address",
        type=str,
        required=False,
        default="0.0.0.0",
        help="Address to bind the server to.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    init_stdout_logging()

    logger.info("Warming up DB connection and JWT key map...")
    get_psql()
    get_keyid_to_key_map()

    logger.info("Starting server...")
    uvicorn.run(application, host=args.address, port=args.port)
