import argparse
import datetime
import logging
from typing import Optional
from uuid import uuid4

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.routing import APIRouter

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
    ExecutionPlanTemplate,
    GetAgentOutputResponse,
    GetAgentTaskOutputResponse,
    GetAgentWorklogBoardResponse,
    GetAgentWorklogOutputResponse,
    GetAllAgentsResponse,
    GetChatHistoryResponse,
    UpdateAgentRequest,
    UpdateAgentResponse,
)
from agent_service.endpoints.utils import get_agent_hierarchical_worklogs
from agent_service.io_type_utils import load_io_type
from agent_service.types import ChatContext, Message
from agent_service.utils.date_utils import get_now_utc
from agent_service.utils.environment import EnvironmentUtils
from agent_service.utils.logs import init_stdout_logging
from agent_service.utils.postgres import DEFAULT_AGENT_NAME, get_psql
from agent_service.utils.prefect import prefect_create_execution_plan
from agent_service.utils.sentry_utils import init_sentry

DEFAULT_IP = "0.0.0.0"
DEFAULT_DAL_PORT = 8000
SERVICE_NAME = "AgentService"

logger = logging.getLogger(__name__)

init_sentry(disable_sentry=not EnvironmentUtils.is_deployed)

application = FastAPI(title="Agent Service")
router = APIRouter(prefix="/api")
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
@router.get("/", response_class=HTMLResponse, status_code=200)
def confirm_working() -> str:
    return "<html>Agent Service is online</html>"


@router.get("/health")
def health() -> str:
    return "OK"


####################################################################################################
# Agent endpoints
####################################################################################################
@router.post(
    "/agent/create-agent", response_model=CreateAgentResponse, status_code=status.HTTP_201_CREATED
)
async def create_agent(
    req: CreateAgentRequest, user: User = Depends(parse_header)
) -> CreateAgentResponse:
    """Create an agent - Client should send the first prompt from user
    1. Generate initial response from GPT -> Allow retry if fails
    2. Insert agent and messages into DB -> Allow retry if fails
    3. Kick off Prefect job -> retry is forbidden since it's a major system issue
    4. Return success or failure to client. If success, return agent ID
    """

    now = get_now_utc()
    agent_id = str(uuid4())
    try:
        logger.info("Generating initial response from GPT...")
        agent = AgentMetadata(
            agent_id=agent_id,
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

    plan_id = str(uuid4())
    logger.info(f"Creating execution plan {plan_id} for {agent_id=}")
    try:
        await prefect_create_execution_plan(
            agent_id=agent_id, plan_id=plan_id, user_id=user.user_id, run_plan_immediately=True
        )
    except Exception:
        logger.exception("Failed to kick off execution plan creation")
        return CreateAgentResponse(success=False, allow_retry=False)

    return CreateAgentResponse(success=True, allow_retry=False, agent_id=agent.agent_id)


@router.delete(
    "/agent/delete-agent/{agent_id}",
    response_model=DeleteAgentResponse,
    status_code=status.HTTP_200_OK,
)
def delete_agent(agent_id: str, user: User = Depends(parse_header)) -> DeleteAgentResponse:
    validate_user_agent_access(user.user_id, agent_id)

    get_psql().delete_agent_by_id(agent_id)
    return DeleteAgentResponse(success=True)


@router.put(
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


@router.get(
    "/agent/get-all-agents", response_model=GetAllAgentsResponse, status_code=status.HTTP_200_OK
)
def get_all_agents(user: User = Depends(parse_header)) -> GetAllAgentsResponse:
    return GetAllAgentsResponse(agents=get_psql().get_user_all_agents(user.user_id))


@router.post(
    "/agent/chat-with-agent", response_model=ChatWithAgentResponse, status_code=status.HTTP_200_OK
)
async def chat_with_agent(
    req: ChatWithAgentRequest, user: User = Depends(parse_header)
) -> ChatWithAgentResponse:
    """Chat with agent - Client should send a prompt from user
    1. Validate user has access to agent
    2. Generate initial response from GPT -> Allow retry if fails
    3. Insert user message and GPT response into DB -> Allow retry if fails
    4. Kick off Prefect job -> retry is forbidden since it's a major system issue
    5. Return success or failure to client
    """

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

    # kick off Prefect job -> retry is forbidden
    # TODO we should check if we NEED to create a new plan
    plan_id = str(uuid4())
    logger.info(f"Creating execution plan {plan_id} for {req.agent_id=}")
    try:
        await prefect_create_execution_plan(
            agent_id=req.agent_id, plan_id=plan_id, user_id=user.user_id, run_plan_immediately=True
        )
    except Exception:
        logger.exception("Failed to kick off execution plan creation")
        return ChatWithAgentResponse(success=False, allow_retry=False)

    return ChatWithAgentResponse(success=True, allow_retry=False)


@router.get(
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
    """Get chat history for an agent

    Args:
        agent_id (str): agent ID
        start (Optional[datetime.datetime]): start time to filter messages, inclusive
        end (Optional[datetime.datetime]): end time to filter messages, inclusive
        user (User): User object from `parse_header`
    """
    validate_user_agent_access(user.user_id, agent_id)
    chat_context = get_psql().get_chats_history_for_agent(agent_id, start, end)
    return GetChatHistoryResponse(messages=chat_context.messages)


@router.get(
    "/agent/get-agent-worklog-board/{agent_id}",
    response_model=GetAgentWorklogBoardResponse,
    status_code=status.HTTP_200_OK,
)
async def get_agent_worklog_board(
    agent_id: str,
    start_date: Optional[datetime.date] = None,
    end_date: Optional[datetime.date] = None,
    most_recent_num_run: Optional[int] = None,
    user: User = Depends(parse_header),
) -> GetAgentWorklogBoardResponse:
    """Get agent worklogs to build the Work Log Board
    Except `agent_id`, all other arguments are optional and can be used to filter the work log, but
    strongly recommend to have at least 1 filter to avoid returning too many entries.
    NOTE: If any of `start_date` or `end_date` is provided, and `most_recent_num` is also provided,
    it will be most recent N entries within the date range.

    Args:
        agent_id (str): agent ID
        start (Optional[datetime.date]): start DATE to filter work log, inclusive
        end (Optional[datetime.date]): end DATE to filter work log, inclusive
        most_recent_num_run (Optional[int]): number of most recent plan runs to return
    """
    logger.info(f"Validating if user {user.user_id} has access to agent {agent_id}.")
    validate_user_agent_access(user.user_id, agent_id)

    run_history = await get_agent_hierarchical_worklogs(
        agent_id, start_date, end_date, most_recent_num_run
    )

    # TODO: For now just get the latest plan. Later we can switch to LIVE plan
    plan_id, execution_plan = get_psql().get_latest_execution_plan(agent_id)
    if plan_id is None or execution_plan is None:
        execution_plan_template = None
    else:
        execution_plan_template = ExecutionPlanTemplate(
            plan_id=plan_id, task_names=[node.description for node in execution_plan.nodes]
        )

    return GetAgentWorklogBoardResponse(
        run_history=run_history, execution_plan_template=execution_plan_template
    )


@router.get(
    "/agent/get-agent-worklog-output/{agent_id}/{log_id}",
    response_model=GetAgentWorklogOutputResponse,
    status_code=status.HTTP_200_OK,
)
def get_agent_worklog_output(
    agent_id: str, log_id: str, user: User = Depends(parse_header)
) -> GetAgentWorklogOutputResponse:
    validate_user_agent_access(user.user_id, agent_id)
    rows = get_psql().get_log_data_from_log_id(agent_id, log_id)
    if not rows:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"{log_id} not found")
    log_data = rows[0]["log_data"]
    output = load_io_type(log_data) if log_data is not None else None

    return GetAgentWorklogOutputResponse(output=output)


@router.get(
    "/agent/get-agent-task-output/{agent_id}/{plan_run_id}/{task_id}",
    response_model=GetAgentTaskOutputResponse,
    status_code=status.HTTP_200_OK,
)
def get_agent_task_output(
    agent_id: str, plan_run_id: str, task_id: str, user: User = Depends(parse_header)
) -> GetAgentTaskOutputResponse:
    """Get the final outputs of a task once it's completed for Work Log Board

    Args:
        agent_id (str): agent ID
        plan_run_id (str): the run ID from Prefect
        task_id (str): the task ID of a run from Prefect
    """
    validate_user_agent_access(user.user_id, agent_id)

    task_output = get_psql().get_task_output(
        agent_id=agent_id, plan_run_id=plan_run_id, task_id=task_id
    )

    return GetAgentTaskOutputResponse(log_data=task_output)


@router.get(
    "/agent/get-agent-output/{agent_id}",
    response_model=GetAgentOutputResponse,
    status_code=status.HTTP_200_OK,
)
def get_agent_output(agent_id: str, user: User = Depends(parse_header)) -> GetAgentOutputResponse:
    """Get agent's LATEST output - An agent can have many runs and we always want the latest output

    Args:
        agent_id (str): agent ID
    """
    validate_user_agent_access(user.user_id, agent_id)
    outputs = get_psql().get_agent_outputs(agent_id=agent_id)
    if not outputs:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"No output found for {agent_id=}"
        )

    final_outputs = [output for output in outputs if not output.is_intermediate]
    if final_outputs:
        return GetAgentOutputResponse(outputs=final_outputs)

    return GetAgentOutputResponse(outputs=outputs)


application.include_router(router)


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
