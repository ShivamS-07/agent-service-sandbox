import argparse
import asyncio
import datetime
import logging
from typing import Optional

import uvicorn
from fastapi import Depends, FastAPI, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.routing import APIRouter
from sse_starlette.sse import AsyncContentStream, EventSourceResponse, ServerSentEvent

from agent_service.agent_service_impl import AgentServiceImpl
from agent_service.endpoints.authz_helper import (
    User,
    get_keyid_to_key_map,
    parse_header,
    validate_user_agent_access,
    validate_user_plan_run_access,
)
from agent_service.endpoints.models import (
    ChatWithAgentRequest,
    ChatWithAgentResponse,
    CreateAgentResponse,
    DeleteAgentResponse,
    GetAgentOutputResponse,
    GetAgentTaskOutputResponse,
    GetAgentWorklogBoardResponse,
    GetAllAgentsResponse,
    GetChatHistoryResponse,
    SharePlanRunRequest,
    SharePlanRunResponse,
    UnsharePlanRunRequest,
    UnsharePlanRunResponse,
    UpdateAgentRequest,
    UpdateAgentResponse,
)
from agent_service.GPT.requests import _get_gpt_service_stub
from agent_service.utils.async_db import AsyncDB
from agent_service.utils.async_postgres_base import AsyncPostgresBase
from agent_service.utils.environment import EnvironmentUtils
from agent_service.utils.feature_flags import is_user_agent_admin
from agent_service.utils.logs import init_stdout_logging
from agent_service.utils.postgres import get_psql
from agent_service.utils.prefect_task_executor import PrefectTaskExecutor
from agent_service.utils.sentry_utils import init_sentry
from no_auth_endpoints import initialize_unauthed_endpoints

DEFAULT_IP = "0.0.0.0"
DEFAULT_DAL_PORT = 8000
SERVICE_NAME = "AgentService"


logger = logging.getLogger(__name__)


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
async def create_agent(user: User = Depends(parse_header)) -> CreateAgentResponse:
    return await application.state.agent_service_impl.create_agent(user=user)


@router.delete(
    "/agent/delete-agent/{agent_id}",
    response_model=DeleteAgentResponse,
    status_code=status.HTTP_200_OK,
)
async def delete_agent(agent_id: str, user: User = Depends(parse_header)) -> DeleteAgentResponse:
    validate_user_agent_access(user.user_id, agent_id)
    return await application.state.agent_service_impl.delete_agent(agent_id=agent_id)


@router.put(
    "/agent/update-agent/{agent_id}",
    response_model=UpdateAgentResponse,
    status_code=status.HTTP_200_OK,
)
async def update_agent(
    agent_id: str, req: UpdateAgentRequest, user: User = Depends(parse_header)
) -> UpdateAgentResponse:
    # NOTE: currently only allow updating agent name
    validate_user_agent_access(user.user_id, agent_id)

    return await application.state.agent_service_impl.update_agent(agent_id=agent_id, req=req)


@router.get(
    "/agent/get-all-agents", response_model=GetAllAgentsResponse, status_code=status.HTTP_200_OK
)
async def get_all_agents(user: User = Depends(parse_header)) -> GetAllAgentsResponse:
    return await application.state.agent_service_impl.get_all_agents(user=user)


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

    logger.info(f"Validating if user {user.user_id} has access to agent {req.agent_id}.")
    validate_user_agent_access(user.user_id, req.agent_id)
    return await application.state.agent_service_impl.chat_with_agent(req=req, user=user)


@router.get(
    "/agent/get-chat-history/{agent_id}",
    response_model=GetChatHistoryResponse,
    status_code=status.HTTP_200_OK,
)
async def get_chat_history(
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
    if not (user.is_super_admin or is_user_agent_admin(user.user_id)):
        validate_user_agent_access(user.user_id, agent_id)

    return await application.state.agent_service_impl.get_chat_history(
        agent_id=agent_id, start=start, end=end
    )


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
    if not (user.is_super_admin or is_user_agent_admin(user.user_id)):
        logger.info(f"Validating if user {user.user_id} has access to agent {agent_id}.")
        validate_user_agent_access(user.user_id, agent_id)

    return await application.state.agent_service_impl.get_agent_worklog_board(
        agent_id=agent_id,
        start_date=start_date,
        end_date=end_date,
        most_recent_num_run=most_recent_num_run,
    )


@router.get(
    "/agent/get-agent-task-output/{agent_id}/{plan_run_id}/{task_id}",
    response_model=GetAgentTaskOutputResponse,
    status_code=status.HTTP_200_OK,
)
async def get_agent_task_output(
    agent_id: str, plan_run_id: str, task_id: str, user: User = Depends(parse_header)
) -> GetAgentTaskOutputResponse:
    """Get the final outputs of a task once it's completed for Work Log Board

    Args:
        agent_id (str): agent ID
        plan_run_id (str): the run ID from Prefect
        task_id (str): the task ID of a run from Prefect
    """
    if not (user.is_super_admin or is_user_agent_admin(user.user_id)):
        validate_user_agent_access(user.user_id, agent_id)

    return await application.state.agent_service_impl.get_agent_task_output(
        agent_id=agent_id, plan_run_id=plan_run_id, task_id=task_id
    )


@router.get(
    "/agent/get-agent-log-output/{agent_id}/{plan_run_id}/{log_id}",
    response_model=GetAgentTaskOutputResponse,
    status_code=status.HTTP_200_OK,
)
async def get_agent_log_output(
    agent_id: str, plan_run_id: str, log_id: str, user: User = Depends(parse_header)
) -> GetAgentTaskOutputResponse:
    """Get the final outputs of a task once it's completed for Work Log Board

    Args:
        agent_id (str): agent ID
        plan_run_id (str): the run ID from Prefect
        task_id (str): the task ID of a run from Prefect
    """
    if not (user.is_super_admin or is_user_agent_admin(user.user_id)):
        validate_user_agent_access(user.user_id, agent_id)

    return await application.state.agent_service_impl.get_agent_log_output(
        agent_id=agent_id, plan_run_id=plan_run_id, log_id=log_id
    )


@router.get(
    "/agent/get-agent-output/{agent_id}",
    response_model=GetAgentOutputResponse,
    status_code=status.HTTP_200_OK,
)
async def get_agent_output(
    agent_id: str, user: User = Depends(parse_header)
) -> GetAgentOutputResponse:
    """Get agent's LATEST output - An agent can have many runs and we always want the latest output

    Args:
        agent_id (str): agent ID
    """
    if not (user.is_super_admin or is_user_agent_admin(user.user_id)):
        validate_user_agent_access(user.user_id, agent_id)

    return await application.state.agent_service_impl.get_agent_output(agent_id=agent_id)


@router.get(
    "/agent/stream/{agent_id}",
    status_code=status.HTTP_200_OK,
)
async def steam_agent_events(
    agent_id: str, user: User = Depends(parse_header)
) -> EventSourceResponse:
    """Set up a data stream that returns messages based on backend events.

    Args:
        agent_id (str): agent ID
    """
    if not (user.is_super_admin or is_user_agent_admin(user.user_id)):
        validate_user_agent_access(user.user_id, agent_id)

    async def _wrap_serializer() -> AsyncContentStream:
        try:
            async for event in application.state.agent_service_impl.stream_agent_events(
                agent_id=agent_id
            ):
                yield ServerSentEvent(data=event.model_dump_json(), event="agent-event")
        except asyncio.CancelledError as e:
            logger.info(f"Event stream client disconnected for {agent_id=}")
            raise e

    return EventSourceResponse(content=_wrap_serializer())


@router.post(
    "/agent/share-plan-run",
    response_model=SharePlanRunResponse,
    status_code=status.HTTP_200_OK,
)
async def share_plan_run(
    req: SharePlanRunRequest, user: User = Depends(parse_header)
) -> ChatWithAgentResponse:
    """Share agent plan run (set shared status to true)

    Args:
        plan_run_id (str): plan run ID
    """

    validate_user_plan_run_access(user.user_id, req.plan_run_id)
    return await application.state.agent_service_impl.share_plan_run(plan_run_id=req.plan_run_id)


@router.post(
    "/agent/unshare-plan-run",
    response_model=UnsharePlanRunResponse,
    status_code=status.HTTP_200_OK,
)
async def unshare_plan_run(
    req: UnsharePlanRunRequest, user: User = Depends(parse_header)
) -> ChatWithAgentResponse:
    """Unshare agent plan run (set shared status to false)

    Args:
        plan_run_id (str): plan run ID
    """

    validate_user_plan_run_access(user.user_id, req.plan_run_id)
    return await application.state.agent_service_impl.unshare_plan_run(plan_run_id=req.plan_run_id)


initialize_unauthed_endpoints(application)
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
    init_sentry(disable_sentry=not EnvironmentUtils.is_deployed)

    logger.info("Warming up DB connection and JWT key map...")
    get_psql()
    get_keyid_to_key_map()

    logger.info("Starting server...")
    application.state.agent_service_impl = AgentServiceImpl(
        task_executor=PrefectTaskExecutor(),
        gpt_service_stub=_get_gpt_service_stub()[0],
        async_db=AsyncDB(pg=AsyncPostgresBase()),
    )
    uvicorn.run(application, host=args.address, port=args.port)
