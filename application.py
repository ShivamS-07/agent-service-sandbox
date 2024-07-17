import argparse
import asyncio
import dataclasses
import datetime
import json
import logging
import time
import traceback
import uuid
from typing import Any, Callable, Dict, Optional

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.routing import APIRouter
from gbi_common_py_utils.utils.environment import (
    PROD_TAG,
    STAGING_TAG,
    get_environment_tag,
)
from gbi_common_py_utils.utils.event_logging import log_event
from sse_starlette.sse import AsyncContentStream, EventSourceResponse, ServerSentEvent
from starlette.requests import Request

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
    DisableAgentAutomationRequest,
    DisableAgentAutomationResponse,
    EnableAgentAutomationRequest,
    EnableAgentAutomationResponse,
    GetAgentDebugInfoResponse,
    GetAgentOutputResponse,
    GetAgentTaskOutputResponse,
    GetAgentWorklogBoardResponse,
    GetAllAgentsResponse,
    GetChatHistoryResponse,
    GetSecureUserResponse,
    GetTestCaseInfoResponse,
    GetTestSuiteRunInfoResponse,
    GetTestSuiteRunsIdsResponse,
    MarkNotificationsAsReadRequest,
    MarkNotificationsAsReadResponse,
    SharePlanRunRequest,
    SharePlanRunResponse,
    UnsharePlanRunRequest,
    UnsharePlanRunResponse,
    UpdateAgentRequest,
    UpdateAgentResponse,
    UploadFileResponse,
)
from agent_service.external.grpc_utils import create_jwt
from agent_service.GPT.requests import _get_gpt_service_stub
from agent_service.utils.async_db import AsyncDB
from agent_service.utils.async_postgres_base import AsyncPostgresBase
from agent_service.utils.clickhouse import Clickhouse
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

REQUEST_COUNTER = int(time.time())


@dataclasses.dataclass
class AuditInfo:
    path: str
    internal_request_id: str
    received_timestamp: datetime.datetime
    user_id: Optional[str] = None
    request_body: Optional[Dict[str, Any]] = None

    response_timestamp: Optional[datetime.datetime] = None
    internal_processing_time: Optional[float] = None
    total_processing_time: Optional[float] = None
    error: Optional[str] = None
    client_timestamp: Optional[str] = None
    client_request_id: Optional[str] = None
    request_number: int = -1

    def to_json_dict(self) -> Dict[str, Any]:
        data = dataclasses.asdict(self)

        return {key: value for key, value in data.items() if value is not None}


def update_audit_info_with_response_info(
    audit_info: AuditInfo, received_timestamp: datetime.datetime
) -> None:
    response_timestamp = datetime.datetime.utcnow()
    audit_info.response_timestamp = response_timestamp
    audit_info.internal_processing_time = (response_timestamp - received_timestamp).total_seconds()
    if audit_info.client_timestamp:
        audit_info.total_processing_time = (
            response_timestamp - datetime.datetime.fromisoformat(audit_info.client_timestamp)
        ).total_seconds()


@application.middleware("http")
async def add_process_time_header(request: Request, call_next: Callable) -> Any:
    received_timestamp = datetime.datetime.utcnow()
    global REQUEST_COUNTER
    REQUEST_COUNTER += 1
    client_timestamp = request.headers.get("clienttimestamp", None)
    if client_timestamp:
        client_timestamp = (
            client_timestamp[:-1] if client_timestamp.endswith("Z") else client_timestamp
        )

    audit_info = AuditInfo(
        path=request.url.path,
        internal_request_id=str(uuid.uuid4()),
        received_timestamp=received_timestamp,
        client_timestamp=client_timestamp,
        client_request_id=request.headers.get("clientrequestid", None),
        request_number=REQUEST_COUNTER,
    )
    try:
        authorization = request.headers.get("Authorization", None)
        if authorization:
            user_info = parse_header(request=request, auth_token=authorization)
            request.state.user_info = user_info
            audit_info.user_id = user_info.user_id
        request_body = await request.body()
        audit_info.request_body = json.loads(request_body) if request_body else None
    except Exception:
        audit_info.error = traceback.format_exc()

    try:
        response = await call_next(request)
    except Exception as e:
        error = traceback.format_exc()
        audit_info.error = audit_info.error + "/n" + error if audit_info.error else error
        update_audit_info_with_response_info(
            audit_info=audit_info, received_timestamp=received_timestamp
        )
        log_event(event_name="AgentService-RequestError", event_data=audit_info.to_json_dict())
        raise e
    update_audit_info_with_response_info(
        audit_info=audit_info, received_timestamp=received_timestamp
    )
    log_event(event_name="AgentService-RequestCompleted", event_data=audit_info.to_json_dict())
    return response


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
    req: ChatWithAgentRequest,
    user: User = Depends(parse_header),
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


@router.post(
    "/agent/upload-file/{agent_id}",
    response_model=UploadFileResponse,
    status_code=status.HTTP_200_OK,
)
async def upload_file(
    agent_id: str,
    upload: UploadFile,
    user: User = Depends(parse_header),
) -> ChatWithAgentResponse:
    validate_user_agent_access(user.user_id, agent_id)
    return await application.state.agent_service_impl.upload_file(
        upload=upload, user=user, agent_id=agent_id
    )


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
    "/agent/get-agent-output/{agent_id}/{plan_run_id}",
    response_model=GetAgentOutputResponse,
    status_code=status.HTTP_200_OK,
)
async def get_agent_plan_output(
    agent_id: str, plan_run_id: str, user: User = Depends(parse_header)
) -> GetAgentOutputResponse:
    """Get agent's output for a specific plan_run_id

    Args:
        agent_id (str): agent ID
        plan_run_id (str): plan run ID
    """
    if not (user.is_super_admin or is_user_agent_admin(user.user_id)):
        validate_user_agent_access(user.user_id, agent_id)

    return await application.state.agent_service_impl.get_agent_plan_output(
        agent_id=agent_id, plan_run_id=plan_run_id
    )


@router.get(
    "/agent/stream/{agent_id}",
    status_code=status.HTTP_200_OK,
)
async def stream_agent_events(
    request: Request, agent_id: str, user: User = Depends(parse_header)
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
                request=request, agent_id=agent_id
            ):
                try:
                    to_send = event.model_dump_json()
                except Exception:
                    logger.exception(f"Error while sending agent event {agent_id=}")
                    log_event(
                        event_name="agent-event-sent",
                        event_data={
                            "agent_id": agent_id,
                            "user_id": user.user_id,
                            "error_msg": traceback.format_exc(),
                        },
                    )
                    continue

                yield ServerSentEvent(data=to_send, event="agent-event")
                log_event(
                    event_name="agent-event-sent",
                    event_data={"agent_id": agent_id, "user_id": user.user_id, "data": to_send},
                )
        except Exception as e:
            log_event(
                event_name="agent-event-sent",
                event_data={
                    "agent_id": agent_id,
                    "user_id": user.user_id,
                    "error_msg": traceback.format_exc(),
                },
            )
            logger.info(f"Event stream client disconnected for {agent_id=}")
            raise e

    return EventSourceResponse(content=_wrap_serializer())


@router.get(
    "/notifications/stream",
    status_code=status.HTTP_200_OK,
)
async def stream_notification_events(
    request: Request, user: User = Depends(parse_header)
) -> EventSourceResponse:
    """
    Set up a data stream that returns messages based on notification events.
    """

    async def _wrap_serializer() -> AsyncContentStream:
        try:
            async for event in application.state.agent_service_impl.stream_notification_events(
                request=request, user_id=user.user_id
            ):
                try:
                    yield ServerSentEvent(data=event.model_dump_json(), event="notification-event")
                except Exception:
                    logger.exception(f"Error while sending notification event {user.user_id=}")

        except asyncio.CancelledError as e:
            logger.info(f"Event stream client disconnected for {user.user_id=}")
            raise e

    return EventSourceResponse(content=_wrap_serializer())


@router.post(
    "/agent/share-plan-run",
    response_model=SharePlanRunResponse,
    status_code=status.HTTP_200_OK,
)
async def share_plan_run(
    req: SharePlanRunRequest, user: User = Depends(parse_header)
) -> SharePlanRunResponse:
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
) -> UnsharePlanRunResponse:
    """Unshare agent plan run (set shared status to false)

    Args:
        plan_run_id (str): plan run ID
    """

    validate_user_plan_run_access(user.user_id, req.plan_run_id)
    return await application.state.agent_service_impl.unshare_plan_run(plan_run_id=req.plan_run_id)


@router.post(
    "/agent/mark-notifications-as-read",
    response_model=MarkNotificationsAsReadResponse,
    status_code=status.HTTP_200_OK,
)
async def mark_notifications_as_read(
    req: MarkNotificationsAsReadRequest, user: User = Depends(parse_header)
) -> MarkNotificationsAsReadResponse:
    """Mark all agent notifications as read

    Args:
        agent_id (str): agent ID
        timestamp (optional int): int representing timestamp in UTC
    """
    validate_user_agent_access(user.user_id, req.agent_id)
    return await application.state.agent_service_impl.mark_notifications_as_read(
        req.agent_id, req.timestamp
    )


@router.post(
    "/agent/enable-automation",
    response_model=EnableAgentAutomationResponse,
    status_code=status.HTTP_200_OK,
)
async def enable_agent_automation(
    req: EnableAgentAutomationRequest, user: User = Depends(parse_header)
) -> EnableAgentAutomationResponse:
    """
    Enable agent automation

    Args:
        agent_id (str): agent ID
    """
    validate_user_agent_access(user.user_id, req.agent_id)
    return await application.state.agent_service_impl.enable_agent_automation(agent_id=req.agent_id)


@router.post(
    "/agent/disable-automation",
    response_model=DisableAgentAutomationResponse,
    status_code=status.HTTP_200_OK,
)
async def disable_agent_automation(
    req: DisableAgentAutomationRequest, user: User = Depends(parse_header)
) -> DisableAgentAutomationResponse:
    """
    Disable agent automation

    Args:
        agent_id (str): agent ID
    """
    validate_user_agent_access(user.user_id, req.agent_id)
    return await application.state.agent_service_impl.disable_agent_automation(
        agent_id=req.agent_id
    )


@router.get(
    "/feature-flag/get-secure-user",
    status_code=status.HTTP_200_OK,
)
async def get_secure_ld_user(user: User = Depends(parse_header)) -> GetSecureUserResponse:
    """
    Get a secure mode hash and LD user context.
    """
    return application.state.agent_service_impl.get_secure_ld_user(user_id=user.user_id)


@router.get(
    "/debug/{agent_id}",
    response_model=GetAgentDebugInfoResponse,
    status_code=status.HTTP_200_OK,
)
async def get_agent_debug_info(
    agent_id: str, user: User = Depends(parse_header)
) -> GetAgentDebugInfoResponse:
    if not user.is_super_admin and not is_user_agent_admin(user.user_id):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="User is not authorized"
        )
    return await application.state.agent_service_impl.get_agent_debug_info(agent_id=agent_id)


@router.get(
    "/regression-test/{test_run_id}",
    response_model=GetTestSuiteRunInfoResponse,
    status_code=status.HTTP_200_OK,
)
def get_info_for_test_run_id(
    test_run_id: str, user: User = Depends(parse_header)
) -> GetTestSuiteRunInfoResponse:
    if get_environment_tag() in [STAGING_TAG, PROD_TAG]:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="")
    if not user.is_super_admin and not is_user_agent_admin(user.user_id):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="User is not authorized"
        )
    return application.state.agent_service_impl.get_info_for_test_suite_run(test_run_id=test_run_id)


@router.get(
    "/regression-test-suite-run-ids",
    response_model=GetTestSuiteRunsIdsResponse,
    status_code=status.HTTP_200_OK,
)
def get_test_suite_runs(user: User = Depends(parse_header)) -> GetTestSuiteRunsIdsResponse:
    if get_environment_tag() in [STAGING_TAG, PROD_TAG]:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="")
    if not user.is_super_admin and not is_user_agent_admin(user.user_id):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="User is not authorized"
        )
    return application.state.agent_service_impl.get_test_suite_runs()


@router.get(
    "/regression-test-case/{test_name}",
    response_model=GetTestCaseInfoResponse,
    status_code=status.HTTP_200_OK,
)
def get_info_for_test_case(
    test_name: str, user: User = Depends(parse_header)
) -> GetTestCaseInfoResponse:
    if get_environment_tag() in [STAGING_TAG, PROD_TAG]:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="")
    if not user.is_super_admin and not is_user_agent_admin(user.user_id):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="User is not authorized"
        )
    return application.state.agent_service_impl.get_info_for_test_case(test_name=test_name)


# Account Management Enpoints


@router.get("/account-management/generate-jwt/{user_id}", status_code=status.HTTP_200_OK)
async def generate_jwt(user_id: str, user: User = Depends(parse_header)) -> str:
    if (
        "manual-account-override-basic" not in user.groups
        and "manual-account-override-full" not in user.groups
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="User is not authorized"
        )
    return create_jwt(user_id=user_id, expiry_hours=1, include_aud=True)


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
        clickhouse_db=Clickhouse(),
    )
    uvicorn.run(application, host=args.address, port=args.port)
