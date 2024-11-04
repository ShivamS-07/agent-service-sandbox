import argparse
import asyncio
import dataclasses
import datetime
import json
import logging
import os
import time
import traceback
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Dict, List, Optional

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Response, UploadFile, status
from fastapi.datastructures import State
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
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
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from agent_service.agent_service_impl import AgentServiceImpl
from agent_service.endpoints.authz_helper import (
    User,
    get_keyid_to_key_map,
    parse_header,
    validate_user_agent_access,
    validate_user_plan_run_access,
)
from agent_service.endpoints.models import (
    AgentInfo,
    AgentQCResponse,
    ChatWithAgentRequest,
    ChatWithAgentResponse,
    ConvertMarkdownRequest,
    CopyAgentToUsersRequest,
    CopyAgentToUsersResponse,
    CreateAgentRequest,
    CreateAgentResponse,
    CreateCustomNotificationRequest,
    CreatePromptTemplateRequest,
    CreatePromptTemplateResponse,
    CreateSectionRequest,
    CreateSectionResponse,
    CustomNotification,
    CustomNotificationStatusResponse,
    DeleteAgentOutputRequest,
    DeleteAgentOutputResponse,
    DeleteAgentResponse,
    DeleteMemoryResponse,
    DeletePromptTemplateRequest,
    DeletePromptTemplateResponse,
    DeleteSectionRequest,
    DeleteSectionResponse,
    DisableAgentAutomationRequest,
    DisableAgentAutomationResponse,
    EnableAgentAutomationRequest,
    EnableAgentAutomationResponse,
    GenTemplatePlanRequest,
    GenTemplatePlanResponse,
    GetAccountInfoResponse,
    GetAgentDebugInfoResponse,
    GetAgentFeedBackResponse,
    GetAgentOutputResponse,
    GetAgentsQCRequest,
    GetAgentTaskOutputResponse,
    GetAgentWorklogBoardResponse,
    GetAllAgentsResponse,
    GetAutocompleteItemsRequest,
    GetAutocompleteItemsResponse,
    GetAvailableVariablesResponse,
    GetCannedPromptsResponse,
    GetChatHistoryResponse,
    GetCompaniesResponse,
    GetCustomDocumentFileInfoResponse,
    GetDebugToolArgsResponse,
    GetDebugToolResultResponse,
    GetLiveAgentsQCResponse,
    GetMemoryContentResponse,
    GetOrderedSecuritiesRequest,
    GetOrderedSecuritiesResponse,
    GetPlanRunDebugInfoResponse,
    GetPromptTemplatesResponse,
    GetSecureUserResponse,
    GetTeamAccountsResponse,
    GetTestCaseInfoResponse,
    GetTestCasesResponse,
    GetTestSuiteRunInfoResponse,
    GetTestSuiteRunsResponse,
    GetToolLibraryResponse,
    GetUsersRequest,
    GetUsersResponse,
    GetVariableCoverageRequest,
    GetVariableCoverageResponse,
    GetVariableHierarchyResponse,
    ListCustomDocumentsResponse,
    ListMemoryItemsResponse,
    LockAgentOutputRequest,
    LockAgentOutputResponse,
    MarkNotificationsAsReadRequest,
    MarkNotificationsAsReadResponse,
    MarkNotificationsAsUnreadRequest,
    MarkNotificationsAsUnreadResponse,
    ModifyPlanRunArgsRequest,
    ModifyPlanRunArgsResponse,
    NotificationEmailsResponse,
    RearrangeSectionRequest,
    RearrangeSectionResponse,
    RenameMemoryRequest,
    RenameMemoryResponse,
    RenameSectionRequest,
    RenameSectionResponse,
    RestoreAgentResponse,
    RunTemplatePlanRequest,
    RunTemplatePlanResponse,
    SearchAgentQCRequest,
    SearchAgentQCResponse,
    SetAgentFeedBackRequest,
    SetAgentFeedBackResponse,
    SetAgentScheduleRequest,
    SetAgentScheduleResponse,
    SetAgentSectionRequest,
    SetAgentSectionResponse,
    SetPromptTemplateVisibilityRequest,
    SetPromptTemplateVisibilityResponse,
    SharePlanRunRequest,
    SharePlanRunResponse,
    TerminateAgentRequest,
    TerminateAgentResponse,
    UnlockAgentOutputRequest,
    UnlockAgentOutputResponse,
    UnsharePlanRunRequest,
    UnsharePlanRunResponse,
    UpdateAgentDraftStatusRequest,
    UpdateAgentDraftStatusResponse,
    UpdateAgentQCRequest,
    UpdateAgentQCResponse,
    UpdateAgentRequest,
    UpdateAgentResponse,
    UpdateNotificationEmailsRequest,
    UpdateNotificationEmailsResponse,
    UpdatePromptTemplateRequest,
    UpdatePromptTemplateResponse,
    UpdateUserRequest,
    UpdateUserResponse,
    UploadFileResponse,
    UserHasAccessResponse,
)
from agent_service.external.grpc_utils import create_jwt
from agent_service.GPT.requests import _get_gpt_service_stub
from agent_service.io_types.citations import CitationType, GetCitationDetailsResponse
from agent_service.slack.slack_sender import SlackSender
from agent_service.utils.async_db import AsyncDB
from agent_service.utils.async_postgres_base import AsyncPostgresBase
from agent_service.utils.cache_utils import get_redis_cache_backend_for_output
from agent_service.utils.clickhouse import Clickhouse
from agent_service.utils.custom_documents_utils import CustomDocumentException
from agent_service.utils.date_utils import get_now_utc
from agent_service.utils.environment import EnvironmentUtils
from agent_service.utils.feature_flags import (
    agent_output_cache_enabled,
    is_user_agent_admin,
    user_has_qc_tool_access,
)
from agent_service.utils.logs import init_stdout_logging
from agent_service.utils.postgres import get_psql
from agent_service.utils.prefect_task_executor import PrefectTaskExecutor
from agent_service.utils.sentry_utils import init_sentry
from no_auth_endpoints import initialize_unauthed_endpoints

DEFAULT_IP = "0.0.0.0"
DEFAULT_DAL_PORT = 8000
SERVICE_NAME = "AgentService"

logger = logging.getLogger(__name__)


# Helper wrapper for mypy/intellisense in usage of application.state
class AgentServiceState(State):
    agent_service_impl: AgentServiceImpl


class FastAPIExtended(FastAPI):
    state: AgentServiceState


@asynccontextmanager
async def lifespan(application: FastAPIExtended) -> AsyncGenerator:
    init_stdout_logging()
    init_sentry(disable_sentry=not EnvironmentUtils.is_deployed)

    logger.info("Warming up DB connection and JWT key map...")
    get_psql()
    get_keyid_to_key_map()

    logger.info("Starting server...")
    env = get_environment_tag()
    base_url = "alfa.boosted.ai" if env == "ALPHA" else "agent-dev.boosted.ai"
    channel = "alfa-client-queries" if env == "ALPHA" else "alfa-client-queries-dev"

    cache = None
    if agent_output_cache_enabled() and os.getenv("REDIS_HOST"):
        logger.info(f"Using redis output cache. Connecting to {os.getenv('REDIS_HOST')}")
        cache = get_redis_cache_backend_for_output()

    logger.info("Warming up AsyncDB connection")
    async_db = AsyncDB(pg=AsyncPostgresBase(min_pool_size=1, max_pool_size=3))
    await async_db.pg.generic_read("SELECT * FROM agent.agents LIMIT 1")

    application.state.agent_service_impl = AgentServiceImpl(
        task_executor=PrefectTaskExecutor(),
        gpt_service_stub=_get_gpt_service_stub()[0],
        async_db=async_db,
        clickhouse_db=Clickhouse(),
        slack_sender=SlackSender(channel=channel),
        base_url=base_url,
        cache=cache,
    )

    yield


application = FastAPIExtended(title="Agent Service", lifespan=lifespan)


router = APIRouter(prefix="/api")
application.add_middleware(
    CORSMiddleware,  # Add CORS middleware
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class CustomGZipMiddleware(GZipMiddleware):
    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await super().__call__(scope, receive, send)
            return

        if any(
            prefix in scope["path"] for prefix in ("/api/notification/stream", "/api/agent/stream")
        ):
            await self.app(scope, receive, send)
        else:
            await super().__call__(scope, receive, send)


# compress responses that >1MB (level 1 to 9, the higher the level, the smaller size is)
application.add_middleware(CustomGZipMiddleware, minimum_size=100 * 1000, compresslevel=5)

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
    real_user_id: Optional[str] = None
    request_number: int = -1
    frontend_version: Optional[str] = None
    fullstory_link: Optional[str] = None

    def to_json_dict(self) -> Dict[str, Any]:
        data = dataclasses.asdict(self)

        return {key: value for key, value in data.items() if value is not None}


def update_audit_info_with_response_info(
    audit_info: AuditInfo, received_timestamp: datetime.datetime
) -> None:
    response_timestamp_tz = get_now_utc()
    response_timestamp_no_tz = response_timestamp_tz.replace(tzinfo=None)
    if received_timestamp.tzinfo:
        response_timestamp = response_timestamp_tz
    else:
        response_timestamp = response_timestamp_no_tz
    audit_info.response_timestamp = response_timestamp
    audit_info.internal_processing_time = (response_timestamp - received_timestamp).total_seconds()
    if audit_info.client_timestamp:
        client_ts = datetime.datetime.fromisoformat(audit_info.client_timestamp)
        if client_ts.tzinfo:
            response_timestamp = response_timestamp_tz
        else:
            response_timestamp = response_timestamp_no_tz
        audit_info.total_processing_time = (response_timestamp - client_ts).total_seconds()


class ProcessTimeMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        global REQUEST_COUNTER
        REQUEST_COUNTER += 1

        received_timestamp = get_now_utc()

        message_queue: asyncio.Queue[Message] = asyncio.Queue()

        async def receive_wrapper() -> Message:
            message = await receive()
            await message_queue.put(message)
            return message

        async def receive_from_queue() -> Any:
            return await message_queue.get()

        request = Request(scope, receive_wrapper)

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
            frontend_version=request.headers.get("clientversion", None),
            fullstory_link=request.headers.get("fullstorylink", None),
        )

        try:
            authorization = request.headers.get("Authorization", None)
            if authorization:
                user_info = parse_header(request=request, auth_token=authorization)
                request.state.user_info = user_info
                audit_info.user_id = user_info.user_id
                audit_info.real_user_id = user_info.real_user_id

            request_body = await request.body()
            audit_info.request_body = json.loads(request_body) if request_body else None
        except Exception:
            audit_info.error = traceback.format_exc()

        try:
            # Consuming `request.body()` in the middleware reads the request body from `receive`,
            # which means the downstream app won't be able to access it because it's already consumed.
            # To prevent blocking downstream apps from accessing the request body, we use `message_queue`
            # to store the messages read from `receive`. Then, we pass `receive_from_queue` to the
            # downstream app, so it can read the same messages as if they were coming directly from `receive`.
            # This ensures the request body is available to downstream apps without loss of data.
            await self.app(scope, receive_from_queue, send)
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


application.add_middleware(ProcessTimeMiddleware)


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
    return await application.state.agent_service_impl.create_agent(user=user, is_draft=req.is_draft)


@router.post(
    "/agent/update-draft-status/{agent_id}",
    response_model=UpdateAgentDraftStatusResponse,
    status_code=status.HTTP_200_OK,
)
async def update_agent_draft_status(
    agent_id: str, req: UpdateAgentDraftStatusRequest, user: User = Depends(parse_header)
) -> UpdateAgentDraftStatusResponse:
    await validate_user_agent_access(
        user.user_id, agent_id, async_db=application.state.agent_service_impl.pg
    )
    return await application.state.agent_service_impl.update_agent_draft_status(
        agent_id=agent_id, is_draft=req.is_draft
    )


@router.post(
    "/agent/terminate/{agent_id}",
    response_model=TerminateAgentResponse,
    status_code=status.HTTP_200_OK,
)
async def terminate_agent(
    agent_id: str,
    req: TerminateAgentRequest,
    user: User = Depends(parse_header),
) -> TerminateAgentResponse:
    """
    Terminate a running agent, 2 ways:
    1. Terminate a running plan by `plan_run_id` -> so `plan_id` is not required and the plan is
        still useable
    2. Terminate a running plan by `plan_id` -> this plan won't be useable anymore. You must create
        a new plan
    """
    if not req.plan_id and not req.plan_run_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Either plan_id or plan_run_id must be provided",
        )

    await validate_user_agent_access(
        user.user_id, agent_id, async_db=application.state.agent_service_impl.pg
    )
    return await application.state.agent_service_impl.terminate_agent(
        agent_id=agent_id, plan_id=req.plan_id, plan_run_id=req.plan_run_id
    )


@router.delete(
    "/agent/delete-agent/{agent_id}",
    response_model=DeleteAgentResponse,
    status_code=status.HTTP_200_OK,
)
async def delete_agent(agent_id: str, user: User = Depends(parse_header)) -> DeleteAgentResponse:
    await validate_user_agent_access(
        user.user_id,
        agent_id,
        async_db=application.state.agent_service_impl.pg,
        invalidate_cache=True,
    )
    return await application.state.agent_service_impl.delete_agent(agent_id=agent_id)


@router.post(
    "/agent/restore-agent/{agent_id}",
    response_model=RestoreAgentResponse,
    status_code=status.HTTP_200_OK,
)
async def restore_agent(agent_id: str, user: User = Depends(parse_header)) -> RestoreAgentResponse:
    if not (user.is_super_admin or is_user_agent_admin(user.user_id)):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only admins can restore agents",
        )

    return await application.state.agent_service_impl.restore_agent(agent_id=agent_id)


@router.put(
    "/agent/update-agent/{agent_id}",
    response_model=UpdateAgentResponse,
    status_code=status.HTTP_200_OK,
)
async def update_agent(
    agent_id: str, req: UpdateAgentRequest, user: User = Depends(parse_header)
) -> UpdateAgentResponse:
    # NOTE: currently only allow updating agent name
    await validate_user_agent_access(
        user.user_id, agent_id, async_db=application.state.agent_service_impl.pg
    )

    return await application.state.agent_service_impl.update_agent(agent_id=agent_id, req=req)


@router.get(
    "/agent/get-all-agents", response_model=GetAllAgentsResponse, status_code=status.HTTP_200_OK
)
async def get_all_agents(user: User = Depends(parse_header)) -> GetAllAgentsResponse:
    return await application.state.agent_service_impl.get_all_agents(user=user)


@router.get("/agent/get-agent/{agent_id}", response_model=AgentInfo, status_code=status.HTTP_200_OK)
async def get_agent(agent_id: str, user: User = Depends(parse_header)) -> AgentInfo:
    logger.info(f"Validating if {user.user_id=} has access to {agent_id=}.")
    if not (user.is_super_admin or is_user_agent_admin(user.user_id)):
        await validate_user_agent_access(
            user.user_id, agent_id, async_db=application.state.agent_service_impl.pg
        )

    return await application.state.agent_service_impl.get_agent(agent_id=agent_id)


@router.get(
    "/agent/notification-criteria/{agent_id}",
    response_model=List[CustomNotification],
    status_code=status.HTTP_200_OK,
)
async def get_all_agent_notification_criteria(
    agent_id: str, user: User = Depends(parse_header)
) -> List[CustomNotification]:
    logger.info(f"Validating if {user.user_id=} has access to {agent_id=}.")
    if not (user.is_super_admin or is_user_agent_admin(user.user_id)):
        await validate_user_agent_access(
            user.user_id, agent_id, async_db=application.state.agent_service_impl.pg
        )
    return await application.state.agent_service_impl.get_all_agent_notification_criteria(
        agent_id=agent_id
    )


@router.post(
    "/agent/notification-criteria/create",
    response_model=CustomNotificationStatusResponse,
    status_code=status.HTTP_200_OK,
)
async def create_agent_notification_criteria(
    req: CreateCustomNotificationRequest, user: User = Depends(parse_header)
) -> CustomNotificationStatusResponse:
    logger.info(f"Validating if {user.user_id=} has access to {req.agent_id=}.")
    await validate_user_agent_access(
        user.user_id, req.agent_id, async_db=application.state.agent_service_impl.pg
    )
    cn_id = await application.state.agent_service_impl.create_agent_notification_criteria(req=req)
    return CustomNotificationStatusResponse(custom_notification_id=cn_id, success=True)


@router.delete(
    "/agent/notification-criteria/delete/{agent_id}/{notification_criteria_id}",
    response_model=CustomNotificationStatusResponse,
    status_code=status.HTTP_200_OK,
)
async def delete_agent_notification_criteria(
    agent_id: str, notification_criteria_id: str, user: User = Depends(parse_header)
) -> CustomNotificationStatusResponse:
    logger.info(f"Validating if {user.user_id=} has access to {agent_id=}.")
    await validate_user_agent_access(
        user.user_id, agent_id, async_db=application.state.agent_service_impl.pg
    )
    await application.state.agent_service_impl.delete_agent_notification_criteria(
        agent_id=agent_id, custom_notification_id=notification_criteria_id
    )
    return CustomNotificationStatusResponse(
        custom_notification_id=notification_criteria_id, success=True
    )


@router.get(
    "/agent/notification-emails/{agent_id}",
    response_model=NotificationEmailsResponse,
    status_code=status.HTTP_200_OK,
)
async def get_agent_notification_emails(
    agent_id: str, user: User = Depends(parse_header)
) -> NotificationEmailsResponse:
    logger.info(f"Validating if {user.user_id=} has access to {agent_id=}.")
    if not (user.is_super_admin or is_user_agent_admin(user.user_id)):
        await validate_user_agent_access(
            user.user_id, agent_id, async_db=application.state.agent_service_impl.pg
        )
    return await application.state.agent_service_impl.get_agent_notification_emails(
        agent_id=agent_id
    )


@router.post(
    "/agent/notification-emails/update",
    response_model=UpdateNotificationEmailsResponse,
    status_code=status.HTTP_200_OK,
)
async def update_agent_notification_emails(
    req: UpdateNotificationEmailsRequest, user: User = Depends(parse_header)
) -> UpdateNotificationEmailsResponse:
    agent_id = req.agent_id
    emails = req.emails
    try:
        logger.info(f"Validating if {user.user_id=} has access to {agent_id=}.")
        if not (user.is_super_admin or is_user_agent_admin(user.user_id)):
            await validate_user_agent_access(
                user.user_id, agent_id, async_db=application.state.agent_service_impl.pg
            )
        await application.state.agent_service_impl.set_agent_notification_emails(
            agent_id=agent_id, emails=emails, user_id=user.user_id
        )
        return UpdateNotificationEmailsResponse(success=True, bad_emails=[])
    except Exception as e:
        logger.warning(f"error in updating agent:{req.agent_id} emails:{req.emails}, error: {e}")
        return UpdateNotificationEmailsResponse(success=False, bad_emails=[])


@router.post(
    "/agent/set-agent-feedback",
    response_model=SetAgentFeedBackResponse,
    status_code=status.HTTP_200_OK,
)
async def set_agent_feedback(
    req: SetAgentFeedBackRequest, user: User = Depends(parse_header)
) -> SetAgentFeedBackResponse:
    return await application.state.agent_service_impl.set_agent_feedback(
        feedback_data=req, user_id=user.user_id
    )


@router.get(
    "/agent/get-agent-feedback/{agent_id}/{plan_id}/{plan_run_id}/{output_id}",
    response_model=GetAgentFeedBackResponse,
    status_code=status.HTTP_200_OK,
)
async def get_agent_feedback(
    agent_id: str,
    plan_id: str,
    plan_run_id: str,
    output_id: str,
    user: User = Depends(parse_header),
) -> GetAgentFeedBackResponse:
    return await application.state.agent_service_impl.get_agent_feedback(
        agent_id=agent_id,
        plan_id=plan_id,
        plan_run_id=plan_run_id,
        output_id=output_id,
        user_id=user.user_id,
    )


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
    await validate_user_agent_access(
        user.user_id, req.agent_id, async_db=application.state.agent_service_impl.pg
    )
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
) -> UploadFileResponse:
    await validate_user_agent_access(
        user.user_id, agent_id, async_db=application.state.agent_service_impl.pg
    )
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
    start_index: Optional[int] = 0,
    limit_num: Optional[int] = None,
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
        await validate_user_agent_access(
            user.user_id, agent_id, async_db=application.state.agent_service_impl.pg
        )

    return await application.state.agent_service_impl.get_chat_history(
        agent_id=agent_id, start=start, end=end, start_index=start_index, limit_num=limit_num
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
    start_index: Optional[int] = 0,
    limit_num: Optional[int] = None,
    user: User = Depends(parse_header),
) -> GetAgentWorklogBoardResponse:
    """Get agent worklogs to build the Work Log Board
    Except `agent_id`, all other arguments are optional and can be used to filter the work log, but
    strongly recommend to have at least 1 filter to avoid returning too many entries.
    NOTE: If any of `start_date` or `end_date` is provided, and `limit_num` is also provided,
    it will be most recent N entries within the date range.

    Args:
        agent_id (str): agent ID
        start (Optional[datetime.date]): start DATE to filter work log, inclusive
        end (Optional[datetime.date]): end DATE to filter work log, inclusive
        start_index (Optional[int]): start index to filter work log
        limit_num (Optional[int]): number of plan runs to return
    """
    if not (user.is_super_admin or is_user_agent_admin(user.user_id)):
        logger.info(f"Validating if user {user.user_id} has access to agent {agent_id}.")
        await validate_user_agent_access(
            user.user_id, agent_id, async_db=application.state.agent_service_impl.pg
        )

    return await application.state.agent_service_impl.get_agent_worklog_board(
        agent_id=agent_id,
        start_date=start_date,
        end_date=end_date,
        start_index=start_index,
        limit_num=limit_num,
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
        await validate_user_agent_access(
            user.user_id, agent_id, async_db=application.state.agent_service_impl.pg
        )

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
        await validate_user_agent_access(
            user.user_id, agent_id, async_db=application.state.agent_service_impl.pg
        )

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
        await validate_user_agent_access(
            user.user_id, agent_id, async_db=application.state.agent_service_impl.pg
        )

    return await application.state.agent_service_impl.get_agent_plan_output(agent_id=agent_id)


@router.post(
    "/agent/delete-agent-output/{agent_id}",
    response_model=DeleteAgentOutputResponse,
    status_code=status.HTTP_200_OK,
)
async def delete_agent_output(
    agent_id: str, req: DeleteAgentOutputRequest, user: User = Depends(parse_header)
) -> DeleteAgentOutputResponse:
    """
    Delete an agent output, creating a new modified plan without the output step.
    """
    await validate_user_agent_access(
        user.user_id, agent_id, async_db=application.state.agent_service_impl.pg
    )

    return await application.state.agent_service_impl.delete_agent_output(
        agent_id=agent_id, req=req
    )


@router.post(
    "/agent/lock-agent-output/{agent_id}",
    response_model=LockAgentOutputResponse,
    status_code=status.HTTP_200_OK,
)
async def lock_agent_output(
    agent_id: str, req: LockAgentOutputRequest, user: User = Depends(parse_header)
) -> LockAgentOutputResponse:
    """
    Lock an agent output, which will force it to be included always.
    """
    if not (user.is_super_admin or is_user_agent_admin(user.user_id)):
        await validate_user_agent_access(
            user.user_id, agent_id, async_db=application.state.agent_service_impl.pg
        )

    return await application.state.agent_service_impl.lock_agent_output(agent_id=agent_id, req=req)


@router.post(
    "/agent/unlock-agent-output/{agent_id}",
    response_model=UnlockAgentOutputResponse,
    status_code=status.HTTP_200_OK,
)
async def unlock_agent_output(
    agent_id: str, req: UnlockAgentOutputRequest, user: User = Depends(parse_header)
) -> UnlockAgentOutputResponse:
    """
    Unlock an agent output.
    """
    if not (user.is_super_admin or is_user_agent_admin(user.user_id)):
        await validate_user_agent_access(
            user.user_id, agent_id, async_db=application.state.agent_service_impl.pg
        )

    return await application.state.agent_service_impl.unlock_agent_output(
        agent_id=agent_id, req=req
    )


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
        await validate_user_agent_access(
            user.user_id, agent_id, async_db=application.state.agent_service_impl.pg
        )

    return await application.state.agent_service_impl.get_agent_plan_output(
        agent_id=agent_id, plan_run_id=plan_run_id
    )


@router.get(
    "/agent/get-citation-details/{citation_type}/{citation_id}",
    response_model=GetCitationDetailsResponse,
    status_code=status.HTTP_200_OK,
)
async def get_citation_details(
    citation_type: CitationType, citation_id: str, user: User = Depends(parse_header)
) -> GetCitationDetailsResponse:
    details = await application.state.agent_service_impl.get_citation_details(
        citation_type=citation_type, citation_id=citation_id, user_id=user.user_id
    )
    return GetCitationDetailsResponse(details=details)


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
        await validate_user_agent_access(
            user.user_id, agent_id, async_db=application.state.agent_service_impl.pg
        )

    async def _wrap_serializer() -> AsyncContentStream:
        try:
            async for event in application.state.agent_service_impl.stream_agent_events(
                request=request, agent_id=agent_id
            ):
                try:
                    to_send = event.model_dump_json()
                except Exception:
                    logger.exception(f"Error while sending agent event {agent_id=}")
                    continue

                yield ServerSentEvent(data=to_send, event="agent-event")
                log_event(
                    event_name="agent-event-sent",
                    event_data={"agent_id": agent_id, "user_id": user.user_id, "data": to_send},
                )
        except Exception as e:
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

    await validate_user_plan_run_access(
        user.user_id, req.plan_run_id, async_db=application.state.agent_service_impl.pg
    )
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

    await validate_user_plan_run_access(
        user.user_id, req.plan_run_id, async_db=application.state.agent_service_impl.pg
    )
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
    await validate_user_agent_access(
        user.user_id, req.agent_id, async_db=application.state.agent_service_impl.pg
    )
    return await application.state.agent_service_impl.mark_notifications_as_read(
        req.agent_id, req.timestamp
    )


@router.post(
    "/agent/mark-notifications-as-unread",
    response_model=MarkNotificationsAsUnreadResponse,
    status_code=status.HTTP_200_OK,
)
async def mark_notifications_as_unread(
    req: MarkNotificationsAsUnreadRequest, user: User = Depends(parse_header)
) -> MarkNotificationsAsUnreadResponse:
    """Mark agent notifications as unread after a given timestamp

    Args:
        agent_id (str): agent ID
        message_id: message ID - set all messages with created_at >= message timestamp as unread
    """
    await validate_user_agent_access(
        user.user_id, req.agent_id, async_db=application.state.agent_service_impl.pg
    )
    return await application.state.agent_service_impl.mark_notifications_as_unread(
        req.agent_id, req.message_id
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
    await validate_user_agent_access(
        user.user_id, req.agent_id, async_db=application.state.agent_service_impl.pg
    )
    return await application.state.agent_service_impl.enable_agent_automation(
        agent_id=req.agent_id, user_id=user.user_id
    )


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
    await validate_user_agent_access(
        user.user_id, req.agent_id, async_db=application.state.agent_service_impl.pg
    )
    return await application.state.agent_service_impl.disable_agent_automation(
        agent_id=req.agent_id, user_id=user.user_id
    )


@router.post(
    "/agent/set-schedule",
    response_model=SetAgentScheduleResponse,
    status_code=status.HTTP_200_OK,
)
async def set_agent_schedule(
    req: SetAgentScheduleRequest, user: User = Depends(parse_header)
) -> SetAgentScheduleResponse:
    await validate_user_agent_access(
        user.user_id, req.agent_id, async_db=application.state.agent_service_impl.pg
    )
    return await application.state.agent_service_impl.set_agent_schedule(req=req)


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
    "/debug/get-plan-run-debug-info/{agent_id}/{plan_run_id}",
    response_model=GetPlanRunDebugInfoResponse,
    status_code=status.HTTP_200_OK,
)
async def get_plan_run_debug_info(
    agent_id: str, plan_run_id: str, user: User = Depends(parse_header)
) -> GetPlanRunDebugInfoResponse:
    """Return detailed information about a plan run for debugging purposes,
    include the list of tools, inputs, outputs
    """
    if not (user.is_super_admin or is_user_agent_admin(user.user_id)):
        await validate_user_agent_access(
            user.user_id, agent_id, async_db=application.state.agent_service_impl.pg
        )

    return await application.state.agent_service_impl.get_plan_run_debug_info(
        agent_id=agent_id, plan_run_id=plan_run_id
    )


@router.post(
    "/debug/modify-plan-run-args/{agent_id}/{plan_run_id}",
    response_model=ModifyPlanRunArgsResponse,
    status_code=status.HTTP_200_OK,
)
async def modify_plan_run_args(
    agent_id: str,
    plan_run_id: str,
    req: ModifyPlanRunArgsRequest,
    user: User = Depends(parse_header),
) -> ModifyPlanRunArgsResponse:
    """Duplicate the plan with modified input variables and rerun it"""
    if not (user.is_super_admin or is_user_agent_admin(user.user_id)):
        await validate_user_agent_access(
            user.user_id, agent_id, async_db=application.state.agent_service_impl.pg
        )

    return await application.state.agent_service_impl.modify_plan_run_args(
        agent_id=agent_id, plan_run_id=plan_run_id, user_id=user.user_id, req=req
    )


@router.get(
    "/debug/args/{replay_id}",
    response_model=GetDebugToolArgsResponse,
    status_code=status.HTTP_200_OK,
)
async def get_agent_debug_args(
    replay_id: str, user: User = Depends(parse_header)
) -> GetDebugToolArgsResponse:
    if not user.is_super_admin and not is_user_agent_admin(user.user_id):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="User is not authorized"
        )
    return await application.state.agent_service_impl.get_debug_tool_args(replay_id=replay_id)


@router.get(
    "/debug/result/{replay_id}",
    response_model=GetDebugToolResultResponse,
    status_code=status.HTTP_200_OK,
)
async def get_agent_debug_result(
    replay_id: str, user: User = Depends(parse_header)
) -> GetDebugToolResultResponse:
    if not user.is_super_admin and not is_user_agent_admin(user.user_id):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="User is not authorized"
        )
    return await application.state.agent_service_impl.get_debug_tool_result(replay_id=replay_id)


@router.get(
    "/debug/tool/get-tool-library",
    response_model=GetToolLibraryResponse,
    status_code=status.HTTP_200_OK,
)
async def get_tool_library(user: User = Depends(parse_header)) -> GetToolLibraryResponse:
    return await application.state.agent_service_impl.get_tool_library(user=user)


@router.get(
    "/memory/list-memory-items",
    response_model=ListMemoryItemsResponse,
    status_code=status.HTTP_200_OK,
)
async def list_memory_items(user: User = Depends(parse_header)) -> ListMemoryItemsResponse:
    """
    List memory items
    """
    return await application.state.agent_service_impl.list_memory_items(user_id=user.user_id)


@router.post(
    "/memory/get-autocomplete-items",
    response_model=GetAutocompleteItemsResponse,
    status_code=status.HTTP_200_OK,
)
async def get_autocomplete_items(
    req: GetAutocompleteItemsRequest,
    user: User = Depends(parse_header),
) -> GetAutocompleteItemsResponse:
    """
    Gets autocomplete items
    """
    return await application.state.agent_service_impl.get_autocomplete_items(
        user_id=user.user_id, text=req.text
    )


@router.get(
    "/memory/get-memory-content/{type}/{id}",
    response_model=GetMemoryContentResponse,
    status_code=status.HTTP_200_OK,
)
async def get_memory_content(
    type: str,
    id: str,
    user: User = Depends(parse_header),
) -> GetMemoryContentResponse:
    """
    Gets preview of memory content (output in text or table form)
    Args:
        type (str): memory type (portfolio / watchlist)
        id (str): the ID of the memory type
    """
    return await application.state.agent_service_impl.get_memory_content(
        user_id=user.user_id, type=type, id=id
    )


@router.delete(
    "/memory/delete-memory/{type}/{id}",
    response_model=DeleteMemoryResponse,
    status_code=status.HTTP_200_OK,
)
async def delete_memory(
    type: str,
    id: str,
    user: User = Depends(parse_header),
) -> DeleteMemoryResponse:
    """
    Delete memory item
    """
    return await application.state.agent_service_impl.delete_memory(
        user_id=user.user_id, type=type, id=id
    )


@router.post(
    "/memory/rename-memory",
    response_model=RenameMemoryResponse,
    status_code=status.HTTP_200_OK,
)
async def rename_memory(
    req: RenameMemoryRequest,
    user: User = Depends(parse_header),
) -> RenameMemoryResponse:
    """
    Rename memory item
    """
    return await application.state.agent_service_impl.rename_memory(
        user_id=user.user_id, type=req.type, id=req.id, new_name=req.new_name
    )


@router.get(
    "/regression-test/{service_version}",
    response_model=GetTestSuiteRunInfoResponse,
    status_code=status.HTTP_200_OK,
)
async def get_info_for_test_run(
    service_version: str, user: User = Depends(parse_header)
) -> GetTestSuiteRunInfoResponse:
    if get_environment_tag() in [STAGING_TAG, PROD_TAG]:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="")
    if not user.is_super_admin and not is_user_agent_admin(user.user_id):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="User is not authorized"
        )
    return await application.state.agent_service_impl.get_info_for_test_suite_run(
        service_version=service_version
    )


@router.get(
    "/regression-test-suite-runs",
    response_model=GetTestSuiteRunsResponse,
    status_code=status.HTTP_200_OK,
)
async def get_test_suite_runs(user: User = Depends(parse_header)) -> GetTestSuiteRunsResponse:
    if get_environment_tag() in [STAGING_TAG, PROD_TAG]:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="")
    if not user.is_super_admin and not is_user_agent_admin(user.user_id):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="User is not authorized"
        )
    return await application.state.agent_service_impl.get_test_suite_runs()


@router.get(
    "/regression-test-cases",
    response_model=GetTestCasesResponse,
    status_code=status.HTTP_200_OK,
)
async def get_test_cases(user: User = Depends(parse_header)) -> GetTestCasesResponse:
    if get_environment_tag() in [STAGING_TAG, PROD_TAG]:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="")
    if not user.is_super_admin and not is_user_agent_admin(user.user_id):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="User is not authorized"
        )
    return await application.state.agent_service_impl.get_test_cases()


@router.get(
    "/regression-test-case/{test_name}",
    response_model=GetTestCaseInfoResponse,
    status_code=status.HTTP_200_OK,
)
async def get_info_for_test_case(
    test_name: str, user: User = Depends(parse_header)
) -> GetTestCaseInfoResponse:
    if get_environment_tag() in [STAGING_TAG, PROD_TAG]:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="")
    if not user.is_super_admin and not is_user_agent_admin(user.user_id):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="User is not authorized"
        )
    return await application.state.agent_service_impl.get_info_for_test_case(test_name=test_name)


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
    return create_jwt(user_id=user_id, expiry_hours=1)


# Account Endpoints


@router.post(
    "/user/update-user",
    response_model=UpdateUserResponse,
    status_code=status.HTTP_200_OK,
)
async def update_user(
    req: UpdateUserRequest,
    user: User = Depends(parse_header),
) -> UpdateUserResponse:
    return await application.state.agent_service_impl.update_user(
        user_id=user.user_id, name=req.name, username=req.username, email=req.email
    )


@router.get(
    "/user/get-account-info", response_model=GetAccountInfoResponse, status_code=status.HTTP_200_OK
)
async def get_account_info(user: User = Depends(parse_header)) -> GetAccountInfoResponse:
    account = await application.state.agent_service_impl.get_account_info(user=user)
    return GetAccountInfoResponse(account=account)


@router.post("/user/get-users", response_model=GetUsersResponse, status_code=status.HTTP_200_OK)
async def get_users_info(
    req: GetUsersRequest, user: User = Depends(parse_header)
) -> GetUsersResponse:
    accounts = await application.state.agent_service_impl.get_users_info(
        user=user, user_ids=req.user_ids
    )
    return GetUsersResponse(accounts=accounts)


@router.get(
    "/user/get-team-accounts",
    response_model=GetTeamAccountsResponse,
    status_code=status.HTTP_200_OK,
)
async def get_team_accounts(user: User = Depends(parse_header)) -> GetTeamAccountsResponse:
    accounts = await application.state.agent_service_impl.get_valid_notification_users(
        user_id=user.user_id
    )
    return GetTeamAccountsResponse(accounts=accounts)


@router.get(
    "/user/has-access", response_model=UserHasAccessResponse, status_code=status.HTTP_200_OK
)
async def get_user_has_access(user: User = Depends(parse_header)) -> UserHasAccessResponse:
    if user.is_admin or user.is_super_admin:
        return UserHasAccessResponse(success=True)

    has_access = await application.state.agent_service_impl.get_user_has_alfa_access(user=user)
    return UserHasAccessResponse(success=has_access)


@router.post(
    "/convert/markdown",
    response_class=Response,
    status_code=status.HTTP_200_OK,
)
async def convert_markdown(
    req: ConvertMarkdownRequest, user: User = Depends(parse_header)
) -> Response:
    raw_bytes, media_type = await application.state.agent_service_impl.convert_markdown(
        content=req.content, new_type=req.format
    )
    return Response(content=raw_bytes, media_type=media_type)


@router.get(
    "/canned-prompts",
    response_model=GetCannedPromptsResponse,
    status_code=status.HTTP_200_OK,
)
def get_canned_prompts(user: User = Depends(parse_header)) -> GetCannedPromptsResponse:
    return application.state.agent_service_impl.get_canned_prompts()


# Sections Endpoints
@router.post(
    "/create-section",
    response_model=CreateSectionResponse,
    status_code=status.HTTP_200_OK,
)
async def create_section(
    req: CreateSectionRequest, user: User = Depends(parse_header)
) -> CreateSectionResponse:

    return CreateSectionResponse(
        section_id=await application.state.agent_service_impl.create_sidebar_section(
            name=req.name, user=user
        )
    )


@router.post(
    "/delete-section",
    response_model=DeleteSectionResponse,
    status_code=status.HTTP_200_OK,
)
async def delete_section(
    req: DeleteSectionRequest, user: User = Depends(parse_header)
) -> DeleteSectionResponse:
    await application.state.agent_service_impl.delete_sidebar_section(
        section_id=req.section_id, user=user
    )
    return DeleteSectionResponse(success=True)


@router.post(
    "/rename-section",
    response_model=RenameSectionResponse,
    status_code=status.HTTP_200_OK,
)
async def rename_section(
    req: RenameSectionRequest, user: User = Depends(parse_header)
) -> RenameSectionResponse:
    await application.state.agent_service_impl.rename_sidebar_section(
        section_id=req.section_id, new_name=req.new_name, user=user
    )
    return RenameSectionResponse(success=True)


@router.post(
    "/set-agent-section",
    response_model=SetAgentSectionResponse,
    status_code=status.HTTP_200_OK,
)
async def set_agent_section(
    req: SetAgentSectionRequest, user: User = Depends(parse_header)
) -> SetAgentSectionResponse:
    await application.state.agent_service_impl.set_agent_sidebar_section(
        new_section_id=req.new_section_id, agent_id=req.agent_id, user=user
    )
    return SetAgentSectionResponse(success=True)


@router.post(
    "/rearrange-section",
    response_model=RearrangeSectionResponse,
    status_code=status.HTTP_200_OK,
)
async def rearrange_section(
    req: RearrangeSectionRequest, user: User = Depends(parse_header)
) -> RearrangeSectionResponse:
    await application.state.agent_service_impl.rearrange_sidebar_section(
        section_id=req.section_id, new_index=req.new_index, user=user
    )
    return RearrangeSectionResponse(success=True)


@router.get(
    "/template/get-all-companies",
    response_model=GetCompaniesResponse,
    status_code=status.HTTP_200_OK,
)
async def get_all_companies(user: User = Depends(parse_header)) -> GetCompaniesResponse:
    if not is_user_agent_admin(user.user_id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="User does not have access"
        )
    return await application.state.agent_service_impl.get_all_companies(user)


@router.get(
    "/template/get-prompt-templates",
    response_model=GetPromptTemplatesResponse,
    status_code=status.HTTP_200_OK,
)
async def get_prompt_templates(user: User = Depends(parse_header)) -> GetPromptTemplatesResponse:
    is_user_admin = is_user_agent_admin(user.user_id)
    templates = await application.state.agent_service_impl.get_prompt_templates(user, is_user_admin)
    return GetPromptTemplatesResponse(prompt_templates=templates)


@router.post(
    "/template/create-prompt-template",
    response_model=CreatePromptTemplateRequest,
    status_code=status.HTTP_200_OK,
)
async def create_prompt_template(req: CreatePromptTemplateRequest) -> CreatePromptTemplateResponse:
    return await application.state.agent_service_impl.create_prompt_template(
        name=req.name,
        description=req.description,
        prompt=req.prompt,
        category=req.category,
        plan_run_id=req.plan_run_id,
        organization_ids=req.organization_ids,
    )


@router.post(
    "/template/set-prompt-template-visibility",
    response_model=SetPromptTemplateVisibilityRequest,
    status_code=status.HTTP_200_OK,
)
async def set_prompt_template_visibility(
    req: SetPromptTemplateVisibilityRequest,
) -> SetPromptTemplateVisibilityResponse:
    return await application.state.agent_service_impl.set_prompt_template_visibility(
        template_id=req.template_id,
        is_visible=req.is_visible,
    )


@router.post(
    "/template/generate-template-plan",
    response_model=GenTemplatePlanResponse,
    status_code=status.HTTP_200_OK,
)
async def gen_template_plan(
    req: GenTemplatePlanRequest, user: User = Depends(parse_header)
) -> GenTemplatePlanResponse:
    return await application.state.agent_service_impl.gen_template_plan(
        template_prompt=req.template_prompt,
        user=user,
    )


@router.post(
    "/template/run-template-plan",
    response_model=RunTemplatePlanResponse,
    status_code=status.HTTP_200_OK,
)
async def create_agent_and_run_template_plan(
    req: RunTemplatePlanRequest, user: User = Depends(parse_header)
) -> RunTemplatePlanResponse:

    return await application.state.agent_service_impl.create_agent_and_run_template_plan(
        template_prompt=req.template_prompt,
        plan=req.plan,
        is_draft=req.is_draft,
        user=user,
    )


@router.post(
    "/template/delete-template-prompt",
    response_model=DeletePromptTemplateResponse,
    status_code=status.HTTP_200_OK,
)
async def delete_prompt_template(
    req: DeletePromptTemplateRequest, user: User = Depends(parse_header)
) -> DeletePromptTemplateResponse:

    return await application.state.agent_service_impl.delete_prompt_template(
        template_id=req.template_id
    )


@router.post(
    "/template/update-template-prompt",
    response_model=UpdatePromptTemplateResponse,
    status_code=status.HTTP_200_OK,
)
async def update_prompt_template(
    req: UpdatePromptTemplateRequest, user: User = Depends(parse_header)
) -> UpdatePromptTemplateResponse:

    return await application.state.agent_service_impl.update_prompt_template(
        template_id=req.template_id,
        name=req.name,
        description=req.description,
        category=req.category,
        prompt=req.prompt,
        plan=req.plan,
        is_visible=req.is_visible,
        organization_ids=req.organization_ids,
    )


@router.post(
    "/copy-agent",
    response_model=CopyAgentToUsersResponse,
    status_code=status.HTTP_200_OK,
)
async def copy_agent(
    req: CopyAgentToUsersRequest, user: User = Depends(parse_header)
) -> CopyAgentToUsersResponse:
    response_dict = await application.state.agent_service_impl.copy_agent(
        src_agent_id=req.src_agent_id,
        dst_user_ids=req.dst_user_ids,
        dst_agent_name=req.dst_agent_name,
    )
    # Check to see if user is authorized to duplicate
    # they are only allowed to request with one user_id and it must be their own
    # if they are spoofing (user_id != real_user_id), allow request
    if not is_user_agent_admin(user.user_id):
        if user.user_id == user.real_user_id and (
            len(req.dst_user_ids) > 1 or user.user_id not in req.dst_user_ids
        ):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="User is not authorized"
            )

    return CopyAgentToUsersResponse(user_id_to_new_agent_id_map=response_dict)


# variables/data endpoints
@router.get(
    "/data/variables/available-variables",
    response_model=GetAvailableVariablesResponse,
    status_code=status.HTTP_200_OK,
)
async def get_available_variable(
    user: User = Depends(parse_header),
) -> GetAvailableVariablesResponse:
    """
    Retrieves relevant metadata about all variables available to the user.
    """
    return await application.state.agent_service_impl.get_all_available_variables(user=user)


@router.get(
    "/data/variables/hierarchy",
    response_model=GetVariableHierarchyResponse,
    status_code=status.HTTP_200_OK,
)
async def get_all_variable_hierarchy(
    user: User = Depends(parse_header),
) -> GetVariableHierarchyResponse:
    """
    Retrieves all variable display hierarchies in a flat format.
    """
    return await application.state.agent_service_impl.get_variable_hierarchy(user=user)


@router.post(
    "/data/variables/coverage",
    response_model=GetVariableCoverageResponse,
    status_code=status.HTTP_200_OK,
)
async def get_variable_coverage(
    req: GetVariableCoverageRequest,
    user: User = Depends(parse_header),
) -> GetVariableCoverageResponse:
    """
    Retrieves coverage information for all available variables.
    Default universe SPY
    """
    return await application.state.agent_service_impl.get_variable_coverage(
        user=user, feature_ids=req.feature_ids, universe_id=req.universe_id
    )


# custom doc endpoints
@router.get(
    "/custom-documents",
    response_model=ListCustomDocumentsResponse,
    status_code=status.HTTP_200_OK,
)
async def list_custom_docs(
    user: User = Depends(parse_header),
) -> ListCustomDocumentsResponse:
    """
    Gets custom document file content as a byte stream
    Args:
        file_id (str): the file's ID
    """
    try:
        return await application.state.agent_service_impl.list_custom_documents(user=user)
    except CustomDocumentException as e:
        logger.exception("Error while listing custom docs")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=e.message)


@router.get(
    "/custom-documents/{file_id}/download",
    response_class=Response,
    status_code=status.HTTP_200_OK,
)
async def get_custom_doc_file(
    file_id: str,
    preview: bool = False,
    user: User = Depends(parse_header),
) -> Response:
    """
    Gets custom document file content as a byte stream
    Args:
        file_id (str): the file's ID
    Query Params:
        preview (bool): whether to return a previewable version of the file
                        ie: a PDF for files more complex than txt.
    """
    try:
        resp = await application.state.agent_service_impl.get_custom_doc_file_content(
            user=user, file_id=file_id, return_previewable_file=preview
        )
        return Response(content=resp.content, media_type=resp.file_type)
    except CustomDocumentException as e:
        logger.exception(f"Error while downloading custom doc {file_id}; {preview=}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=e.message
        ) from e


@router.get(
    "/custom-documents/{file_id}/info",
    response_model=GetCustomDocumentFileInfoResponse,
    status_code=status.HTTP_200_OK,
)
async def get_custom_doc_details(
    file_id: str, user: User = Depends(parse_header)
) -> GetCustomDocumentFileInfoResponse:
    """
    Gets custom document details
    Args:
        file_id (str): the file's ID
    """
    try:
        return await application.state.agent_service_impl.get_custom_doc_file_info(
            user=user, file_id=file_id
        )
    except CustomDocumentException as e:
        logger.exception(f"Error while getting custom doc metadata {file_id}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=e.message)


@router.post(
    "/stock/get-ordered-securities",
    response_model=GetOrderedSecuritiesResponse,
    status_code=status.HTTP_200_OK,
)
async def get_ordered_securities(
    req: GetOrderedSecuritiesRequest, user: User = Depends(parse_header)
) -> GetOrderedSecuritiesResponse:
    return await application.state.agent_service_impl.get_ordered_securities(req, user)


@router.get(
    "/agent/qc/{id}",
    response_model=List[AgentQCResponse],
    status_code=status.HTTP_200_OK,
)
async def get_qc_agent_by_id(id: str, user: User = Depends(parse_header)) -> List[AgentQCResponse]:
    """
    Get QC Agent by ID

    Args:
        id (UUID4): The ID of the agent QC.

    Returns:
        A list of AgentQC objects.
    """
    # Validate user access to QC tool
    if not user_has_qc_tool_access(user_id=user.user_id):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="User is not authorized to use QC tool"
        )

    # Call the function to retrieve agent QC by ID
    agent_qcs = await application.state.agent_service_impl.get_agent_qc_by_id([id])

    # Return the list of AgentQC objects
    return [AgentQCResponse(agent_qc=agent_qc) for agent_qc in agent_qcs]


@router.get(
    "/agent/qc/user/{user_id}",
    response_model=List[AgentQCResponse],
    status_code=status.HTTP_200_OK,
)
async def get_qc_agent_by_user(
    user_id: str, user: User = Depends(parse_header)
) -> List[AgentQCResponse]:
    """
    Get QC Agents by User ID

    Args:
        user_id (UUID4): The ID of the user whose QC agents are being requested.

    Returns:
        A list of AgentQC objects.
    """
    # Validate user access to QC tool
    if not user_has_qc_tool_access(user_id=user.user_id):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="User is not authorized to use QC tool"
        )

    # Call the function to retrieve agent QCs by user_id
    agent_qcs = await application.state.agent_service_impl.get_agent_qc_by_user(user_id)

    # Return the list of AgentQC objects
    return [AgentQCResponse(agent_qc=agent_qc) for agent_qc in agent_qcs]


@router.post(
    "/agent/qc/search",
    response_model=SearchAgentQCResponse,
    status_code=status.HTTP_200_OK,
)
async def search_agent_qc(
    req: SearchAgentQCRequest, user: User = Depends(parse_header)
) -> SearchAgentQCResponse:
    """
    Search Agent QC records based on various filters

    Args:
        start_date (Optional[date]): The date to filter by.
        end_date (Optional[date]): The date to filter by.
        use_case (Optional[str]): The use case to filter by.
        score_rating (Optional[int]): The score rating to filter by.
        tool_failed (Optional[bool]): Whether the tool failed.
        problem_type (Optional[str]): The type of problem to filter by.

    Returns:
        A list of AgentQC records matching the search criteria.
    """
    # Validate user access to QC tool
    if not user_has_qc_tool_access(user_id=user.user_id):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="User is not authorized to use QC tool"
        )

    # Call the service function to search based on the filters
    agent_qcs, total_agent_qcs = await application.state.agent_service_impl.search_agent_qcs(
        search_params=req.search_criteria,
        pagination=req.pagination,
    )

    # Return the list of AgentQC records in the response model format
    return SearchAgentQCResponse(agent_qcs=agent_qcs, total_agent_qcs=total_agent_qcs)


@router.post(
    "/agent/qc/get-live-agents",
    response_model=GetLiveAgentsQCResponse,
    status_code=status.HTTP_200_OK,
)
async def get_live_agents_qc(
    req: GetAgentsQCRequest, user: User = Depends(parse_header)
) -> GetLiveAgentsQCResponse:
    # Validate user access to QC tool
    if not user_has_qc_tool_access(user_id=user.user_id):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="User is not authorized to use QC tool"
        )

    return await application.state.agent_service_impl.get_live_agents_qc(req=req)


@router.post(
    "/agent/qc/update",
    response_model=UpdateAgentQCResponse,
    status_code=status.HTTP_200_OK,
)
async def update_agent_qc(
    req: UpdateAgentQCRequest, user: User = Depends(parse_header)
) -> UpdateAgentQCResponse:
    # Validate user access to QC tool
    if not user_has_qc_tool_access(user_id=user.user_id):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="User is not authorized to use QC tool"
        )

    agent_qc = req.agent_qc
    # Call the service function to search based on the filters
    res = await application.state.agent_service_impl.update_qc_agent(agent_qc)

    # Return the list of AgentQC records in the response model format
    return UpdateAgentQCResponse(success=res)


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
    parser.add_argument(
        "-w",
        "--workers",
        type=int,
        required=False,
        default=1,
        help="Number of workers to run the server with.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    num_cpu = os.cpu_count() or 1
    num_workers = min(args.workers, num_cpu + 1)
    print(f"########## Found {num_cpu} CPUs - Thus running with {num_workers} workers ##########")

    try:
        print("Using uvloop for faster event loop.")
        uvicorn.run(
            app="application:application",
            host=args.address,
            port=args.port,
            loop="uvloop",
            workers=num_workers,
        )
    except ImportError:
        print(
            "Failed to use uvloop (either not installed or you're on Windows). "
            "Using default event loop."
        )
        uvicorn.run(
            app="application:application",
            host=args.address,
            port=args.port,
            workers=num_workers,
        )
