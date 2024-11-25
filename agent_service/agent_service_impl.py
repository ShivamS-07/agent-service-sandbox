import asyncio
import datetime
import json
import logging
import re
import time
import traceback
import uuid
from collections import defaultdict
from typing import (
    Any,
    AsyncGenerator,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
    cast,
    get_args,
    get_origin,
)
from uuid import uuid4

import pandas as pd
import pandoc
import sentry_sdk
from fastapi import HTTPException, Request, UploadFile, status
from gbi_common_py_utils.numpy_common.numpy_cube import NumpyCube
from gbi_common_py_utils.utils.environment import (
    LOCAL_TAG,
    PROD_TAG,
    STAGING_TAG,
    get_environment_tag,
)
from gpt_service_proto_v1.service_grpc import GPTServiceStub
from grpclib import GRPCError
from grpclib import Status as GrpcStatus
from stock_universe_service_proto_v1.custom_data_service_pb2 import (
    GetFileContentsResponse,
    GetFileInfoResponse,
    ListDocumentsResponse,
)

from agent_service.agent_quality_worker.jira_integration import JiraIntegration
from agent_service.canned_prompts.canned_prompts import CANNED_PROMPTS
from agent_service.chatbot.chatbot import Chatbot
from agent_service.endpoints.authz_helper import User
from agent_service.endpoints.models import (
    Account,
    AddCustomDocumentsResponse,
    AgentEvent,
    AgentHelpRequest,
    AgentInfo,
    AgentMetadata,
    AgentQC,
    AgentUserSettingsSetRequest,
    AvailableVariable,
    CannedPrompt,
    ChatWithAgentRequest,
    ChatWithAgentResponse,
    CheckCustomDocumentUploadQuotaResponse,
    CreateAgentResponse,
    CreateCustomNotificationRequest,
    CreateJiraTicketRequest,
    CreateJiraTicketResponse,
    CreatePromptTemplateResponse,
    CustomDocumentSummaryChunk,
    CustomNotification,
    Debug,
    DeleteAgentOutputRequest,
    DeleteAgentOutputResponse,
    DeleteAgentResponse,
    DeleteCustomDocumentsResponse,
    DeleteMemoryResponse,
    DeletePromptTemplateResponse,
    DisableAgentAutomationResponse,
    EnableAgentAutomationResponse,
    ExecutionPlanTemplate,
    ExperimentalGetFormulaDataResponse,
    FindTemplatesRelatedToPromptResponse,
    GenPromptTemplateFromPlanResponse,
    GenTemplatePlanResponse,
    GetAgentDebugInfoResponse,
    GetAgentFeedBackResponse,
    GetAgentOutputResponse,
    GetAgentsQCRequest,
    GetAgentTaskOutputResponse,
    GetAgentWorklogBoardResponse,
    GetAllAgentsResponse,
    GetAutocompleteItemsResponse,
    GetAvailableVariablesResponse,
    GetCannedPromptsResponse,
    GetChatHistoryResponse,
    GetCompaniesResponse,
    GetCompanyDescriptionResponse,
    GetCustomDocumentFileInfoResponse,
    GetCustomDocumentFileResponse,
    GetDebugToolArgsResponse,
    GetDebugToolResultResponse,
    GetDividendYieldResponse,
    GetEarningsSummaryResponse,
    GetExecutiveEarningsSummaryResponse,
    GetHistoricalPricesRequest,
    GetHistoricalPricesResponse,
    GetLiveAgentsQCResponse,
    GetMarketDataResponse,
    GetMemoryContentResponse,
    GetNewsSummaryRequest,
    GetNewsSummaryResponse,
    GetOrderedSecuritiesRequest,
    GetOrderedSecuritiesResponse,
    GetPlanRunDebugInfoResponse,
    GetPlanRunOutputResponse,
    GetPreviousEarningsResponse,
    GetPriceDataResponse,
    GetRealTimePriceResponse,
    GetSecureUserResponse,
    GetSecurityProsConsResponse,
    GetSecurityResponse,
    GetTestCaseInfoResponse,
    GetTestCasesResponse,
    GetTestSuiteRunInfoResponse,
    GetTestSuiteRunsResponse,
    GetToolLibraryResponse,
    GetUpcomingEarningsResponse,
    GetVariableCoverageResponse,
    GetVariableHierarchyResponse,
    HorizonCriteria,
    JiraTicketCriteria,
    ListCustomDocumentsResponse,
    ListMemoryItemsResponse,
    LockAgentOutputRequest,
    LockAgentOutputResponse,
    MarkNotificationsAsReadResponse,
    MarkNotificationsAsUnreadResponse,
    MasterSecurity,
    MediaType,
    MemoryItem,
    ModifyPlanRunArgsRequest,
    ModifyPlanRunArgsResponse,
    NotificationEmailsResponse,
    NotificationEvent,
    Pagination,
    PlanRunStatusInfo,
    PlanRunToolDebugInfo,
    PlanTemplateTask,
    RenameMemoryResponse,
    RestoreAgentResponse,
    RetryPlanRunRequest,
    RetryPlanRunResponse,
    RunTemplatePlanResponse,
    SetAgentFeedBackRequest,
    SetAgentFeedBackResponse,
    SetAgentScheduleRequest,
    SetAgentScheduleResponse,
    SharePlanRunResponse,
    Status,
    TerminateAgentResponse,
    ToolArgInfo,
    ToolMetadata,
    ToolPromptInfo,
    Tooltips,
    UnlockAgentOutputRequest,
    UnlockAgentOutputResponse,
    UnsharePlanRunResponse,
    UpdateAgentDraftStatusResponse,
    UpdateAgentRequest,
    UpdateAgentResponse,
    UpdateAgentWidgetNameRequest,
    UpdateAgentWidgetNameResponse,
    UpdatePromptTemplateResponse,
    UpdateUserResponse,
    UploadFileResponse,
    ValidateArgError,
    VariableHierarchyNode,
)
from agent_service.endpoints.utils import (
    get_agent_hierarchical_worklogs,
    get_plan_preview,
)
from agent_service.external.custom_data_svc_client import (
    check_custom_doc_upload_quota,
    delete_custom_docs,
    document_listing_status_to_str,
    get_custom_doc_file_contents,
    get_custom_doc_file_info,
    list_custom_docs,
    process_uploaded_s3_documents,
)
from agent_service.external.feature_coverage_svc_client import (
    get_feature_coverage_client,
)
from agent_service.external.feature_svc_client import (
    experimental_get_formula_data,
    get_all_variable_hierarchy,
    get_all_variables_metadata,
)
from agent_service.external.pa_svc_client import get_all_watchlists, get_all_workspaces
from agent_service.external.user_svc_client import (
    get_user_cached,
    get_users,
    list_team_members,
    update_user,
)
from agent_service.external.webserver import (
    get_company_description,
    get_earnings_summary,
    get_executive_earnings_summary,
    get_ordered_securities,
    get_previous_earnings,
    get_security,
    get_security_pros_cons,
    get_stock_dividend_yield,
    get_stock_historical_prices,
    get_stock_market_data,
    get_stock_news_summary,
    get_stock_price_data,
    get_stock_real_time_price,
    get_upcoming_earnings,
)
from agent_service.io_type_utils import (
    TableColumnType,
    get_clean_type_name,
    load_io_type,
)
from agent_service.io_types.citations import CitationDetailsType, CitationType
from agent_service.io_types.stock import StockID
from agent_service.io_types.table import (
    STOCK_ID_COL_NAME_DEFAULT,
    StockTable,
    TableColumnMetadata,
)
from agent_service.io_types.text_objects import (
    PortfolioTextObject,
    VariableTextObject,
    WatchlistTextObject,
    extract_ordered_text_object_parts_from_text,
    extract_text_objects_from_text,
)
from agent_service.planner.action_decide import FirstActionDecider
from agent_service.planner.constants import CHAT_DIFF_TEMPLATE, FirstAction
from agent_service.planner.planner import Planner
from agent_service.planner.planner_types import ExecutionPlan, PlanStatus, Variable
from agent_service.slack.slack_sender import SlackSender, get_user_info_slack_string
from agent_service.tool import ToolCategory, ToolRegistry
from agent_service.tools.graphs import MakeLineGraphArgs, make_line_graph
from agent_service.types import (
    AgentUserSettingsSource,
    ChatContext,
    MemoryType,
    Message,
    MessageMetadata,
    MessageSpecialFormatting,
    PlanRunContext,
)
from agent_service.uploads import UploadHandler
from agent_service.utils.agent_event_utils import (
    publish_agent_execution_plan,
    publish_agent_name,
    publish_agent_plan_status,
    send_chat_message,
    update_agent_help_requested,
)
from agent_service.utils.agent_name import generate_name_for_agent
from agent_service.utils.async_db import AsyncDB
from agent_service.utils.async_utils import (
    async_wrap,
    gather_with_concurrency,
    run_async_background,
)
from agent_service.utils.cache_utils import CacheBackend, RedisCacheBackend
from agent_service.utils.clickhouse import Clickhouse
from agent_service.utils.constants import MEDIA_TO_MIMETYPE
from agent_service.utils.custom_documents_utils import (
    CustomDocumentException,
    CustomDocumentQuotaExceededException,
    custom_doc_listing_proto_to_model,
)
from agent_service.utils.date_utils import get_now_utc
from agent_service.utils.event_logging import log_event
from agent_service.utils.feature_flags import (
    get_custom_user_dict,
    get_ld_flag_async,
    get_secure_mode_hash,
    get_user_context_async,
    is_database_access_check_enabled_for_user,
)
from agent_service.utils.find_template import get_matched_templates
from agent_service.utils.get_agent_outputs import get_agent_output
from agent_service.utils.logs import async_perf_logger
from agent_service.utils.memory_handler import MemoryHandler, get_handler
from agent_service.utils.output_utils.output_construction import get_output_from_io_type
from agent_service.utils.plan_to_template import generate_template_from_plan
from agent_service.utils.postgres import DEFAULT_AGENT_NAME
from agent_service.utils.prefect import kick_off_run_execution_plan
from agent_service.utils.prompt_template import PromptTemplate
from agent_service.utils.redis_queue import (
    get_agent_event_channel,
    get_notification_event_channel,
    wait_for_messages,
)
from agent_service.utils.s3_upload import S3FileUploadTuple, async_upload_files_to_s3
from agent_service.utils.scheduling import (
    AgentSchedule,
    get_schedule_from_user_description,
)
from agent_service.utils.sidebar_sections import SidebarSection, find_sidebar_section
from agent_service.utils.task_executor import TaskExecutor
from agent_service.utils.variable_explorer_utils import (
    VariableExplorerBadInputError,
    VariableExplorerException,
)

LOGGER = logging.getLogger(__name__)

EDITABLE_TYPE = {
    int,
    str,
    float,
    bool,
    Optional[int],
    Optional[str],
    Optional[float],
    Optional[bool],
    List[int],
    List[str],
    List[float],
    List[bool],
}

PRIMITIVE_TYPES = {
    int: int,
    str: str,
    float: float,
    bool: bool,
}

OPTIONAL_TYPES = {
    Optional[int]: int,
    Optional[str]: str,
    Optional[float]: float,
    Optional[bool]: bool,
}

LIST_TYPES = {
    List[int]: int,
    List[str]: str,
    List[float]: float,
    List[bool]: bool,
}


class AgentServiceImpl:
    def __init__(
        self,
        task_executor: TaskExecutor,
        gpt_service_stub: GPTServiceStub,
        async_db: AsyncDB,
        clickhouse_db: Clickhouse,
        slack_sender: SlackSender,
        base_url: str,
        jira_integration: JiraIntegration,
        env: str = LOCAL_TAG,
        cache: Optional[CacheBackend] = None,
    ):
        self.pg = async_db
        self.ch = clickhouse_db
        self.task_executor = task_executor
        self.gpt_service_stub = gpt_service_stub
        self.slack_sender = slack_sender
        self.base_url = base_url
        self.cache = cache
        self.jira_integration = jira_integration
        self.env = env

    async def create_agent(
        self, user: User, is_draft: bool = False, created_from_template: bool = False
    ) -> CreateAgentResponse:
        """Create an agent entry in the DB and return ID immediately"""

        now = get_now_utc()
        agent_meta = AgentMetadata(created_from_template=created_from_template)
        if user.real_user_id and user.real_user_id != user.user_id:
            agent_meta.created_while_spoofed = True
            agent_meta.real_user_id = user.real_user_id
        agent = AgentInfo(
            agent_id=str(uuid4()),
            user_id=user.user_id,
            agent_name=DEFAULT_AGENT_NAME,
            created_at=now,
            last_updated=now,
            deleted=False,
            is_draft=is_draft,
            agent_metadata=agent_meta,
        )
        await self.pg.create_agent(agent)
        return CreateAgentResponse(success=True, allow_retry=False, agent_id=agent.agent_id)

    async def update_agent_draft_status(
        self, agent_id: str, is_draft: bool = False
    ) -> UpdateAgentDraftStatusResponse:
        """Updates is_draft status of agent"""

        await self.pg.update_agent_draft_status(agent_id=agent_id, is_draft=is_draft)
        return UpdateAgentDraftStatusResponse(success=True)

    async def get_all_agents(self, user: User) -> GetAllAgentsResponse:
        agents, sections = await gather_with_concurrency(
            [self.pg.get_user_all_agents(user.user_id), self.pg.get_sidebar_sections(user.user_id)]
        )
        for i, section in enumerate(sections):
            section.index = i
        return GetAllAgentsResponse(agents=agents, sections=sections)

    async def get_agent(self, agent_id: str, is_admin: bool = False) -> AgentInfo:
        agents, cost_info = await asyncio.gather(
            self.pg.get_user_all_agents(agent_ids=[agent_id], include_deleted=is_admin),
            self.ch.get_agents_cost_info(agent_ids=[agent_id]),
        )

        if not agents:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=f"Agent {agent_id} not found"
            )

        agent_metadata = agents[0]
        agent_metadata.cost_info = cost_info.get(agent_id)

        return agent_metadata

    async def terminate_agent(
        self,
        agent_id: str,
        plan_id: Optional[str] = None,
        plan_run_id: Optional[str] = None,
    ) -> TerminateAgentResponse:
        """
        1. Insert `plan_id/plan_run_id` into `agent.cancelled_ids` so executor will stop when it checks
        2. Update the task status to `CANCELLED` (will be done in executor)
        3. Send an SSE to FE "Your agent has been cancelled successfully."
        """
        if not plan_id and not plan_run_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Either plan_id or plan_run_id must be provided",
            )

        # it's safer to run these in sequence
        LOGGER.info(f"Cancelling agent {agent_id=} with {plan_id=} and {plan_run_id=}")
        await self.pg.cancel_agent_plan(plan_id=plan_id, plan_run_id=plan_run_id)
        await send_chat_message(
            message=Message(
                agent_id=agent_id,
                message="Analyst has been cancelled successfully.",
                is_user_message=False,
                visible_to_llm=False,
                plan_run_id=plan_run_id,
            ),
            db=self.pg,
        )
        return TerminateAgentResponse(success=True)

    async def delete_agent(self, agent_id: str) -> DeleteAgentResponse:
        await self.pg.delete_agent_by_id(agent_id)  # soft delete
        await send_chat_message(
            message=Message(
                agent_id=agent_id,
                message="Analyst has been deleted successfully.",
                is_user_message=False,
                visible_to_llm=False,
            ),
            db=self.pg,
        )
        return DeleteAgentResponse(success=True)

    async def restore_agent(self, agent_id: str) -> RestoreAgentResponse:
        await self.pg.restore_agent_by_id(agent_id)
        await send_chat_message(
            message=Message(
                agent_id=agent_id,
                message="Analyst has been restored successfully.",
                is_user_message=False,
                visible_to_llm=False,
            ),
            db=self.pg,
        )
        return RestoreAgentResponse(success=True)

    async def update_agent(self, agent_id: str, req: UpdateAgentRequest) -> UpdateAgentResponse:
        await self.pg.update_agent_name(agent_id, req.agent_name)
        await send_chat_message(
            message=Message(
                agent_id=agent_id,
                message=f"Analyst's name has been updated to <{req.agent_name}>.",
                is_user_message=False,
                visible_to_llm=False,
            ),
            db=self.pg,
        )
        return UpdateAgentResponse(success=True)

    async def set_agent_help_requested(
        self, agent_id: str, req: AgentHelpRequest, requesting_user: User
    ) -> UpdateAgentResponse:

        owner_user_id = await self.pg.get_agent_owner(agent_id=agent_id)
        if not owner_user_id:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Agent not found")

        message_metadata = MessageMetadata()
        if req.is_help_requested:
            message_metadata.formatting = MessageSpecialFormatting.HELP_REQUESTED
            message = (
                "I've alerted a human to help with your request. I will notify you when I "
                "have an update."
            )
        else:
            message = "Help request cancelled."
            if req.resolved_by_cs:
                message_metadata.formatting = MessageSpecialFormatting.HELP_RESOLVED
                message = "A human has addressed the issue I was having. I should be all set!"
        message_obj = Message(
            agent_id=agent_id,
            message=message,
            is_user_message=False,
            visible_to_llm=False,
            message_metadata=message_metadata,
        )
        await update_agent_help_requested(
            agent_id=agent_id,
            user_id=owner_user_id,
            help_requested=req.is_help_requested,
            db=self.pg,
            send_message=message_obj,
            slack_sender=self.slack_sender,
            send_slack_if_requested=True,
            requesting_user=requesting_user,
        )

        return UpdateAgentResponse(success=True)

    async def get_all_agent_notification_criteria(self, agent_id: str) -> List[CustomNotification]:
        return await self.pg.get_all_agent_custom_notifications(agent_id=agent_id)

    async def create_agent_notification_criteria(self, req: CreateCustomNotificationRequest) -> str:
        cn = CustomNotification(
            agent_id=req.agent_id, notification_prompt=req.notification_prompt, auto_generated=False
        )
        await self.pg.insert_agent_custom_notification(cn=cn)

        await send_chat_message(
            db=self.pg,
            message=Message(
                agent_id=req.agent_id,
                message=(
                    f"Notification criteria <{req.notification_prompt}> "
                    "has been created successfully."
                ),
                is_user_message=False,
                visible_to_llm=False,
            ),
        )

        return cn.custom_notification_id

    async def delete_agent_notification_criteria(
        self, agent_id: str, custom_notification_id: str
    ) -> None:
        await self.pg.delete_agent_custom_notification_prompt(
            agent_id=agent_id, custom_notification_id=custom_notification_id
        )
        await send_chat_message(
            db=self.pg,
            message=Message(
                agent_id=agent_id,
                message="Notification criteria has been deleted successfully.",
                is_user_message=False,
                visible_to_llm=False,
            ),
        )

    async def get_agent_notification_emails(self, agent_id: str) -> NotificationEmailsResponse:
        emails = await self.pg.get_agent_subscriptions(agent_id=agent_id)
        # this is to ensure that even if there's a duplicates in the db we do not
        # return the duplicates
        emails_to_return = []
        already_seen = set()
        for email in emails:
            if email.email not in already_seen:
                already_seen.add(email.email)
                emails_to_return.append(email.email)
        return NotificationEmailsResponse(emails=emails_to_return)

    async def set_agent_notification_emails(
        self, agent_id: str, emails: List[str], user_id: str
    ) -> None:
        """

        Args:
            agent_id: string agent id
            emails: List of emails we want to add
            user_id: string user id

        Returns: None

        """
        # get valid users
        valid_users = await self.get_valid_notification_users(user_id=user_id)

        valid_users_to_email = {user.email: user for user in valid_users}
        email_to_user_id = {}
        for email in emails:
            user = valid_users_to_email.get(email, None)
            if user:
                email_to_user_id[email] = user.user_id
            else:
                email_to_user_id[email] = ""
        await self.pg.set_agent_subscriptions(agent_id=agent_id, email_to_user_id=email_to_user_id)

    async def delete_agent_notification_emails(self, agent_id: str, email: str) -> None:
        await self.pg.delete_agent_emails(agent_id=agent_id, email=email)
        await send_chat_message(
            db=self.pg,
            message=Message(
                agent_id=agent_id,
                message=f"Notification email <{email}> has been deleted successfully.",
                is_user_message=False,
                visible_to_llm=False,
            ),
        )

    async def get_valid_notification_users(self, user_id: str) -> List[Account]:
        # first we get all the teams that the user is a part of
        user = await get_user_cached(user_id, async_db=self.pg)
        if not user:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

        coroutines = [
            list_team_members(team_id=team.team_id.id, user_id=user_id)
            for team in user.team_memberships
        ]
        results = await asyncio.gather(*coroutines)

        # Flatten the list of results into valid_users
        accounts: List[Account] = []
        for result in results:
            accounts.extend(
                Account(
                    user_id=user.user_id.id,
                    username=user.username,
                    name=user.name,
                    email=user.email,
                )
                for user in result
            )
        return accounts

    @async_perf_logger
    async def agent_quality_ingestion(
        self,
        agent_id: str,
        prompt: str,
        user: User,
        plan_id: str,
        is_spoofed: Optional[bool] = False,
    ) -> None:
        agent_quality_id = str(uuid4())
        try:
            agent_qc = AgentQC(
                agent_qc_id=agent_quality_id,
                agent_id=agent_id,
                query=prompt,
                user_id=user.user_id,
                created_at=datetime.datetime.now(),
                last_updated=datetime.datetime.now(),
                agent_status="CS",
                query_order=0,
                plan_id=plan_id,
                fullstory_link=user.fullstory_link.replace("https://", ""),
                is_spoofed=is_spoofed if is_spoofed else False,
            )
            await self.pg.insert_agent_qc(agent_qc=agent_qc)
        except Exception:
            LOGGER.exception(f"Unable to insert in agent quality db for  {agent_id=}")

    @async_perf_logger
    async def chat_with_agent(self, req: ChatWithAgentRequest, user: User) -> ChatWithAgentResponse:
        agent_id = req.agent_id
        try:
            LOGGER.info(f"Inserting user's new message to DB for {agent_id=}")
            user_msg = Message(
                agent_id=agent_id,
                message=req.prompt,
                is_user_message=True,
                message_author=user.real_user_id,
            )
            await self.pg.insert_chat_messages(messages=[user_msg])
        except Exception as e:
            LOGGER.exception(
                f"Failed to insert user message into DB for {agent_id=} with exception: {e}"
            )
            return ChatWithAgentResponse(success=False, allow_retry=True)

        if req.is_first_prompt:
            LOGGER.info(
                f"Creating future task to generate agent name and store in DB for {agent_id=}"
            )
            run_async_background(
                self._generate_agent_name_and_store(user.user_id, agent_id, user_msg)  # 2s
            )

            if not req.skip_agent_response:
                LOGGER.info(
                    (
                        "Creating future tasks to generate initial "
                        f"response and send slack message, {agent_id=}"
                    )
                )
                run_async_background(self._slack_chat_msg(req, user))  # 0.1s
                await self._create_initial_response(user, agent_id, user_msg)  # 2s

            if req.canned_prompt_id:
                log_event(
                    event_name="agent-canned-prompt",
                    event_data={
                        "agent_id": req.agent_id,
                        "canned_prompt_id": req.canned_prompt_id,
                        "canned_prompt_text": req.prompt,
                        "user_id": user.user_id,
                    },
                )
        else:
            try:
                LOGGER.info(f"Updating execution plan after user's new message for {req.agent_id=}")
                await self.task_executor.update_execution_after_input(  # >3s
                    agent_id=req.agent_id, user_id=user.user_id, chat_context=None
                )
            except Exception as e:
                LOGGER.exception((f"Failed to update {agent_id=} execution plan: {e}"))
                return ChatWithAgentResponse(success=False, allow_retry=False)

        return ChatWithAgentResponse(success=True, allow_retry=False)

    @async_perf_logger
    async def _generate_agent_name_and_store(
        self, user_id: str, agent_id: str, user_msg: Message
    ) -> str:
        LOGGER.info("Getting existing agents' names")
        existing_agents = await self.pg.get_existing_agents_names(user_id)

        LOGGER.info("Calling GPT to generate agent name")
        name = await generate_name_for_agent(
            agent_id=agent_id,
            chat_context=ChatContext(messages=[user_msg]),
            existing_names=existing_agents,
            gpt_service_stub=self.gpt_service_stub,
            user_id=user_id,
        )

        LOGGER.info(f"Updating agent name to {name} in DB")

        await asyncio.gather(
            self.pg.update_agent_name(agent_id=agent_id, agent_name=name),
            publish_agent_name(agent_id=agent_id, agent_name=name),
        )
        return name

    @async_perf_logger
    async def _create_initial_response(
        self, user: User, agent_id: str, user_msg: Message
    ) -> FirstAction:
        LOGGER.info(f"Generating initial response from GPT (first prompt), {agent_id=}")
        chatbot = Chatbot(agent_id, gpt_service_stub=self.gpt_service_stub)
        chat_context = ChatContext(messages=[user_msg])
        action_decider = FirstActionDecider(agent_id=agent_id, skip_db_commit=True)

        LOGGER.info(f"Deciding action for {agent_id=}")
        action = await action_decider.decide_action(chat_context=chat_context)

        LOGGER.info(f"Action decided: {action}. Now generating response for {agent_id=}")
        if action == FirstAction.REFER:
            gpt_resp = await chatbot.generate_first_response_refer(chat_context=chat_context)
        elif action == FirstAction.NOTIFICATION:
            gpt_resp = await chatbot.generate_first_response_notification(chat_context=chat_context)
        elif action == FirstAction.PLAN:
            gpt_resp = await chatbot.generate_initial_preplan_response(chat_context=chat_context)
        else:  # action == FirstAction.NONE
            gpt_resp = await chatbot.generate_first_response_none(chat_context=chat_context)

        gpt_msg = Message(agent_id=agent_id, message=gpt_resp, is_user_message=False)

        LOGGER.info(f"Inserting GPT's response to DB and publishing GPT response for {agent_id=}")
        tasks = [
            self.pg.insert_chat_messages(messages=[gpt_msg]),
            send_chat_message(gpt_msg, self.pg, insert_message_into_db=False),
        ]

        if action == FirstAction.PLAN:
            plan_id = str(uuid4())
            LOGGER.info(f"Creating execution plan {plan_id} for {agent_id=}")
            tasks.extend(
                (
                    self.task_executor.create_execution_plan(  # this isn't async
                        agent_id=agent_id,
                        plan_id=plan_id,
                        user_id=user.user_id,
                        run_plan_in_prefect_immediately=True,
                    ),
                    publish_agent_plan_status(
                        agent_id=agent_id, plan_id=plan_id, status=PlanStatus.CREATING, db=self.pg
                    ),
                )
            )
            if await get_ld_flag_async(
                flag_name="qc_tool_flag",
                default=False,
                user_id=user.user_id,
                async_db=self.pg,
            ):
                prompt: str = str(user_msg.message)
                is_spoofed = user_msg.is_user_message and user_msg.message_author != user.user_id

                run_async_background(
                    self.agent_quality_ingestion(
                        agent_id=agent_id,
                        prompt=prompt,
                        user=user,
                        plan_id=plan_id,
                        is_spoofed=is_spoofed,
                    )
                )

        await asyncio.gather(*tasks)
        return action

    @async_perf_logger
    async def _slack_chat_msg(self, req: ChatWithAgentRequest, user: User) -> None:
        try:
            user_email, user_info_slack_string = await get_user_info_slack_string(
                self.pg, user.user_id
            )
            if (
                not user_email.endswith("@boosted.ai")
                and not user_email.endswith("@gradientboostedinvestments.com")
                and user.user_id == user.real_user_id
            ):
                if user.fullstory_link:
                    user_info_slack_string += (
                        f"\nfullstory_link: {user.fullstory_link.replace('https://', '')}"
                    )
                six_hours_from_now = int(time.time() + (60 * 60 * 2))
                await self.slack_sender.send_message_async(
                    message_text=f"{req.prompt}\n"
                    f"Link: {self.base_url}/chat/{req.agent_id}\n"
                    f"canned_prompt_id: {req.canned_prompt_id}\n"
                    f"{user_info_slack_string}",
                    send_at=six_hours_from_now,
                )
        except Exception:
            LOGGER.warning(f"Unable to send slack message for {user.user_id=}")
            LOGGER.warning(traceback.format_exc())

    async def get_chat_history(
        self,
        agent_id: str,
        start: Optional[datetime.datetime],
        end: Optional[datetime.datetime],
        start_index: Optional[int] = 0,
        limit_num: Optional[int] = None,
    ) -> GetChatHistoryResponse:
        chat_context = await self.pg.get_chats_history_for_agent(
            agent_id, start, end, start_index, limit_num
        )

        report_updated_message = CHAT_DIFF_TEMPLATE.split("\n")[0]
        for message in chat_context.messages:
            if not message.is_user_message and message.message.startswith(  # type: ignore
                report_updated_message
            ):
                # get the first two words
                report_updated_text = " ".join(CHAT_DIFF_TEMPLATE.split()[:2])
                report_updated_dict = {
                    "type": "output_report",
                    "text": report_updated_text,
                    "plan_run_id": message.plan_run_id,
                }
                message.message = message.message.replace(  # type: ignore
                    report_updated_text, "```" + json.dumps(report_updated_dict) + "```"  # type: ignore
                )

        return GetChatHistoryResponse(
            messages=chat_context.messages,
            total_message_count=chat_context.total_message_count,
            start_index=start_index,
        )

    @async_perf_logger
    async def get_agent_worklog_board(
        self,
        agent_id: str,
        start_date: Optional[datetime.date] = None,
        end_date: Optional[datetime.date] = None,
        start_index: Optional[int] = 0,
        limit_num: Optional[int] = None,
    ) -> GetAgentWorklogBoardResponse:
        """Get agent worklogs to build the Work Log Board
        Except `agent_id`, all other arguments are optional and can be used to filter the work log, but
        strongly recommend to have at least 1 filter to avoid returning too many entries.
        NOTE: If any of `start_date` or `end_date` is provided, and `limit_num` is also provided,
        it will be most recent N entries within the date range

        Args:
            agent_id (str): agent ID
            start (Optional[datetime.date]): start DATE to filter work log, inclusive
            end (Optional[datetime.date]): end DATE to filter work log, inclusive
            start_index (Optional[int]): start index to filter work log
            limit_num (Optional[int]): number of plan runs to return
        """
        LOGGER.info("Creating a future to get latest execution plan")
        future_task = run_async_background(self.pg.get_latest_execution_plan(agent_id))

        LOGGER.info("Getting agent worklogs")
        run_history, total_plan_count = await get_agent_hierarchical_worklogs(
            agent_id, self.pg, start_date, end_date, start_index, limit_num
        )

        LOGGER.info("Waiting for getting latest execution plan (future) to complete")
        (
            plan_id,
            execution_plan,
            _,
            status,
            upcoming_plan_run_id,
        ) = await future_task

        if (
            plan_id is None
            or execution_plan is None
            or status not in (Status.RUNNING.value, Status.NOT_STARTED.value)
        ):
            # Only set the execution plan template if the current status is
            # NOT_STARTED or RUNNING
            execution_plan_template = None
        else:
            execution_plan_template = ExecutionPlanTemplate(
                plan_id=plan_id,
                upcoming_plan_run_id=upcoming_plan_run_id,
                tasks=[
                    PlanTemplateTask(task_id=node.tool_task_id, task_name=node.description)
                    for node in execution_plan.nodes
                ],
                preview=get_plan_preview(execution_plan),
            )

        return GetAgentWorklogBoardResponse(
            run_history=run_history,
            execution_plan_template=execution_plan_template,
            latest_plan_status=status,
            total_plan_count=total_plan_count,
            start_index=start_index,
        )

    async def get_agent_task_output(
        self, agent_id: str, plan_run_id: str, task_id: str
    ) -> GetAgentTaskOutputResponse:
        task_output = await self.pg.get_task_output(
            agent_id=agent_id, plan_run_id=plan_run_id, task_id=task_id
        )

        return GetAgentTaskOutputResponse(
            output=await get_output_from_io_type(task_output, pg=self.pg.pg) if task_output else None  # type: ignore
        )

    async def get_agent_log_output(
        self, agent_id: str, plan_run_id: str, log_id: str
    ) -> GetAgentTaskOutputResponse:
        log_output = await self.pg.get_log_output(
            agent_id=agent_id, plan_run_id=plan_run_id, log_id=log_id
        )

        return GetAgentTaskOutputResponse(
            output=await get_output_from_io_type(log_output, pg=self.pg.pg) if log_output else None  # type: ignore
        )

    async def get_agent_plan_output(
        self, agent_id: str, plan_run_id: Optional[str] = None
    ) -> GetAgentOutputResponse:
        return await get_agent_output(
            pg=self.pg,
            agent_id=agent_id,
            plan_run_id=plan_run_id,
            cache=cast(RedisCacheBackend, self.cache),
        )

    async def delete_agent_output(
        self, agent_id: str, req: DeleteAgentOutputRequest
    ) -> DeleteAgentOutputResponse:

        plan_map = await self.pg.get_execution_plans(plan_ids=[req.plan_id])
        if not plan_map:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Plan ID not found")

        plan = plan_map[req.plan_id][0]
        current_deleted = set(plan.deleted_task_ids)
        to_delete = set(req.task_ids)
        plan.deleted_task_ids = list(current_deleted.union(to_delete))
        write_plan_task = self.pg.write_execution_plan(
            plan_id=req.plan_id, agent_id=agent_id, plan=plan
        )

        # 4. Delete outputs
        delete_outputs_task = self.pg.delete_agent_outputs(
            agent_id=agent_id, output_ids=req.output_ids
        )

        await asyncio.gather(write_plan_task, delete_outputs_task)

        await send_chat_message(
            db=self.pg,
            message=Message(
                agent_id=agent_id,
                message="Outputs have been deleted successfully.",
                is_user_message=False,
                visible_to_llm=False,
            ),
        )

        return DeleteAgentOutputResponse()

    async def lock_agent_output(
        self, agent_id: str, req: LockAgentOutputRequest
    ) -> LockAgentOutputResponse:
        old_plan_id, _, _, _, _ = await self.pg.get_latest_execution_plan(
            agent_id=agent_id, only_finished_plans=True
        )
        if not old_plan_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="Cannot lock output without plan"
            )
        await self.pg.lock_plan_tasks(agent_id=agent_id, plan_id=old_plan_id, task_ids=req.task_ids)

        await send_chat_message(
            db=self.pg,
            message=Message(
                agent_id=agent_id,
                message="Outputs have been locked successfully.",
                is_user_message=False,
                visible_to_llm=False,
            ),
        )

        return LockAgentOutputResponse(success=True)

    async def unlock_agent_output(
        self, agent_id: str, req: UnlockAgentOutputRequest
    ) -> UnlockAgentOutputResponse:
        old_plan_id, _, _, _, _ = await self.pg.get_latest_execution_plan(
            agent_id=agent_id, only_finished_plans=True
        )
        if not old_plan_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="Cannot unlock output without plan"
            )
        await self.pg.unlock_plan_tasks(
            agent_id=agent_id, plan_id=old_plan_id, task_ids=req.task_ids
        )

        await send_chat_message(
            db=self.pg,
            message=Message(
                agent_id=agent_id,
                message="Outputs have been unlocked successfully.",
                is_user_message=False,
                visible_to_llm=False,
            ),
        )

        return UnlockAgentOutputResponse(success=True)

    async def get_citation_details(
        self, citation_type: CitationType, citation_id: str, user_id: str
    ) -> Optional[CitationDetailsType]:
        if isinstance(citation_type, str):
            # Just to be safe
            citation_type = CitationType(citation_type)

        citation_class = citation_type.to_citation_class()
        citation_details = await citation_class.get_citation_details(
            citation_id=citation_id, db=self.pg.pg, user_id=user_id
        )
        return citation_details

    async def stream_agent_events(
        self, request: Request, agent_id: str
    ) -> AsyncGenerator[AgentEvent, None]:
        LOGGER.info(f"Listening to events on channel for {agent_id=}")
        async with get_agent_event_channel(agent_id=agent_id) as channel:
            async for message in wait_for_messages(channel, request):
                resp = AgentEvent.model_validate_json(message)
                LOGGER.info(
                    f"Got event on channel for {agent_id=} of type '{resp.event.event_type.value}'"
                )
                yield resp

    async def stream_notification_events(
        self, request: Request, user_id: str
    ) -> AsyncGenerator[NotificationEvent, None]:
        LOGGER.info(f"Listening to notification events on channel for {user_id=}")
        async with get_notification_event_channel(user_id=user_id) as channel:
            async for message in wait_for_messages(channel, request):
                resp = NotificationEvent.model_validate_json(message)
                LOGGER.info(
                    f"Got event on notification channel for {user_id=} of type '{resp.event.event_type.value}'"
                )
                yield resp

    async def share_plan_run(self, plan_run_id: str) -> SharePlanRunResponse:
        await self.pg.set_plan_run_share_status(plan_run_id=plan_run_id, status=True)
        return SharePlanRunResponse(success=True)

    async def unshare_plan_run(self, plan_run_id: str) -> UnsharePlanRunResponse:
        await self.pg.set_plan_run_share_status(plan_run_id=plan_run_id, status=False)
        return UnsharePlanRunResponse(success=True)

    async def mark_notifications_as_read(
        self, agent_id: str, timestamp: Optional[datetime.datetime]
    ) -> MarkNotificationsAsReadResponse:
        await self.pg.mark_notifications_as_read(agent_id=agent_id, timestamp=timestamp)
        return MarkNotificationsAsReadResponse(success=True)

    async def mark_notifications_as_unread(
        self, agent_id: str, message_id: str
    ) -> MarkNotificationsAsUnreadResponse:
        await self.pg.mark_notifications_as_unread(agent_id=agent_id, message_id=message_id)
        return MarkNotificationsAsUnreadResponse(success=True)

    async def enable_agent_automation(
        self, agent_id: str, user_id: str
    ) -> EnableAgentAutomationResponse:
        await self.pg.set_agent_automation_enabled(agent_id=agent_id, enabled=True)
        schedule = await self.pg.get_agent_schedule(agent_id=agent_id)
        if not schedule:
            schedule = AgentSchedule.default()
            await self.pg.update_agent_schedule(agent_id=agent_id, schedule=schedule)
        next_run = schedule.get_next_run()
        await self.pg.set_latest_plan_for_automated_run(agent_id=agent_id)

        # Now automatically subscribe the agent owner to email notification
        # get the email for the user
        user = await get_user_cached(user_id, async_db=self.pg)
        if not user:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

        await self.pg.set_agent_subscriptions(
            agent_id=agent_id,
            email_to_user_id={user.email: user_id},
            delete_previous_emails=False,
        )

        await send_chat_message(
            db=self.pg,
            message=Message(
                agent_id=agent_id,
                message="Automation has been enabled successfully.",
                is_user_message=False,
                visible_to_llm=False,
            ),
        )

        return EnableAgentAutomationResponse(success=True, next_run=next_run)

    async def disable_agent_automation(
        self, agent_id: str, user_id: str
    ) -> DisableAgentAutomationResponse:
        await self.pg.set_agent_automation_enabled(agent_id=agent_id, enabled=False)

        await send_chat_message(
            db=self.pg,
            message=Message(
                agent_id=agent_id,
                message="Automation has been disabled successfully.",
                is_user_message=False,
                visible_to_llm=False,
            ),
        )

        return DisableAgentAutomationResponse(success=True)

    async def set_agent_schedule(self, req: SetAgentScheduleRequest) -> SetAgentScheduleResponse:
        schedule, success, error_msg = await get_schedule_from_user_description(
            agent_id=req.agent_id, user_desc=req.user_schedule_description
        )
        if success:
            await self.pg.update_agent_schedule(agent_id=req.agent_id, schedule=schedule)

            await send_chat_message(
                db=self.pg,
                message=Message(
                    agent_id=req.agent_id,
                    message=f"Schedule has been set to <{req.user_schedule_description}> successfully.",
                    is_user_message=False,
                    visible_to_llm=False,
                ),
            )

        return SetAgentScheduleResponse(
            agent_id=req.agent_id,
            schedule=schedule,
            success=success,
            error_msg=error_msg,
        )

    async def retry_plan_run(self, req: RetryPlanRunRequest) -> RetryPlanRunResponse:
        tasks = [
            self.pg.get_plan_run_statuses(plan_run_ids=[req.plan_run_id]),
            self.pg.get_execution_plan_for_run(plan_run_id=req.plan_run_id),
            self.pg.get_agent_owner(agent_id=req.agent_id),
        ]

        statuses: Dict[str, PlanRunStatusInfo]
        plan_id: str
        plan: ExecutionPlan
        user_id: Optional[str]
        statuses, (plan_id, plan), user_id = await asyncio.gather(*tasks)  # type: ignore
        if not user_id:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Agent not found")

        if req.plan_run_id not in statuses:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Plan run not found")

        if statuses[req.plan_run_id].status not in (Status.ERROR, Status.NO_RESULTS_FOUND):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Plan run not in ERROR or NO_RESULTS_FOUND status!",
            )

        ctx = PlanRunContext(
            agent_id=req.agent_id,
            plan_id=plan_id,
            user_id=user_id,
            plan_run_id=req.plan_run_id,
        )
        ctx.chat = await self.pg.get_chats_history_for_agent(agent_id=ctx.agent_id)

        LOGGER.info(f"Retrying for {req.agent_id=}, {req.plan_run_id=}")
        await kick_off_run_execution_plan(
            plan=plan,
            context=ctx,
        )
        return RetryPlanRunResponse()

    async def update_agent_widget_name(
        self, agent_id: str, req: UpdateAgentWidgetNameRequest
    ) -> UpdateAgentWidgetNameResponse:
        old_widget_title = await self.pg.get_agent_widget_title(output_id=req.output_id)
        if not old_widget_title:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Widget title not found"
            )

        await self.pg.update_agent_widget_name(
            agent_id=agent_id,
            output_id=req.output_id,
            old_widget_title=old_widget_title,
            new_widget_title=req.new_widget_title,
            cache=cast(RedisCacheBackend, self.cache),
        )

        await send_chat_message(
            db=self.pg,
            message=Message(
                agent_id=agent_id,
                message="Widget name has been updated successfully.",
                is_user_message=False,
                visible_to_llm=False,
            ),
        )

        return UpdateAgentWidgetNameResponse(success=True)

    async def get_secure_ld_user(self, user_id: str) -> GetSecureUserResponse:
        ld_user = await get_user_context_async(user_id=user_id, async_db=self.pg)
        return GetSecureUserResponse(
            hash=get_secure_mode_hash(ld_user),
            context=get_custom_user_dict(ld_user),
        )

    async def list_memory_items(self, user_id: str) -> ListMemoryItemsResponse:
        # Use PA Service to get all portfolios + watchlists for the user
        workspaces, resp = await asyncio.gather(
            get_all_workspaces(user_id=user_id), get_all_watchlists(user_id=user_id)
        )
        watchlists = resp.watchlists

        memory_items = []

        # Process workspaces
        for workspace in workspaces:
            memory_item = MemoryItem(
                id=workspace.workspace_id.id,
                name=workspace.name,
                type=MemoryType.PORTFOLIO,
                time_created=workspace.created_at.ToDatetime(),
                time_updated=workspace.last_updated.ToDatetime(),
            )
            memory_items.append(memory_item)

        # Process watchlists
        for watchlist in watchlists:
            memory_item = MemoryItem(
                id=watchlist.watchlist_id.id,
                name=watchlist.name,
                type=MemoryType.WATCHLIST,
                time_created=watchlist.created_at.ToDatetime(),
                time_updated=watchlist.last_updated.ToDatetime(),
            )
            memory_items.append(memory_item)

        return ListMemoryItemsResponse(success=True, items=memory_items)

    async def get_autocomplete_items(
        self, user_id: str, text: str, memory_type: Optional[MemoryType] = None
    ) -> GetAutocompleteItemsResponse:
        workspaces = []
        memory_items = []
        watchlists = None
        text_lower = text.lower()

        if memory_type == MemoryType.PORTFOLIO:
            workspaces = await get_all_workspaces(user_id=user_id)
        elif memory_type == MemoryType.WATCHLIST:
            resp = await get_all_watchlists(user_id=user_id)
            watchlists = resp.watchlists
        else:
            # Use PA Service to get portfolios + watchlists for the user in parallel
            workspaces, resp = await asyncio.gather(
                get_all_workspaces(user_id=user_id), get_all_watchlists(user_id=user_id)
            )
            watchlists = resp.watchlists

        # Process workspaces
        for workspace in workspaces:
            if text_lower in workspace.name.lower():
                memory_item = MemoryItem(
                    id=workspace.workspace_id.id,
                    name=workspace.name,
                    type=MemoryType.PORTFOLIO,
                    time_created=workspace.created_at.ToDatetime(),
                    time_updated=workspace.last_updated.ToDatetime(),
                )
                memory_items.append(memory_item)

        # Process watchlists
        if watchlists:
            for watchlist in watchlists:
                if text_lower in watchlist.name.lower():
                    memory_item = MemoryItem(
                        id=watchlist.watchlist_id.id,
                        name=watchlist.name,
                        type=MemoryType.WATCHLIST,
                        time_created=watchlist.created_at.ToDatetime(),
                        time_updated=watchlist.last_updated.ToDatetime(),
                    )
                    memory_items.append(memory_item)

        return GetAutocompleteItemsResponse(success=True, items=memory_items)

    async def get_memory_content(
        self, user_id: str, type: str, id: str
    ) -> GetMemoryContentResponse:
        try:
            handler: MemoryHandler = get_handler(type)
            return GetMemoryContentResponse(output=await handler.get_content(user_id, id))  # type: ignore
        # catch rpc error and raise as HTTPException
        except ValueError as e:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=repr(e))

    async def delete_memory(self, user_id: str, type: str, id: str) -> DeleteMemoryResponse:
        try:
            handler: MemoryHandler = get_handler(type)
            return DeleteMemoryResponse(success=await handler.delete(user_id, id))
        # catch rpc error and raise as HTTPException
        except ValueError as e:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=repr(e))

    async def rename_memory(
        self, user_id: str, type: str, id: str, new_name: str
    ) -> RenameMemoryResponse:
        if new_name == "":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="name cannot be empty"
            )
        try:
            handler: MemoryHandler = get_handler(type)
            return RenameMemoryResponse(success=await handler.rename(user_id, id, new_name))
        # catch rpc error and raise as HTTPException
        except ValueError as e:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=repr(e))

    # Requires no authorization
    async def get_plan_run_output(
        self, agent_id: str, plan_run_id: str
    ) -> GetPlanRunOutputResponse:
        output, agent_name = await asyncio.gather(
            get_agent_output(
                pg=self.pg,
                agent_id=agent_id,
                plan_run_id=plan_run_id,
                cache=cast(RedisCacheBackend, self.cache),
            ),
            self.pg.get_agent_name(agent_id),
        )
        return GetPlanRunOutputResponse(outputs=output.outputs, agent_name=agent_name)

    async def get_agent_qc_by_ids(self, agent_qc_ids: List[str]) -> List[AgentQC]:
        agent_qcs = await self.pg.get_agent_qc_by_ids(agent_qc_ids)

        # Return the list of AgentQC objects
        return agent_qcs

    async def get_agent_qc_by_user_ids(self, user_ids: List[str]) -> List[AgentQC]:
        agent_qcs = await self.pg.get_agent_qc_by_user_ids(user_ids)

        # Return the list of AgentQC objects
        return agent_qcs

    async def get_plan_run_debug_info(
        self, agent_id: str, plan_run_id: str
    ) -> GetPlanRunDebugInfoResponse:
        """
        Get plan nodes from Postgres, `agent.execution_plans` table
        Get input and output from Clickhouse, `agent.tool_calls` table
        """
        (_, execution_plan), tool_calls, tool_prompt_infos = await asyncio.gather(
            self.pg.get_execution_plan_for_run(plan_run_id),
            self.ch.get_plan_run_debug_tool_calls(plan_run_id),
            self.ch.get_plan_run_debug_prompt_infos(plan_run_id),
        )
        task_id_to_tool_call = {tool_call["task_id"]: tool_call for tool_call in tool_calls}

        var_name_to_task: Dict[str, Dict[str, str]] = {}

        resp = GetPlanRunDebugInfoResponse(plan_run_tools=[])
        for node in execution_plan.nodes:
            task_id = node.tool_task_id

            tool_name = node.tool_name
            tool = ToolRegistry.get_tool(tool_name)

            var_name_to_task[node.output_variable_name] = {
                "task_id": task_id,
                "task_name": tool_name,
            }

            tool_call = task_id_to_tool_call.get(task_id)
            arg_values = json.loads(tool_call["args"]) if tool_call and tool_call["args"] else {}
            start_time_utc = tool_call["start_time_utc"] if tool_call else None
            end_time_utc = tool_call["end_time_utc"] if tool_call else None
            duration_seconds = tool_call["duration_seconds"] if tool_call else None

            arg_list: List[ToolArgInfo] = []
            for arg_name, info in tool.input_type.model_fields.items():
                # To be editable, first it must not be a Variable or List[Variable]
                is_editable = True
                tasks_to_depend_on: List[Dict[str, str]] = []
                vars = node.args.get(arg_name)
                if isinstance(vars, Variable):
                    is_editable = False
                    if vars.var_name in var_name_to_task:
                        tasks_to_depend_on.append(var_name_to_task[vars.var_name])
                elif isinstance(vars, list):
                    for v in vars:
                        if isinstance(v, Variable):
                            is_editable = False
                            if v.var_name in var_name_to_task:
                                tasks_to_depend_on.append(var_name_to_task[v.var_name])

                # determine arg type
                arg_value = arg_values.get(arg_name)
                clean_type_name = get_clean_type_name(info.annotation)
                if info.annotation not in EDITABLE_TYPE:
                    # `info.annotation` can be complex/union types and some of them should be editable,
                    # so we check the type of `arg_value`, e.g. Union[str, SelfDefinedType]
                    # if `arg_value` is string, we should still allow it to be a string
                    arg_value_type = type(arg_value)
                    if arg_value is None:
                        is_editable = False
                    elif arg_value_type in PRIMITIVE_TYPES:
                        clean_type_name = get_clean_type_name(arg_value_type)
                    else:
                        if arg_value_type is list:
                            if len(arg_value) == 0:
                                is_editable = False
                            else:
                                inner_type = type(arg_value[0])
                                if all(isinstance(item, inner_type) for item in arg_value):
                                    clean_type_name = f"List[{inner_type.__name__}]"
                                else:
                                    is_editable = False
                        else:
                            is_editable = False

                arg_list.append(
                    ToolArgInfo(
                        arg_name=arg_name,
                        arg_value=arg_value,
                        arg_type_name=clean_type_name,
                        required=info.is_required(),
                        is_editable=is_editable,
                        tasks_to_depend_on=tasks_to_depend_on,
                    )
                )

            tool_prompts = tool_prompt_infos.get(task_id, [])
            tool_prompt_objs = [ToolPromptInfo(**prompt) for prompt in tool_prompts]

            resp.plan_run_tools.append(
                PlanRunToolDebugInfo(
                    tool_id=task_id,
                    tool_name=tool_name,
                    tool_description=tool.description,
                    tool_comment=node.description,
                    arg_list=arg_list,
                    output_variable_name=node.output_variable_name,
                    start_time_utc=start_time_utc,
                    end_time_utc=end_time_utc,
                    duration_seconds=duration_seconds,
                    prompt_infos=tool_prompt_objs,
                )
            )

        return resp

    async def modify_plan_run_args(
        self, agent_id: str, plan_run_id: str, user_id: str, req: ModifyPlanRunArgsRequest
    ) -> ModifyPlanRunArgsResponse:
        """
        1. Download the execution plan from DB
        2. Modify the args in the execution plan. If the type mismatches, append an error into list
        3. If there's any error, return the list of errors:
            - type of arg value is not in EDITABLE_TYPE
            - type of arg value is not matched with expected type
        4. Otherwise, create a new plan entry in DB and kick off the rerun
        """
        LOGGER.info(f"Args to modify:\n{req}")

        arg_mapping: Dict[str, Dict[str, Any]] = defaultdict(dict)
        for arg in req.args_to_modify:
            arg_mapping[arg.tool_id][arg.arg_name] = arg.arg_value

        type_mismatch_error_temp = (
            "The type of arg '{arg_name}' is {arg_val_type}, "
            "not matched with expected type {expected_type}"
        )
        arg_is_variable_error_temp = "The arg '{arg_name}' is a Variable, not editable"
        errors: list[ValidateArgError] = []

        _, execution_plan = await self.pg.get_execution_plan_for_run(plan_run_id)
        for node in execution_plan.nodes:
            if node.tool_task_id not in arg_mapping:
                continue

            tool_name = node.tool_name
            tool = ToolRegistry.get_tool(tool_name)

            for arg_name, info in tool.input_type.model_fields.items():
                if arg_name not in arg_mapping[node.tool_task_id]:
                    continue

                # Check if the arg value is a Variable or a list of Variables
                if arg_name in node.args:
                    vars = node.args[arg_name]
                    if isinstance(vars, Variable):
                        errors.append(
                            ValidateArgError(
                                tool_id=node.tool_task_id,
                                arg_name=arg_name,
                                error=arg_is_variable_error_temp.format(arg_name=arg_name),
                            )
                        )
                        continue
                    elif isinstance(vars, list):
                        is_variable = False
                        for v in vars:
                            if isinstance(v, Variable):
                                errors.append(
                                    ValidateArgError(
                                        tool_id=node.tool_task_id,
                                        arg_name=arg_name,
                                        error=arg_is_variable_error_temp.format(arg_name=arg_name),
                                    )
                                )
                                is_variable = True
                                break
                        if is_variable:
                            continue

                # Validate the type of arg value and check if it's editable
                arg_val = arg_mapping[node.tool_task_id][arg_name]
                expected_type = info.annotation
                correct = True
                if expected_type in PRIMITIVE_TYPES:
                    class_type = PRIMITIVE_TYPES[expected_type]
                    correct = isinstance(arg_val, class_type)
                elif expected_type in OPTIONAL_TYPES:
                    class_type = OPTIONAL_TYPES[expected_type]  # type: ignore
                    correct = arg_val is None or isinstance(arg_val, class_type)
                elif expected_type in LIST_TYPES:
                    class_type = LIST_TYPES[expected_type]  # type: ignore
                    correct = isinstance(arg_val, list) and all(
                        isinstance(item, class_type) for item in arg_val
                    )
                else:
                    # not in any of the above types - can be Union or other types, so we check if
                    # the modified arg has the same type as the original arg's type
                    origin_type = get_origin(expected_type)
                    if origin_type is Union:
                        expected_types = get_args(expected_type)
                        correct = any(isinstance(arg_val, t) for t in expected_types)
                    else:
                        correct = False

                if not correct:
                    expected_clean_type_name = get_clean_type_name(expected_type)
                    errors.append(
                        ValidateArgError(
                            tool_id=node.tool_task_id,
                            arg_name=arg_name,
                            error=type_mismatch_error_temp.format(
                                arg_name=arg_name,
                                arg_val_type=type(arg_val),
                                expected_type=expected_clean_type_name,
                            ),
                        )
                    )
                else:
                    node.args[arg_name] = arg_val

        if errors:
            error_log = "Rejected args to be modified:\n" + "\n".join([f"{err}" for err in errors])
            LOGGER.info(error_log)
            return ModifyPlanRunArgsResponse(errors=errors)

        # change node's taskId to new taskId
        for node in execution_plan.nodes:
            node.tool_task_id = str(uuid4())

        # TODO: The problem is the `node.description` won't change when args are modified
        # e.g. "Get NVDA's prices". If you change `NVDA` to `AAPL`, the description is still the
        # same because I don't want to make another GPT call
        new_plan_id = str(uuid4())
        new_plan_run_id = str(uuid4())
        ctx = PlanRunContext(
            agent_id=agent_id,
            plan_id=new_plan_id,
            user_id=user_id,
            plan_run_id=new_plan_run_id,
        )

        await asyncio.gather(
            send_chat_message(
                db=self.pg,
                message=Message(
                    agent_id=agent_id,
                    message="New plan with modified args has been created. Running the plan now.",
                    is_user_message=False,
                    visible_to_llm=False,
                    plan_run_id=new_plan_run_id,
                ),
            ),
            publish_agent_execution_plan(execution_plan, ctx, db=self.pg),
        )

        # we need to make sure the plan is stored in DB before executing it
        await self.task_executor.run_execution_plan(plan=execution_plan, context=ctx)

        return ModifyPlanRunArgsResponse(errors=[])

    async def get_agent_debug_info(self, agent_id: str) -> GetAgentDebugInfoResponse:
        tasks = [
            self.pg.get_agent_owner(agent_id),
            self.ch.get_agent_debug_plan_selections(agent_id=agent_id),
            self.ch.get_agent_debug_plans(agent_id=agent_id),
            self.ch.get_agent_debug_worker_sqs_log(agent_id=agent_id),
            self.pg.get_agent_debug_tool_calls(agent_id=agent_id),
            self.ch.get_agent_debug_tool_calls(agent_id=agent_id),
            self.ch.get_agent_debug_cost_info(agent_id=agent_id),
            self.ch.get_agent_debug_gpt_service_info(agent_id=agent_id),
        ]

        (
            agent_owner_id,
            plan_selections,
            all_generated_plans,
            worker_sqs_log,
            tool_calls,
            ch_tool_calls,
            cost_info,
            gpt_service_info,
        ) = await gather_with_concurrency(tasks, n=len(tasks))

        env = "prod" if get_environment_tag() in [STAGING_TAG, PROD_TAG] else "dev"
        cloudwatch_url_template = "https://us-west-2.console.aws.amazon.com/cloudwatch/home?region=us-west-2#logsV2:log-groups/log-group/eks-{env}-jobs/log-events/agent$252F{pod_name}"  # noqa

        tool_tips = Tooltips(
            create_execution_plans="Contains one entry for every 'create_execution_plan' SQS "
            "message processed, grouped by plan_id. Each entry will include "
            "all of the information from the sqs message and additionally "
            "will include every generated plan and each plan selection",
            run_execution_plans="Contains one entry for each 'run_execution_plan' SQS message "
            "processed for this agent, grouped by plan_id and plan_run_id. "
            "Includes all tool_calls.",
        )

        run_execution_plans: Dict[Any, Any] = defaultdict(dict)
        for _, value in worker_sqs_log["run_execution_plan"].items():
            top_level_key = f"plan_id={value['plan_id']}"
            plan_run_id = value["plan_run_id"]
            inner_key = f"plan_run_id={plan_run_id}"
            if top_level_key not in run_execution_plans:
                run_execution_plans[top_level_key] = {}
            if inner_key not in run_execution_plans[top_level_key]:
                run_execution_plans[top_level_key][inner_key] = {}
            run_execution_plans[top_level_key][inner_key]["sqs_info"] = value
            del value["plan_id"]
            del value["plan_run_id"]

        for tool_call in ch_tool_calls.keys():
            if tool_call not in tool_calls:
                tool_calls[tool_call] = ch_tool_calls[tool_call]

        plan_run_id_dict = {}
        fetch_plan_run_ids = set()
        for tool_plan_run_id, tool_value in tool_calls.items():
            for tool_name, tool_call_dict in tool_value.items():
                if tool_call_dict["plan_id"]:
                    plan_run_id_dict[str(tool_call_dict["plan_run_id"])] = tool_call_dict["plan_id"]
                else:
                    fetch_plan_run_ids.add(str(tool_call_dict["plan_run_id"]))
        if fetch_plan_run_ids:
            fetched_plan_run_id_dict = await self.pg.get_plan_ids(list(fetch_plan_run_ids))
            plan_run_id_dict.update(fetched_plan_run_id_dict)

        for tool_plan_run_id, tool_value in tool_calls.items():
            plan_id = ""
            pod_name = None
            for tool_name, tool_call_dict in tool_value.items():
                plan_id = plan_run_id_dict[str(tool_call_dict["plan_run_id"])]
                pod_name = tool_call_dict.get("pod_name")
                del tool_call_dict["plan_id"]
                del tool_call_dict["plan_run_id"]
            top_level_key = f"plan_id={plan_id}"
            plan_run_id = tool_plan_run_id
            inner_key = f"plan_run_id={plan_run_id}"
            if inner_key not in run_execution_plans[top_level_key]:
                run_execution_plans[top_level_key][inner_key] = {}
            if tool_plan_run_id == plan_run_id:
                run_execution_plans[top_level_key][inner_key]["tool_calls"] = tool_value
                run_execution_plans[top_level_key][inner_key]["cloudwatch_url"] = (
                    cloudwatch_url_template.format(env=env, pod_name=pod_name)
                )
        create_execution_plans: Dict[Any, Any] = defaultdict(dict)
        for _, value in worker_sqs_log["create_execution_plan"].items():
            plan_id = value["plan_id"]
            top_level_key = f"plan_id={plan_id}"
            create_execution_plans[top_level_key] = {}
            create_execution_plans[top_level_key]["sqs_info"] = value
            create_execution_plans[top_level_key]["all_generated_plans"] = []
            create_execution_plans[top_level_key]["plan_selections"] = []
            del value["plan_id"]
            del value["plan_run_id"]
        for plan in all_generated_plans:
            plan_id = plan["plan_id"]
            top_level_key = f"plan_id={plan_id}"
            if top_level_key not in create_execution_plans:
                create_execution_plans[top_level_key] = {}
            if "all_generated_plans" not in create_execution_plans[top_level_key]:
                create_execution_plans[top_level_key]["all_generated_plans"] = []
            create_execution_plans[top_level_key]["all_generated_plans"].append(plan)
            del plan["plan_id"]
        for plan_selection in plan_selections:
            plan_id = plan_selection["plan_id"]
            top_level_key = f"plan_id={plan_id}"
            if top_level_key not in create_execution_plans:
                create_execution_plans[top_level_key] = {}
            if "plan_selections" not in create_execution_plans[top_level_key]:
                create_execution_plans[top_level_key]["plan_selections"] = []
            if plan_selection["plan_id"] == plan_id:
                create_execution_plans[top_level_key]["plan_selections"].append(plan_selection)
            del plan_selection["plan_id"]
        debug = Debug(
            run_execution_plans=run_execution_plans,
            create_execution_plans=create_execution_plans,
            agent_owner_id=agent_owner_id,
            cost_info=cost_info,
            gpt_service_info=gpt_service_info,
        )
        return GetAgentDebugInfoResponse(tooltips=tool_tips, debug=debug)

    async def get_debug_tool_args(self, replay_id: str) -> GetDebugToolArgsResponse:
        args = await self.ch.get_debug_tool_args(replay_id=replay_id)
        return GetDebugToolArgsResponse(args=json.loads(args))

    async def get_debug_tool_result(self, replay_id: str) -> GetDebugToolResultResponse:
        result = await self.ch.get_debug_tool_result(replay_id=replay_id)
        return GetDebugToolResultResponse(result=json.loads(result))

    async def get_tool_library(self, user: Optional[User] = None) -> GetToolLibraryResponse:
        tools = []
        for tool_name, tool in ToolRegistry._REGISTRY_ALL_TOOLS_MAP.items():
            if not tool.enabled:
                continue
            elif tool.enabled_checker_func and not (
                user and tool.enabled_checker_func(user.user_id)
            ):
                continue

            category = ToolRegistry._TOOL_NAME_TO_CATEGORY[tool_name]

            tools.append(
                ToolMetadata(
                    tool_name=tool_name,
                    tool_description=tool.description,
                    tool_header=tool.to_function_header(),
                    category=category.value.title(),
                )
            )

        tool_category_map = {
            category.value.title(): category.get_description() for category in ToolCategory
        }

        return GetToolLibraryResponse(tools=tools, tool_category_map=tool_category_map)

    async def get_info_for_test_suite_run(
        self, service_version: str
    ) -> GetTestSuiteRunInfoResponse:
        infos = await self.ch.get_info_for_test_suite_run(service_version=service_version)
        for test_name, test_info in infos.items():
            if "output" in test_info and test_info["output"]:
                try:
                    my_output = test_info["output"]
                    output_str = load_io_type(json.dumps(my_output[0]))
                    output_from_io_type = await get_output_from_io_type(
                        val=output_str, pg=self.pg.pg
                    )
                    test_info["formatted_output"] = output_from_io_type.model_dump()
                except Exception:
                    LOGGER.info(
                        f"Error while displaying formatted output for test case {test_name}, "
                        f"version={test_info['service_version']}: {traceback.format_exc()}"
                    )
        return GetTestSuiteRunInfoResponse(test_suite_run_info=infos)

    async def get_test_suite_runs(self) -> GetTestSuiteRunsResponse:
        return GetTestSuiteRunsResponse(test_suite_runs=await self.ch.get_test_suite_runs())

    async def get_test_cases(self) -> GetTestCasesResponse:
        return GetTestCasesResponse(test_cases=await self.ch.get_test_cases())

    async def get_info_for_test_case(self, test_name: str) -> GetTestCaseInfoResponse:
        infos = await self.ch.get_info_for_test_case(test_name=test_name)
        for version, info in infos.items():
            if "output" in info and info["output"]:
                try:
                    my_output = info["output"]
                    output_str = load_io_type(json.dumps(my_output[0]))
                    output_from_io_type = await get_output_from_io_type(
                        val=output_str, pg=self.pg.pg
                    )
                    info["formatted_output"] = output_from_io_type.model_dump()
                except Exception:
                    LOGGER.info(
                        f"Error while displaying formatted output for test case {test_name}, "
                        f"version={version}: {traceback.format_exc()}"
                    )
        return GetTestCaseInfoResponse(test_case_info=infos)

    async def upload_file(
        self, upload: UploadFile, user: User, agent_id: Optional[str] = None
    ) -> UploadFileResponse:
        upload_handler = UploadHandler(
            user_id=user.user_id,
            upload=upload,
            db=self.pg,
            agent_id=agent_id,
            send_chat_updates=True,
        )
        await upload_handler.handle_upload()
        return UploadFileResponse()

    async def update_user(
        self, user_id: str, name: str, username: str, email: str
    ) -> UpdateUserResponse:
        if username == "" or email == "" or name == "":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="name and email cannot be empty"
            )
        success = await update_user(user_id, name, username, email)
        return UpdateUserResponse(success=success)

    async def update_user_settings(
        self, user: User, req: AgentUserSettingsSetRequest
    ) -> UpdateUserResponse:
        await self.pg.update_user_agent_settings(
            entity_id=user.user_id, entity_type=AgentUserSettingsSource.USER, settings=req
        )
        return UpdateUserResponse(success=True)

    async def get_users_info(self, user: User, user_ids: List[str]) -> List[Account]:
        if len(user_ids) == 1 and user_ids[0] == user.user_id:
            cached_user = await get_user_cached(user.user_id, async_db=self.pg)
            if not cached_user:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"No Account Info found for {user.user_id}",
                )
            users = [cached_user]
        else:
            users = await get_users(
                user.user_id, user_ids, include_user_enabled=True, async_db=self.pg
            )
            if not users:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Users not found")

        return [
            Account(
                user_id=account.user_id.id,
                email=account.email,
                username=account.username,
                name=account.name,
                organization_id=account.organization_membership.organization_id.id,
            )
            for account in users
        ]

    async def get_account_info(self, user: User) -> Account:
        cached_user = await get_user_cached(user.user_id, async_db=self.pg)
        if not cached_user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No Account Info found for {user.user_id}",
            )
        return Account(
            user_id=cached_user.user_id.id,
            email=cached_user.email,
            username=cached_user.username,
            name=cached_user.name,
            organization_id=cached_user.organization_membership.organization_id.id,
        )

    @async_perf_logger
    @async_wrap
    def convert_markdown(self, content: str, new_type: MediaType) -> Tuple[bytes, str]:
        # preprocess
        # remove in-line citations
        content = re.sub(r" *```\{ [\s\S]*? \}``` ?", "", content)

        doc = pandoc.read(content, format="markdown")
        if new_type == MediaType.DOCX:
            output = pandoc.write(doc, format="docx")
            mimetype = MEDIA_TO_MIMETYPE["docx"]
        if new_type == MediaType.TXT:
            output = pandoc.write(doc, format="plain")
            mimetype = MEDIA_TO_MIMETYPE["plain"]

        return output, mimetype

    def get_canned_prompts(self) -> GetCannedPromptsResponse:
        canned_prompts = []
        for canned_prompt in CANNED_PROMPTS:
            canned_prompts.append(
                CannedPrompt(id=canned_prompt["id"], prompt=canned_prompt["prompt"])
            )
        return GetCannedPromptsResponse(canned_prompts=canned_prompts)

    async def create_sidebar_section(self, name: str, user: User) -> str:
        section = SidebarSection(name=name)
        sections: List[SidebarSection] = await self.pg.get_sidebar_sections(user_id=user.user_id)
        sections.append(section)
        await self.pg.set_sidebar_sections(user_id=user.user_id, sections=sections)
        return section.id

    async def delete_sidebar_section(self, section_id: str, user: User) -> bool:
        await self.pg.update_agent_sections(
            new_section_id=None, section_id=section_id, user_id=user.user_id
        )
        sections: List[SidebarSection] = await self.pg.get_sidebar_sections(user_id=user.user_id)
        index = find_sidebar_section(sections=sections, section_id=section_id)
        sections.pop(index)
        await self.pg.set_sidebar_sections(user_id=user.user_id, sections=sections)
        return True

    async def rename_sidebar_section(self, section_id: str, new_name: str, user: User) -> bool:
        sections: List[SidebarSection] = await self.pg.get_sidebar_sections(user_id=user.user_id)
        index = find_sidebar_section(sections=sections, section_id=section_id)
        sections[index].name = new_name
        sections[index].updated_at = get_now_utc().isoformat()
        await self.pg.set_sidebar_sections(user_id=user.user_id, sections=sections)
        return True

    async def set_agent_sidebar_section(
        self, new_section_id: Optional[str], agent_id: str, user: User
    ) -> bool:
        await self.pg.set_agent_section(
            new_section_id=new_section_id, agent_id=agent_id, user_id=user.user_id
        )
        return True

    async def rearrange_sidebar_section(self, section_id: str, new_index: int, user: User) -> bool:
        sections: List[SidebarSection] = await self.pg.get_sidebar_sections(user_id=user.user_id)
        index = find_sidebar_section(sections=sections, section_id=section_id)
        rearrange_section = sections.pop(index)
        rearrange_section.updated_at = get_now_utc().isoformat()
        sections.insert(new_index, rearrange_section)
        await self.pg.set_sidebar_sections(user_id=user.user_id, sections=sections)
        return True

    async def set_agent_feedback(
        self, feedback_data: SetAgentFeedBackRequest, user_id: str
    ) -> SetAgentFeedBackResponse:
        await self.pg.set_agent_feedback(feedback_data=feedback_data, user_id=user_id)
        return SetAgentFeedBackResponse(success=True)

    async def get_agent_feedback(
        self, agent_id: str, plan_id: str, plan_run_id: str, output_id: str, user_id: str
    ) -> GetAgentFeedBackResponse:
        feedback = await self.pg.get_agent_feedback(
            agent_id=agent_id,
            plan_id=plan_id,
            plan_run_id=plan_run_id,
            output_id=output_id,
            feedback_user_id=user_id,
        )

        if len(feedback) != 1:
            return GetAgentFeedBackResponse(agent_feedback=None, success=True)

        return GetAgentFeedBackResponse(agent_feedback=feedback[0], success=True)

    async def copy_agent(
        self, src_agent_id: str, dst_user_ids: List[str], dst_agent_name: Optional[str] = None
    ) -> Dict[str, str]:
        res = {}
        for user_id in dst_user_ids:
            new_agent_id = str(uuid.uuid4())
            await self.pg.copy_agent(
                src_agent_id=src_agent_id,
                dst_agent_id=new_agent_id,
                dst_user_id=user_id,
                dst_agent_name=dst_agent_name,
            )
            res[user_id] = new_agent_id
        return res

    # Variables / Data
    async def get_all_available_variables(self, user: User) -> GetAvailableVariablesResponse:
        try:
            resp = await get_all_variables_metadata(user_id=user.user_id)
            return GetAvailableVariablesResponse(
                variables=[
                    AvailableVariable(
                        id=var.feature_id,
                        name=var.name,
                        description=var.description,
                        alternate_names=list(var.alternate_names),
                        is_global=var.is_global,
                        is_preset=var.is_composite_preset,
                        hierarchical_category_id=var.hierarchical_category_id,
                    )
                    for var in resp.features
                ]
            )
        except GRPCError as e:
            raise VariableExplorerException.from_grpc_error(e) from e

    async def get_variable_hierarchy(self, user: User) -> GetVariableHierarchyResponse:
        try:
            resp = await get_all_variable_hierarchy(user_id=user.user_id)
            return GetVariableHierarchyResponse(
                flat_nodes=[
                    VariableHierarchyNode(
                        category_id=cat.category_id,
                        category_name=cat.category_name,
                        parent_category_id=(
                            cat.parent_category_id if cat.parent_category_id else None
                        ),
                    )
                    for cat in resp.categories
                ]
            )
        except GRPCError as e:
            raise VariableExplorerException.from_grpc_error(e) from e

    async def experimental_get_formula_data_impl(
        self,
        user: User,
        markdown_formula: str,
        stock_ids: List[int],
        from_date: datetime.date,
        to_date: datetime.date,
    ) -> ExperimentalGetFormulaDataResponse:
        try:
            formula_parts = extract_ordered_text_object_parts_from_text(markdown_formula)
            formula = ""
            for part in formula_parts:
                if isinstance(part, str):
                    # for non-markdown parts of the formula, add as-is
                    formula += part
                elif isinstance(part, VariableTextObject):
                    # variables are specified in formulas as `variable_id`
                    formula += f"`{part.id}`"
                else:
                    raise VariableExplorerException(
                        message=f"{type(part)=} is not supported in formula evaluation.",
                    )
            LOGGER.info(f"Processing formula as: {formula}")
            resp = await experimental_get_formula_data(
                user_id=user.user_id,
                formula=formula,
                stock_ids=stock_ids,
                from_date=from_date,
                to_date=to_date,
            )
            np_cube = NumpyCube.initialize_from_proto_bytes(resp.security_data, cols_are_dates=True)
            data = np_cube.np_data[np_cube.row_map[formula], :, :]
            df = pd.DataFrame(
                data,
                index=np_cube.columns,
                columns=[int(s) for s in np_cube.fields],
            )
            df = df.dropna(axis="index", how="all")
            df.index.rename("Date", inplace=True)
            df.reset_index(inplace=True)
            df = df.melt(
                id_vars=["Date"],
                var_name=STOCK_ID_COL_NAME_DEFAULT,
                value_name=formula,
            )
            df[STOCK_ID_COL_NAME_DEFAULT] = await StockID.from_gbi_id_list(
                list(df[STOCK_ID_COL_NAME_DEFAULT])
            )

            # We now have a dataframe with only a few columns: Date, Stock ID, Statistic Name
            df = df[df[formula].notna()]
            stock_table = StockTable.from_df_and_cols(
                data=df,
                columns=[
                    TableColumnMetadata(label="Date", col_type=TableColumnType.DATE),
                    TableColumnMetadata(
                        label=STOCK_ID_COL_NAME_DEFAULT, col_type=TableColumnType.STOCK
                    ),
                    TableColumnMetadata(
                        label=formula,
                        col_type=TableColumnType.FLOAT,
                        unit=None,
                        data_src=[formula],
                    ),
                ],
            )
            stock_graph = await make_line_graph(
                MakeLineGraphArgs(input_table=stock_table),
                PlanRunContext.get_dummy(user_id=user.user_id),
            )
            pg = self.pg.pg
            output_table = await stock_table.to_rich_output(pg=pg)
            output_graph = await stock_graph.to_rich_output(pg=pg)  # type: ignore
            return ExperimentalGetFormulaDataResponse(
                output=output_table,  # type: ignore
                output_graph=output_graph,  # type: ignore
            )
        except GRPCError as e:
            raise VariableExplorerException.from_grpc_error(e) from e
        except VariableExplorerBadInputError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=e.message,
            )

    async def get_variable_coverage(
        self,
        user: User,
        feature_ids: Optional[List[str]] = None,
        universe_id: Optional[str] = None,
    ) -> GetVariableCoverageResponse:
        """Get coverage data for specified variables"""
        # Set up universe ID based on environment if not provided
        SPY_UNIVERSE_ID_PROD = "4e5f2fd3-394e-4db9-aad3-5c20abf9bf3c"
        SPY_UNIVERSE_ID_DEV = "249a293b-d9e2-4905-94c1-c53034f877c9"
        default_universe_id = (
            SPY_UNIVERSE_ID_PROD
            if get_environment_tag() in [PROD_TAG, STAGING_TAG]
            else SPY_UNIVERSE_ID_DEV
        )

        # Initialize feature coverage client
        feat_coverage_client = get_feature_coverage_client()

        if not feature_ids:
            available_vars = await self.get_all_available_variables(user=user)
            feature_ids = [var.id for var in available_vars.variables]

        # Use provided universe_id or default
        universe_id = universe_id or default_universe_id

        coverage_response = await feat_coverage_client.get_latest_universe_coverage(
            universe_id=universe_id,
            feature_ids=feature_ids,
        )

        coverage_dict = {item.feature_id: item.coverage for item in coverage_response.features}

        return GetVariableCoverageResponse(coverages=coverage_dict)

    # Custom Documents
    async def list_custom_documents(self, user: User) -> ListCustomDocumentsResponse:
        try:
            resp: ListDocumentsResponse = await list_custom_docs(user_id=user.user_id)
            return ListCustomDocumentsResponse(
                documents=[custom_doc_listing_proto_to_model(listing) for listing in resp.listings],
            )
        except GRPCError as e:
            raise CustomDocumentException.from_grpc_error(e) from e

    async def get_custom_doc_file_content(
        self, user: User, file_id: str, return_previewable_file: bool
    ) -> GetCustomDocumentFileResponse:
        try:
            resp: GetFileContentsResponse = await get_custom_doc_file_contents(
                user_id=user.user_id,
                file_id=file_id,
                return_previewable_file=return_previewable_file,
            )

            return GetCustomDocumentFileResponse(
                is_preview=False,
                file_name=resp.file_name,
                file_type=resp.content_type,
                content=resp.raw_file,
            )
        except GRPCError as e:
            raise CustomDocumentException.from_grpc_error(e) from e

    async def get_custom_doc_file_info(
        self, user: User, file_id: str
    ) -> GetCustomDocumentFileInfoResponse:
        try:
            resp: GetFileInfoResponse = await get_custom_doc_file_info(
                user_id=user.user_id, file_id=file_id
            )

            # rpc is for a list of docs, try to grab the one we requested
            file_info = resp.file_info.get(file_id)
            if file_info is None:
                raise CustomDocumentException(
                    f"document not found for file_id: {file_id}", ["No file info found"]
                )

            return GetCustomDocumentFileInfoResponse(
                file_id=file_info.file_id,
                author=file_info.author,
                status=document_listing_status_to_str(file_info.status),
                file_type=file_info.file_type,
                file_size=file_info.size,
                author_org=file_info.author_org,
                upload_time=file_info.upload_time.ToDatetime(),
                publication_time=file_info.publication_time.ToDatetime(),
                company_name=file_info.company_name,
                spiq_company_id=file_info.spiq_company_id,
                file_paths=[f for f in file_info.file_paths],
                chunks=[
                    CustomDocumentSummaryChunk(
                        chunk_id=ch.chunk_id,
                        headline=ch.headline,
                        summary=ch.summary,
                        long_summary=ch.long_summary,
                        # note: not including citations from this endpoint for now until we have a use case
                    )
                    for ch in file_info.chunks
                ],
            )
        except GRPCError as e:
            raise CustomDocumentException.from_grpc_error(e) from e

    async def add_custom_documents(
        self,
        user: User,
        files: List[UploadFile],
        base_path: Optional[str] = "",
        allow_overwrite: Optional[bool] = True,
    ) -> AddCustomDocumentsResponse:

        # get total size of files
        total_size = sum([file.size or 0 for file in files])

        # check if total size exceeds limit, get bucket paths
        quota_resp = await check_custom_doc_upload_quota(
            user_id=user.user_id, candidate_total_size=total_size
        )
        s3_bucket = quota_resp.authorized_s3_bucket
        s3_prefix = quota_resp.authorized_s3_prefix
        if quota_resp.status.code == GrpcStatus.RESOURCE_EXHAUSTED.value:
            raise CustomDocumentQuotaExceededException(quota_resp.status.message)

        if len(s3_bucket) == 0 or len(s3_prefix) == 0:
            # if we didn't get s3 paths, we can't continue
            raise CustomDocumentException(
                message=(
                    quota_resp.status.message
                    or "Error getting authorized paths for document upload"
                ),
                errors=[str(detail) for detail in quota_resp.status.details],
            )

        # upload files to s3
        to_upload: List[S3FileUploadTuple] = []
        s3_uploaded_file_keys = []
        for file in files:
            file_path = f"{base_path}{file.filename}" if base_path else file.filename
            s3_key = f"{s3_prefix}{file_path}"
            to_upload.append(
                S3FileUploadTuple(file_object=file.file, bucket_name=s3_bucket, key_path=s3_key)
            )
            s3_uploaded_file_keys.append(s3_key)

        await async_upload_files_to_s3(files=to_upload)

        # notify backend of new files
        process_resp = await process_uploaded_s3_documents(
            user_id=user.user_id, s3_bucket=s3_bucket, s3_keys=s3_uploaded_file_keys
        )

        response = AddCustomDocumentsResponse(
            success=process_resp.status.code == 0,
            added_listings=[
                custom_doc_listing_proto_to_model(doc) for doc in process_resp.updated_listings
            ],
        )
        return response

    async def delete_custom_documents(
        self, user: User, file_paths: List[str]
    ) -> DeleteCustomDocumentsResponse:
        try:
            delete_resp = await delete_custom_docs(user_id=user.user_id, file_paths=file_paths)
            return DeleteCustomDocumentsResponse(success=delete_resp.status.code == 0)
        except GRPCError as e:
            raise CustomDocumentException.from_grpc_error(e) from e

    async def check_document_upload_quota(
        self, user: User, candidate_total_size: Optional[int] = 0
    ) -> CheckCustomDocumentUploadQuotaResponse:
        try:
            resp = await check_custom_doc_upload_quota(
                user_id=user.user_id, candidate_total_size=candidate_total_size
            )
            response = CheckCustomDocumentUploadQuotaResponse(
                authorized_s3_bucket=(
                    None if resp.authorized_s3_bucket == "" else resp.authorized_s3_bucket
                ),
                authorized_s3_prefix=(
                    None if resp.authorized_s3_prefix == "" else resp.authorized_s3_prefix
                ),
                total_quota_size=resp.total_quota_size,
                remaining_quota_size=resp.remaining_quota_size,
                message=resp.status.message if len(resp.status.message) > 0 else None,
            )
            return response
        except GRPCError as e:
            raise CustomDocumentException.from_grpc_error(e) from e

    async def get_all_companies(self, user: User) -> GetCompaniesResponse:
        companies = await self.pg.get_all_companies()
        return GetCompaniesResponse(companies=companies)

    async def get_prompt_templates(self, user: User, is_user_admin: bool) -> List[PromptTemplate]:
        prompt_templates_all = await self.pg.get_prompt_templates()
        user_info = await self.get_account_info(user)
        user_org_id = user_info.organization_id

        prompt_templates = []
        for template in prompt_templates_all:
            if template.plan is not None:
                # show templates that have no organization_id or match organization_id
                if not template.organization_ids or user_org_id in template.organization_ids:
                    template.preview = get_plan_preview(template.plan)
                    prompt_templates.append(template)
                    continue
                # show those that have user_id matching the user
                if template.user_id and user.user_id == template.user_id:
                    template.preview = get_plan_preview(template.plan)
                    prompt_templates.append(template)
                    continue

            else:
                LOGGER.error("A prompt template has no plan")

        return prompt_templates

    async def update_prompt_template(
        self,
        template_id: str,
        name: str,
        description: str,
        prompt: str,
        category: str,
        plan: ExecutionPlan,
        cadence_tag: str,
        notification_criteria: Optional[List[str]] = None,
        organization_ids: Optional[List[str]] = None,
    ) -> UpdatePromptTemplateResponse:
        prompt_template = PromptTemplate(
            template_id=template_id,
            name=name,
            description=description,
            prompt=prompt,
            category=category,
            plan=plan,
            cadence_tag=cadence_tag,
            notification_criteria=notification_criteria,
            organization_ids=organization_ids,
            created_at=get_now_utc(),
        )
        await self.pg.update_prompt_template(prompt_template)
        return UpdatePromptTemplateResponse(prompt_template=prompt_template)

    async def create_prompt_template(
        self,
        name: str,
        description: str,
        prompt: str,
        category: str,
        plan_run_id: str,
        user: User,
        cadence_tag: str,
        notification_criteria: Optional[List[str]] = None,
        organization_ids: Optional[List[str]] = None,
    ) -> CreatePromptTemplateResponse:

        # extract plan from agent_id and plan_id
        _, plan = await self.pg.get_execution_plan_for_run(plan_run_id)

        # create prompt template object
        now = get_now_utc()
        prompt_template = PromptTemplate(
            template_id=str(uuid.uuid4()),
            user_id=user.user_id,
            name=name,
            description=description,
            prompt=prompt,
            created_at=now,
            category=category,
            plan=plan,
            cadence_tag=cadence_tag,
            notification_criteria=notification_criteria,
            organization_ids=organization_ids,
        )
        # add template to db
        await self.pg.insert_prompt_template(prompt_template)
        return CreatePromptTemplateResponse(
            template_id=prompt_template.template_id,
            user_id=user.user_id,
            name=prompt_template.name,
            description=prompt_template.description,
            prompt=prompt_template.prompt,
            created_at=prompt_template.created_at,
            category=prompt_template.category,
            plan_run_id=plan_run_id,
            organization_ids=organization_ids,
            cadence_tag=cadence_tag,
            notification_criteria=notification_criteria,
        )

    async def gen_prompt_template_from_plan(
        self, plan_run_id: str, agent_id: str
    ) -> GenPromptTemplateFromPlanResponse:
        # extract plan from agent_id and plan_id
        _, plan = await self.pg.get_execution_plan_for_run(plan_run_id)
        prompt_str = await generate_template_from_plan(
            plan=plan,
            agent_id=agent_id,
            gpt_service_stub=self.gpt_service_stub,
            db=self.pg,
        )

        return GenPromptTemplateFromPlanResponse(prompt_str=prompt_str)

    async def gen_template_plan(self, template_prompt: str, user: User) -> GenTemplatePlanResponse:

        LOGGER.info("Generating template plan....")
        chat_context = ChatContext(
            messages=[Message(message=template_prompt, is_user_message=True)]
        )
        planner = Planner(user_id=user.user_id, skip_db_commit=True, send_chat=False)
        plan = await planner.create_initial_plan(chat_context=chat_context, use_sample_plans=True)
        if plan is None:
            raise HTTPException(status_code=400, detail="Plan generation failed: no plan created.")

        return GenTemplatePlanResponse(plan=plan, preview=get_plan_preview(plan))

    async def create_agent_and_run_template(
        self,
        template_prompt: str,
        notification_criteria: Optional[List[str]],
        plan: ExecutionPlan,
        is_draft: bool,
        cadence_description: Optional[str],
        user: User,
    ) -> RunTemplatePlanResponse:

        # create a new agent
        LOGGER.info("Creating agent for the template plan")
        agent = await self.create_agent(user=user, is_draft=is_draft, created_from_template=True)
        agent_id = agent.agent_id if agent.agent_id else ""

        # insert user's new message to DB for the agent
        LOGGER.info(f"Inserting user's new message to DB for {agent_id=}")
        user_msg = Message(
            agent_id=agent_id,
            message=template_prompt,
            is_user_message=True,
            message_author=user.real_user_id,
        )
        await self.pg.insert_chat_messages(messages=[user_msg])

        # check if portfolio or watchlist names need to be updated in default plan

        watchlisit_name_in_template, portfolio_name_in_template = None, None

        _, text_objects = extract_text_objects_from_text(template_prompt)
        for obj in text_objects:
            if isinstance(obj, WatchlistTextObject):
                watchlisit_name_in_template = obj.label
                watchlisit_id_in_template = obj.id
            if isinstance(obj, PortfolioTextObject):
                portfolio_name_in_template = obj.label
                portfolio_id_in_template = obj.id

        for node in plan.nodes:
            if node.tool_name == "get_user_watchlist_stocks":
                watchlisit_name_in_plan = node.args.get("watchlist_name")
                if watchlisit_name_in_template != watchlisit_name_in_plan:
                    node.args["watchlist_name"] = watchlisit_name_in_template
                    node.args["watchlist_id"] = watchlisit_id_in_template
            if node.tool_name == "convert_portfolio_mention_to_portfolio_id":
                portfolio_name_in_plan = node.args.get("portfolio_name")
                if portfolio_name_in_template != portfolio_name_in_plan:
                    node.args["portfolio_name"] = portfolio_name_in_template
                    node.args["portfolio_uuid"] = portfolio_id_in_template

        # Write complete plan to db and let FE know the plan is ready
        # FE cancellation button will show up after this point
        LOGGER.info("Saving the plan in the DB")
        plan_id = str(uuid.uuid4())
        plan_run_id = str(uuid.uuid4())
        chat_context = ChatContext(
            messages=[
                Message(message=template_prompt, is_user_message=True, plan_run_id=plan_run_id)
            ]
        )
        ctx = PlanRunContext(
            agent_id=agent_id,
            plan_id=plan_id,
            user_id=user.user_id,
            plan_run_id=plan_run_id,
            chat=chat_context,
        )
        await publish_agent_execution_plan(plan, ctx, self.pg)

        # create agent name and store in db
        LOGGER.info("Creating future task to generate agent name and store in DB")
        user_msg = Message(
            agent_id=agent_id,
            message=template_prompt,
            is_user_message=True,
            plan_run_id=plan_run_id,
        )
        run_async_background(self._generate_agent_name_and_store(user.user_id, agent_id, user_msg))

        # run the plan
        LOGGER.info(f"Running the plan for {agent_id=}")
        context = PlanRunContext(
            agent_id=agent_id,
            plan_id=plan_id,
            user_id=user.user_id,
            plan_run_id=plan_run_id,
        )
        await kick_off_run_execution_plan(
            plan=plan,
            context=context,
            do_chat=True,
        )

        # set up cadence and notification criteria if they exist
        if cadence_description:
            agent_schedule = await self.set_agent_schedule(
                SetAgentScheduleRequest(
                    agent_id=agent_id,
                    user_schedule_description=cadence_description,
                )
            )
            LOGGER.info(f"Set up cadence for agent_id={agent_schedule.agent_id} ")
        if notification_criteria:
            for notification_prompt in notification_criteria:
                notif_id = await self.create_agent_notification_criteria(
                    CreateCustomNotificationRequest(
                        agent_id=agent_id,
                        notification_prompt=notification_prompt,
                    )
                )
                LOGGER.info(f"Set up notification criteria with notif_id={notif_id} ")

        return RunTemplatePlanResponse(
            agent_id=agent_id,
        )

    async def delete_prompt_template(self, template_id: str) -> DeletePromptTemplateResponse:
        await self.pg.delete_prompt_template(template_id=template_id)
        return DeletePromptTemplateResponse(template_id=template_id)

    async def find_templates_related_to_prompt(
        self, query: str, user: User, is_user_admin: bool
    ) -> FindTemplatesRelatedToPromptResponse:
        prompt_templates = await self.get_prompt_templates(user=user, is_user_admin=is_user_admin)
        matched_templates = await get_matched_templates(
            query=query, prompt_templates=prompt_templates
        )
        return FindTemplatesRelatedToPromptResponse(prompt_templates=matched_templates)

    async def get_user_has_alfa_access(self, user: User) -> bool:
        with sentry_sdk.start_span(op="get_user_cached", description="get_user_cached"):
            cached_user = await get_user_cached(user_id=user.user_id, async_db=self.pg)
            if not cached_user:
                return False

        # 1. If people are permitted to access by User SVC, we return True
        # 2. Or if feature flag returns False (meaning everyone has access to Alfa), we return True
        return (cached_user.cognito_enabled.value and cached_user.has_alfa_access) or (
            not await is_database_access_check_enabled_for_user(
                user_id=user.user_id, async_db=self.pg
            )
        )

    async def get_ordered_securities(
        self, req: GetOrderedSecuritiesRequest, user: User
    ) -> GetOrderedSecuritiesResponse:
        securities = await get_ordered_securities(user_id=user.user_id, **(req.model_dump()))
        return GetOrderedSecuritiesResponse(
            securities=[MasterSecurity(**security) for security in securities]
        )

    async def get_news_summary(
        self, req: GetNewsSummaryRequest, user: User
    ) -> GetNewsSummaryResponse:
        news_summary = await get_stock_news_summary(
            user_id=user.user_id,
            gbi_id=req.gbi_id,
            delta_horizon=req.delta_horizon,
            show_hypotheses=req.show_hypotheses,
        )
        if news_summary is None:
            return GetNewsSummaryResponse(sentiment=0.0, summary=None, sourceCounts=[])
        return GetNewsSummaryResponse(**news_summary)

    async def get_company_description(
        self, user: User, gbi_id: int
    ) -> GetCompanyDescriptionResponse:
        description = await get_company_description(user_id=user.user_id, gbi_id=gbi_id)
        return GetCompanyDescriptionResponse(description=description)

    async def get_security_pros_cons(self, user: User, gbi_id: int) -> GetSecurityProsConsResponse:
        pros_cons = await get_security_pros_cons(user_id=user.user_id, gbi_id=gbi_id)
        if pros_cons is None:
            return GetSecurityProsConsResponse(pros=None, cons=None)
        return GetSecurityProsConsResponse(
            pros=(
                [GetSecurityProsConsResponse.ProCon(**pro) for pro in pros_cons.get("pros", [])]
                if pros_cons
                else None
            ),
            cons=(
                [GetSecurityProsConsResponse.ProCon(**con) for con in pros_cons.get("cons", [])]
                if pros_cons
                else None
            ),
        )

    async def get_earnings_summary(self, user: User, gbi_id: int) -> GetEarningsSummaryResponse:
        earnings_reports = await get_earnings_summary(user_id=user.user_id, gbi_id=gbi_id)
        if earnings_reports is None:
            return GetEarningsSummaryResponse(reports=None)
        return GetEarningsSummaryResponse(
            reports=[
                GetEarningsSummaryResponse.EarningsReport(**report) for report in earnings_reports
            ]
        )

    async def get_security(self, user: User, gbi_id: int) -> GetSecurityResponse:
        security_dict = await get_security(user_id=user.user_id, gbi_id=gbi_id)
        if security_dict is None:
            return GetSecurityResponse(security=None)
        return GetSecurityResponse(security=GetSecurityResponse.SimpleSecurity(**security_dict))

    async def get_historical_prices(
        self, req: GetHistoricalPricesRequest, user: User
    ) -> GetHistoricalPricesResponse:
        prices = await get_stock_historical_prices(
            user_id=user.user_id,
            gbi_id=req.gbi_id,
            start_date=req.start_date,
            end_date=req.end_date,
        )
        return GetHistoricalPricesResponse(
            prices=(
                [GetHistoricalPricesResponse.HistoricalPrice(**price) for price in prices]
                if prices
                else None
            )
        )

    async def get_real_time_price(self, user: User, gbi_id: int) -> GetRealTimePriceResponse:
        price = await get_stock_real_time_price(user_id=user.user_id, gbi_id=gbi_id)
        return GetRealTimePriceResponse(price=price)

    async def get_price_data(self, user: User, gbi_id: int) -> GetPriceDataResponse:
        price_data = await get_stock_price_data(user_id=user.user_id, gbi_id=gbi_id)
        if price_data is None:
            return GetPriceDataResponse(price_data=None)
        return GetPriceDataResponse(price_data=GetPriceDataResponse.PriceData(**price_data))

    async def search_agent_qcs(
        self,
        filter_criteria: List[HorizonCriteria],
        search_criteria: List[HorizonCriteria],
        pagination: Optional[Pagination] = None,
    ) -> Tuple[List[AgentQC], int]:

        # Call the database search function
        agent_qcs, total_agent_qcs = await self.pg.search_agent_qc(
            filter_criteria, search_criteria, pagination
        )

        # Return the list of AgentQC objects
        return agent_qcs, total_agent_qcs

    async def get_market_data(self, user: User, gbi_id: int) -> GetMarketDataResponse:
        market_data = await get_stock_market_data(user_id=user.user_id, gbi_id=gbi_id)
        return GetMarketDataResponse(
            market_data=(
                [GetMarketDataResponse.MarketData(**data) for data in market_data]
                if market_data
                else None
            )
        )

    async def get_dividend_yield(self, user: User, gbi_id: int) -> GetDividendYieldResponse:
        dividend_yield = await get_stock_dividend_yield(user_id=user.user_id, gbi_id=gbi_id)
        return GetDividendYieldResponse(dividend_yield=dividend_yield)

    async def get_executive_earnings_summary(
        self, user: User, gbi_id: int
    ) -> GetExecutiveEarningsSummaryResponse:
        changes = await get_executive_earnings_summary(user_id=user.user_id, gbi_id=gbi_id)
        if changes is None:
            return GetExecutiveEarningsSummaryResponse(changes=None)
        return GetExecutiveEarningsSummaryResponse(
            changes=[
                GetExecutiveEarningsSummaryResponse.SummaryChange(**change) for change in changes
            ]
        )

    async def get_previous_earnings(self, user: User, gbi_id: int) -> GetPreviousEarningsResponse:
        earnings = await get_previous_earnings(user_id=user.user_id, gbi_id=gbi_id)
        if earnings is None:
            return GetPreviousEarningsResponse(earnings=None)
        return GetPreviousEarningsResponse(
            earnings=GetPreviousEarningsResponse.PreviousEarnings(**earnings)
        )

    async def get_upcoming_earnings(self, user: User, gbi_id: int) -> GetUpcomingEarningsResponse:
        earnings = await get_upcoming_earnings(user_id=user.user_id, gbi_id=gbi_id)
        if earnings is None:
            return GetUpcomingEarningsResponse(earnings=None)
        return GetUpcomingEarningsResponse(
            earnings=GetUpcomingEarningsResponse.UpcomingEarnings(**earnings)
        )

    async def update_qc_agent(self, agent_qc: AgentQC) -> bool:
        try:
            await self.pg.update_agent_qc(agent_qc)
            return True
        except Exception as e:
            logging.warning(f"Failed to update quality agent table {agent_qc.agent_qc_id=}: {e}")
            return False

    async def get_live_agents_qc(self, req: GetAgentsQCRequest) -> GetLiveAgentsQCResponse:
        infos = await self.pg.get_agent_metadata_for_qc(
            live_only=req.live_only, start_dt=req.start_dt, end_dt=req.end_dt
        )
        return GetLiveAgentsQCResponse(agent_infos=infos)

    async def create_jira_ticket(
        self, request: CreateJiraTicketRequest
    ) -> CreateJiraTicketResponse:
        # Prepare the ticket criteria
        criteria = JiraTicketCriteria(
            project_key=request.project_key,
            summary=request.summary,
            description=request.description,
            issue_type=request.issue_type,
            assignee=request.assignee,
            priority=request.priority,  # Default priority
            labels=request.labels,  # Default labels
            parent=request.parent,
        )

        # Add any additional fields specified in the request
        if request.additional_fields:
            # Update optional common fields dynamically
            for field, value in request.additional_fields.items():
                if hasattr(criteria, field) and value is not None:
                    setattr(criteria, field, value)
                else:
                    # Treat as a custom field if it doesn't match a known field
                    if criteria.custom_fields is None:
                        criteria.custom_fields = {}
                    criteria.custom_fields[field] = value
        issue = self.jira_integration.create_ticket(criteria)
        if issue:
            return CreateJiraTicketResponse(
                ticket_id=issue[0].get("key", ""), ticket_url=issue[0].get("self", ""), success=True
            )

        return CreateJiraTicketResponse(ticket_id="", ticket_url="", success=False)
