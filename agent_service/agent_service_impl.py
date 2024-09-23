import asyncio
import datetime
import json
import logging
import re
import time
import traceback
import uuid
from collections import defaultdict
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple, cast
from uuid import uuid4

import pandoc
from fastapi import HTTPException, Request, UploadFile, status
from gpt_service_proto_v1.service_grpc import GPTServiceStub
from grpclib import GRPCError
from stock_universe_service_proto_v1.custom_data_service_pb2 import (
    GetFileContentsResponse,
    GetFileInfoResponse,
    ListDocumentsResponse,
)

from agent_service.canned_prompts.canned_prompts import CANNED_PROMPTS
from agent_service.chatbot.chatbot import Chatbot
from agent_service.endpoints.authz_helper import User
from agent_service.endpoints.models import (
    Account,
    AgentEvent,
    AgentMetadata,
    CannedPrompt,
    ChatWithAgentRequest,
    ChatWithAgentResponse,
    CreateAgentResponse,
    CreateCustomNotificationRequest,
    CustomDocumentListing,
    CustomDocumentSummaryChunk,
    CustomNotification,
    Debug,
    DeleteAgentOutputRequest,
    DeleteAgentOutputResponse,
    DeleteAgentResponse,
    DeleteMemoryResponse,
    DisableAgentAutomationResponse,
    EnableAgentAutomationResponse,
    ExecutionPlanTemplate,
    GenTemplatePlanResponse,
    GetAccountInfoResponse,
    GetAgentDebugInfoResponse,
    GetAgentFeedBackResponse,
    GetAgentOutputResponse,
    GetAgentTaskOutputResponse,
    GetAgentWorklogBoardResponse,
    GetAllAgentsResponse,
    GetAutocompleteItemsResponse,
    GetCannedPromptsResponse,
    GetChatHistoryResponse,
    GetCustomDocumentFileInfoResponse,
    GetCustomDocumentFileResponse,
    GetDebugToolArgsResponse,
    GetDebugToolResultResponse,
    GetMemoryContentResponse,
    GetPlanRunDebugInfoResponse,
    GetPlanRunOutputResponse,
    GetSecureUserResponse,
    GetTestCaseInfoResponse,
    GetTestCasesResponse,
    GetTestSuiteRunInfoResponse,
    GetTestSuiteRunsResponse,
    GetToolLibraryResponse,
    ListCustomDocumentsResponse,
    ListMemoryItemsResponse,
    LockAgentOutputRequest,
    LockAgentOutputResponse,
    MarkNotificationsAsReadResponse,
    MarkNotificationsAsUnreadResponse,
    MediaType,
    MemoryItem,
    ModifyPlanRunArgsRequest,
    ModifyPlanRunArgsResponse,
    NotificationEmailsResponse,
    NotificationEvent,
    OutputType,
    PlanRunToolDebugInfo,
    PlanTemplateTask,
    RenameMemoryResponse,
    RestoreAgentResponse,
    RunTemplatePlanResponse,
    SetAgentFeedBackRequest,
    SetAgentFeedBackResponse,
    SetAgentScheduleRequest,
    SetAgentScheduleResponse,
    SharePlanRunResponse,
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
    UpdateUserResponse,
    UploadFileResponse,
    ValidateArgError,
)
from agent_service.endpoints.utils import get_agent_hierarchical_worklogs
from agent_service.external.custom_data_svc_client import (
    document_listing_status_to_str,
    get_custom_doc_file_contents,
    get_custom_doc_file_info,
    list_custom_docs,
)
from agent_service.external.pa_svc_client import get_all_watchlists, get_all_workspaces
from agent_service.external.user_svc_client import (
    get_users,
    list_team_members,
    update_user,
)
from agent_service.io_type_utils import get_clean_type_name, load_io_type
from agent_service.io_types.citations import CitationDetailsType, CitationType
from agent_service.io_types.graph import BarGraph, LineGraph, PieGraph
from agent_service.io_types.table import Table
from agent_service.io_types.text import Text, TextOutput
from agent_service.planner.action_decide import FirstActionDecider
from agent_service.planner.constants import CHAT_DIFF_TEMPLATE, FirstAction
from agent_service.planner.planner import Planner
from agent_service.planner.planner_types import ExecutionPlan, Variable
from agent_service.slack.slack_sender import SlackSender, get_user_info_slack_string
from agent_service.tool import ToolCategory, ToolRegistry
from agent_service.types import ChatContext, MemoryType, Message, PlanRunContext
from agent_service.uploads import UploadHandler
from agent_service.utils.agent_event_utils import (
    publish_agent_execution_plan,
    publish_agent_name,
    send_chat_message,
)
from agent_service.utils.agent_name import generate_name_for_agent
from agent_service.utils.async_db import AsyncDB
from agent_service.utils.async_utils import (
    async_wrap,
    gather_with_concurrency,
    run_async_background,
)
from agent_service.utils.clickhouse import Clickhouse
from agent_service.utils.constants import MEDIA_TO_MIMETYPE
from agent_service.utils.custom_documents_utils import CustomDocumentException
from agent_service.utils.date_utils import get_now_utc
from agent_service.utils.event_logging import log_event
from agent_service.utils.feature_flags import (
    get_custom_user_dict,
    get_secure_mode_hash,
    get_user_context,
    is_database_access_check_enabled_for_user,
)
from agent_service.utils.logs import async_perf_logger
from agent_service.utils.memory_handler import MemoryHandler, get_handler
from agent_service.utils.output_utils.output_construction import get_output_from_io_type
from agent_service.utils.postgres import DEFAULT_AGENT_NAME
from agent_service.utils.prefect import prefect_run_execution_plan
from agent_service.utils.prompt_template import PromptTemplate
from agent_service.utils.redis_queue import (
    get_agent_event_channel,
    get_notification_event_channel,
    wait_for_messages,
)
from agent_service.utils.scheduling import (
    AgentSchedule,
    get_schedule_from_user_description,
)
from agent_service.utils.sidebar_sections import SidebarSection, find_sidebar_section
from agent_service.utils.string_utils import is_valid_uuid
from agent_service.utils.task_executor import TaskExecutor

LOGGER = logging.getLogger(__name__)

EDITABLE_TYPE_STRS = {
    "int",
    "str",
    "float",
    "bool",
    "Optional[int]",
    "Optional[str]",
    "Optional[float]",
    "Optional[bool]",
    "List[int]",
    "List[str]",
    "List[float]",
    "List[bool]",
}

PRIMITIVE_TYPES = {
    "int": int,
    "str": str,
    "float": float,
    "bool": bool,
}
OPTIONAL_TYPES = {
    "Optional[int]": int,
    "Optional[str]": str,
    "Optional[float]": float,
    "Optional[bool]": bool,
}
LIST_TYPES = {
    "List[int]": int,
    "List[str]": str,
    "List[float]": float,
    "List[bool]": bool,
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
    ):
        self.pg = async_db
        self.ch = clickhouse_db
        self.task_executor = task_executor
        self.gpt_service_stub = gpt_service_stub
        self.slack_sender = slack_sender
        self.base_url = base_url

    async def create_agent(self, user: User, is_draft: bool = False) -> CreateAgentResponse:
        """Create an agent entry in the DB and return ID immediately"""

        now = get_now_utc()
        agent = AgentMetadata(
            agent_id=str(uuid4()),
            user_id=user.user_id,
            agent_name=DEFAULT_AGENT_NAME,
            created_at=now,
            last_updated=now,
            deleted=False,
            is_draft=is_draft,
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
        agent_id_list = [metadata.agent_id for metadata in agents]
        cost_infos = await self.ch.get_agents_cost_info(agent_ids=agent_id_list)
        for agent_metadata in agents:
            if agent_metadata.agent_id in cost_infos:
                agent_metadata.cost_info = cost_infos[agent_metadata.agent_id]
        return GetAllAgentsResponse(agents=agents, sections=sections)

    async def get_agent(self, agent_id: str) -> AgentMetadata:
        agents, cost_info = await asyncio.gather(
            self.pg.get_user_all_agents(agent_ids=[agent_id]),
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
    ) -> List[str]:
        """

        Args:
            agent_id: string agent id
            emails: List of emails we want to add
            user_id: string user id

        Returns: A list of invalid emails that could not be set. An email can be invalid if
        an email can be invalid if it doesnt belong to a user, or it does but the user is not on the correct team.

        """
        # get valid users
        valid_users = await self.get_valid_notification_users(user_id=user_id)
        valid_users_to_email = {user.email: user for user in valid_users}
        email_to_user = {}
        invalid_emails_to_user = []
        for email in emails:
            user = valid_users_to_email.get(email, None)
            if user:
                email_to_user[email] = user
            else:
                invalid_emails_to_user.append(email)
        await self.pg.set_agent_subscriptions(agent_id=agent_id, emails_to_user=email_to_user)

        return invalid_emails_to_user

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
        valid_users = []
        # first we get all the teams that the user is a part of
        user = await get_users(user_id=user_id, user_ids=[user_id], include_user_enabled=True)
        coroutines = [
            list_team_members(team_id=team.team_id.id, user_id=user_id)
            for team in user[0].team_memberships
        ]
        results = await asyncio.gather(*coroutines)

        # Flatten the list of results into valid_users
        for result in results:
            valid_users += result
        return [
            Account(
                user_id=user.user_id.id, username=user.username, name=user.name, email=user.email
            )
            for user in valid_users
        ]

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
            LOGGER.exception(f"Failed to insert user message into DB with exception: {e}")
            return ChatWithAgentResponse(success=False, allow_retry=True)

        if req.is_first_prompt:
            LOGGER.info("Creating future task to generate agent name and store in DB")
            run_async_background(
                self._generate_agent_name_and_store(user.user_id, agent_id, user_msg)  # 2s
            )

            if not req.skip_agent_response:
                LOGGER.info(
                    "Creating future tasks to generate initial response and send slack message"
                )
                run_async_background(
                    self._create_initial_response(user.user_id, agent_id, user_msg)  # 2s
                )
                run_async_background(self._slack_chat_msg(req, user))  # 0.1s

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
        self, user_id: str, agent_id: str, user_msg: Message
    ) -> None:
        LOGGER.info("Generating initial response from GPT (first prompt)")
        chatbot = Chatbot(agent_id, gpt_service_stub=self.gpt_service_stub)
        chat_context = ChatContext(messages=[user_msg])
        action_decider = FirstActionDecider(agent_id=agent_id, skip_db_commit=True)

        LOGGER.info("Deciding action")
        action = await action_decider.decide_action(chat_context=chat_context)

        LOGGER.info(f"Action decided: {action}. Now generating response")
        if action == FirstAction.REFER:
            gpt_resp = await chatbot.generate_first_response_refer(chat_context=chat_context)
        elif action == FirstAction.NOTIFICATION:
            gpt_resp = await chatbot.generate_first_response_notification(chat_context=chat_context)
        elif action == FirstAction.PLAN:
            gpt_resp = await chatbot.generate_initial_preplan_response(chat_context=chat_context)
        else:  # action == FirstAction.NONE
            gpt_resp = await chatbot.generate_first_response_none(chat_context=chat_context)

        gpt_msg = Message(agent_id=agent_id, message=gpt_resp, is_user_message=False)

        LOGGER.info("Inserting GPT's response to DB and publishing GPT response")
        tasks = [
            self.pg.insert_chat_messages(messages=[gpt_msg]),
            send_chat_message(gpt_msg, self.pg, insert_message_into_db=False),
        ]

        if action == FirstAction.PLAN:
            plan_id = str(uuid4())
            LOGGER.info(f"Creating execution plan {plan_id} for {agent_id=}")
            tasks.append(
                self.task_executor.create_execution_plan(  # this isn't async
                    agent_id=agent_id,
                    plan_id=plan_id,
                    user_id=user_id,
                    run_plan_in_prefect_immediately=True,
                )
            )

        await asyncio.gather(*tasks)

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
                        f"\nfullstory_link: {user.fullstory_link.replace("https://", "")}"
                    )
                six_hours_from_now = int(time.time() + (60 * 60 * 2))
                self.slack_sender.send_message_at(
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
        self, agent_id: str, start: Optional[datetime.datetime], end: Optional[datetime.datetime]
    ) -> GetChatHistoryResponse:
        chat_context = await self.pg.get_chats_history_for_agent(agent_id, start, end)

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

        return GetChatHistoryResponse(messages=chat_context.messages)

    @async_perf_logger
    async def get_agent_worklog_board(
        self,
        agent_id: str,
        start_date: Optional[datetime.date] = None,
        end_date: Optional[datetime.date] = None,
        most_recent_num_run: Optional[int] = None,
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
        LOGGER.info("Creating a future to get latest execution plan")
        # TODO: For now just get the latest plan. Later we can switch to LIVE plan
        future_task = run_async_background(self.pg.get_latest_execution_plan(agent_id))

        LOGGER.info("Getting agent worklogs")
        run_history = await get_agent_hierarchical_worklogs(
            agent_id, self.pg, start_date, end_date, most_recent_num_run
        )

        LOGGER.info("Waiting for getting latest execution plan (future) to complete")
        (
            plan_id,
            execution_plan,
            _,
            status,
            upcoming_plan_run_id,
        ) = await future_task

        if plan_id is None or execution_plan is None:
            execution_plan_template = None
        else:
            execution_plan_template = ExecutionPlanTemplate(
                plan_id=plan_id,
                upcoming_plan_run_id=upcoming_plan_run_id,
                tasks=[
                    PlanTemplateTask(task_id=node.tool_task_id, task_name=node.description)
                    for node in execution_plan.nodes
                ],
            )

        return GetAgentWorklogBoardResponse(
            run_history=run_history,
            execution_plan_template=execution_plan_template,
            latest_plan_status=status,
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
        """
        If `plan_run_id` is None, default to get the outputs of the latest run
        """

        outputs = await self.pg.get_agent_outputs(agent_id=agent_id, plan_run_id=plan_run_id)
        if not outputs:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No output found for {agent_id=} and {plan_run_id=}",
            )

        run_metadata = outputs[0].run_metadata

        newly_updated_outputs = (run_metadata.updated_output_ids or []) if run_metadata else []
        run_summary_short = run_metadata.run_summary_short if run_metadata else None

        run_summary_long: Any = run_metadata.run_summary_long if run_metadata else None
        if isinstance(run_summary_long, Text):
            run_summary_long = await run_summary_long.to_rich_output(pg=self.pg.pg)
            run_summary_long = cast(TextOutput, run_summary_long)

            # replace summary title with widget anchor
            for output in outputs:
                widget_title = output.output.title
                if widget_title:
                    summary_title_dict = {
                        "type": "output_widget",
                        "name": widget_title,
                        "output_id": output.output_id,
                        "plan_run_id": output.plan_run_id,
                    }
                    summary_title_anchor = "```" + json.dumps(summary_title_dict) + "```"
                    run_summary_long.val = re.sub(
                        widget_title, summary_title_anchor, run_summary_long.val, count=1
                    )

        final_outputs = [output for output in outputs if not output.is_intermediate]
        if final_outputs:
            return GetAgentOutputResponse(
                outputs=final_outputs,
                run_summary_long=run_summary_long,
                run_summary_short=run_summary_short,
                newly_updated_outputs=newly_updated_outputs,
            )

        return GetAgentOutputResponse(
            outputs=outputs,
            run_summary_long=run_summary_long,
            run_summary_short=run_summary_short,
            newly_updated_outputs=newly_updated_outputs,
        )

    async def delete_agent_output(
        self, agent_id: str, req: DeleteAgentOutputRequest
    ) -> DeleteAgentOutputResponse:
        # 1. Get the plan
        plan_map = await self.pg.get_execution_plans(plan_ids=[req.plan_id])
        if not plan_map:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Plan ID not found")

        # 2. Delete and prune
        old_plan = plan_map[req.plan_id][0]
        new_plan = old_plan.get_pruned_plan(task_ids_to_remove=set(req.task_ids))

        # 3. Insert new plan
        write_plan_task = self.pg.write_execution_plan(
            plan_id=str(uuid4()), agent_id=agent_id, plan=new_plan
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
        await self.pg.lock_plan_tasks(agent_id=agent_id, plan_id=req.plan_id, task_ids=req.task_ids)

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
        await self.pg.unlock_plan_tasks(
            agent_id=agent_id, plan_id=req.plan_id, task_ids=req.task_ids
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
        user = await get_users(user_id=user_id, user_ids=[user_id], include_user_enabled=True)
        email = user[0].email
        await self.pg.set_agent_subscriptions(
            agent_id=agent_id,
            emails_to_user={
                email: Account(
                    user_id=user_id, username=user[0].username, name=user[0].name, email=email
                )
            },
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

    def get_secure_ld_user(self, user_id: str) -> GetSecureUserResponse:
        ld_user = get_user_context(user_id=user_id)
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

    async def get_autocomplete_items(self, user_id: str, text: str) -> GetAutocompleteItemsResponse:
        # Use PA Service to get all portfolios + watchlists for the user in parallel
        workspaces, resp = await asyncio.gather(
            get_all_workspaces(user_id=user_id), get_all_watchlists(user_id=user_id)
        )
        watchlists = resp.watchlists

        text_lower = text.lower()
        memory_items = []

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
    async def get_plan_run_output(self, plan_run_id: str) -> GetPlanRunOutputResponse:
        if not is_valid_uuid(plan_run_id):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid {plan_run_id=}"
            )
        outputs = await self.pg.get_plan_run_outputs(plan_run_id=plan_run_id)
        if not outputs:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=f"No output found for {plan_run_id=}"
            )

        agent_name = await self.pg.get_agent_name(outputs[0].agent_id)
        final_outputs = [output for output in outputs if not output.is_intermediate]
        if final_outputs:
            return GetPlanRunOutputResponse(outputs=final_outputs, agent_name=agent_name)

        return GetPlanRunOutputResponse(outputs=outputs, agent_name=agent_name)

    async def get_plan_run_debug_info(
        self, agent_id: str, plan_run_id: str
    ) -> GetPlanRunDebugInfoResponse:
        """
        Get plan nodes from Postgres, `agent.execution_plans` table
        Get input and output from Clickhouse, `agent.tool_calls` table
        """
        execution_plan, tool_calls, tool_prompt_infos = await asyncio.gather(
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
            arg_values = json.loads(tool_call["args"]) if tool_call else {}
            start_time_utc = tool_call["start_time_utc"] if tool_call else None
            end_time_utc = tool_call["end_time_utc"] if tool_call else None
            duration_seconds = tool_call["duration_seconds"] if tool_call else None

            arg_list: List[ToolArgInfo] = []
            for arg_name, info in tool.input_type.model_fields.items():
                clean_type_name = get_clean_type_name(info.annotation)

                # `arg` is not editable when:
                # 1) not in EDITABLE_TYPE_STRS
                # 2) is a Variable or a list of Variables
                is_editable = True
                if clean_type_name not in EDITABLE_TYPE_STRS:
                    is_editable = False

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

                arg_list.append(
                    ToolArgInfo(
                        arg_name=arg_name,
                        arg_value=arg_values.get(arg_name),
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
            - type of arg value is not in EDITABLE_TYPE_STRS
            - type of arg value is not matched with expected type
        4. Otherwise, create a new plan entry in DB and kick off the rerun
        """

        arg_mapping: Dict[str, Dict[str, Any]] = defaultdict(dict)
        for arg in req.args_to_modify:
            arg_mapping[arg.tool_id][arg.arg_name] = arg.arg_value

        type_mismatch_error_temp = (
            "The type of arg '{arg_name}' is {arg_val_type}, "
            "not matched with expected type {expected_type}"
        )
        arg_is_variable_error_temp = "The arg '{arg_name}' is a Variable, not editable"
        not_editable_type_error_temp = (
            "The type of arg '{arg_name}' is not editable (arg type: {type})"
        )
        errors: list[ValidateArgError] = []

        execution_plan = await self.pg.get_execution_plan_for_run(plan_run_id)
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
                expected_clean_type_name = get_clean_type_name(expected_type)
                if expected_clean_type_name not in EDITABLE_TYPE_STRS:
                    errors.append(
                        ValidateArgError(
                            tool_id=node.tool_task_id,
                            arg_name=arg_name,
                            error=not_editable_type_error_temp.format(
                                arg_name=arg_name, type=type(arg_val)
                            ),
                        )
                    )
                    continue

                correct = True
                if expected_clean_type_name in PRIMITIVE_TYPES:
                    class_type = PRIMITIVE_TYPES[expected_clean_type_name]
                    correct = isinstance(arg_val, class_type)
                elif expected_clean_type_name in OPTIONAL_TYPES:
                    class_type = OPTIONAL_TYPES[expected_clean_type_name]
                    correct = arg_val is None or isinstance(arg_val, class_type)
                else:
                    class_type = LIST_TYPES[expected_clean_type_name]
                    correct = isinstance(arg_val, list)
                    for item in arg_val:
                        if not isinstance(item, class_type):
                            correct = False
                            break

                if not correct:
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
            cost_info,
            gpt_service_info,
        ) = await gather_with_concurrency(tasks, n=len(tasks))

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
        for tool_plan_run_id, tool_value in tool_calls.items():
            plan_id = ""
            for tool_name, tool_call_dict in tool_value.items():
                plan_id = tool_call_dict["plan_id"]
                del tool_call_dict["plan_id"]
                del tool_call_dict["plan_run_id"]
            top_level_key = f"plan_id={plan_id}"
            plan_run_id = tool_plan_run_id
            inner_key = f"plan_run_id={plan_run_id}"
            if inner_key not in run_execution_plans[top_level_key]:
                run_execution_plans[top_level_key][inner_key] = {}
            if tool_plan_run_id == plan_run_id:
                run_execution_plans[top_level_key][inner_key]["tool_calls"] = tool_value
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

    async def get_tool_library(self) -> GetToolLibraryResponse:
        tools = []
        for tool_name, tool in ToolRegistry._REGISTRY_ALL_TOOLS_MAP.items():
            if not tool.enabled:
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
        return UpdateUserResponse(success=await update_user(user_id, name, username, email))

    async def get_account_info(self, user: User) -> GetAccountInfoResponse:
        res = await get_users(user.user_id, [user.user_id], False)
        if len(res) == 0:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No Account Info found for {user.user_id}",
            )
        account = res[0]
        return GetAccountInfoResponse(
            account=Account(
                user_id=account.user_id.id,
                email=account.email,
                username=account.username,
                name=account.name,
            )
        )

    @async_perf_logger
    @async_wrap
    def convert_markdown(self, content: str, new_type: MediaType) -> Tuple[bytes, str]:
        # preprocess
        # remove in-line citations
        content = re.sub(r" *```\{ [\s\S]*? \}``` ?", "", content)
        # gpt outputs only separate by one new line character, markdown needs two
        content = content.replace("\n", "\n\n")

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

    async def copy_agent(self, src_agent_id: str, dst_user_ids: List[str]) -> Dict[str, str]:
        res = {}
        for user_id in dst_user_ids:
            new_agent_id = str(uuid.uuid4())
            await self.pg.copy_agent(
                src_agent_id=src_agent_id, dst_agent_id=new_agent_id, dst_user_id=user_id
            )
            res[user_id] = new_agent_id
        return res

    async def list_custom_documents(self, user: User) -> ListCustomDocumentsResponse:
        try:
            resp: ListDocumentsResponse = await list_custom_docs(user_id=user.user_id)

            return ListCustomDocumentsResponse(
                documents=[
                    CustomDocumentListing(
                        file_id=listing.file_id,
                        name=listing.name,
                        base_path=listing.base_path,
                        full_path=listing.full_path,
                        type=listing.type,
                        size=listing.size,
                        is_dir=listing.is_dir,
                        listing_status=document_listing_status_to_str(
                            listing.listing_status.status
                        ),
                        upload_time=listing.upload_time.ToDatetime(),
                    )
                    for listing in resp.listings
                ],
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

    async def get_prompt_templates(self) -> List[PromptTemplate]:
        prompt_templates = await self.pg.get_prompt_templates()
        for template in prompt_templates:
            template.output_types = self._output_types_from_plan(template.plan)
        return prompt_templates

    def _output_types_from_plan(self, plan: ExecutionPlan) -> List[OutputType]:
        output_types = []
        output_nodes = [node for node in plan.nodes if node.is_output_node]

        output_type_mapping = {
            Table: OutputType.TABLE,
            LineGraph: OutputType.LINE_GRAPH,
            BarGraph: OutputType.BAR_GRAPH,
            PieGraph: OutputType.PIE_GRAPH,
        }

        for node in output_nodes:
            try:
                var_name = node.args.get("object_to_output").var_name  # type: ignore
                matching_node = next(
                    (n for n in plan.nodes if n.output_variable_name == var_name), None
                )

                if matching_node:
                    tool_name = matching_node.tool_name
                    output_type = ToolRegistry.get_tool(tool_name).return_type

                    # Find the corresponding output type or default to TEXT
                    output_types.append(
                        next(
                            (
                                out_type
                                for tool_class, out_type in output_type_mapping.items()
                                if issubclass(output_type, tool_class)
                            ),
                            OutputType.TEXT,
                        )
                    )
            except Exception:
                output_types.append(OutputType.TEXT)

        return output_types

    async def gen_template_plan(self, template_prompt: str, user: User) -> GenTemplatePlanResponse:

        LOGGER.info("Generating template plan")
        chat_context = ChatContext(
            messages=[Message(message=template_prompt, is_user_message=True)]
        )
        planner = Planner(skip_db_commit=True, send_chat=False)
        plan = await planner.create_initial_plan(chat_context=chat_context, use_sample_plans=True)
        if plan is None:
            raise HTTPException(status_code=400, detail="Plan generation failed: no plan created.")

        return GenTemplatePlanResponse(
            plan=plan,
            output_types=self._output_types_from_plan(plan),
        )

    async def create_agent_and_run_template_plan(
        self, template_prompt: str, plan: ExecutionPlan, is_draft: bool, user: User
    ) -> RunTemplatePlanResponse:

        # create a new agent
        LOGGER.info("Creating agent for the template plan")
        agent = await self.create_agent(user=user, is_draft=is_draft)
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

        # Write complete plan to db and let FE know the plan is ready
        # FE cancellation button will show up after this point
        LOGGER.info("Saving the plan in the DB")
        plan_id = str(uuid.uuid4())
        plan_run_id = str(uuid.uuid4())
        chat_context = ChatContext(
            messages=[Message(message=template_prompt, is_user_message=True)]
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
        await prefect_run_execution_plan(
            plan=plan,
            context=context,
            do_chat=True,
        )
        return RunTemplatePlanResponse(
            agent_id=agent_id,
        )

    async def get_user_has_alfa_access(self, user: User) -> bool:
        db_user = await get_users(
            user_id=user.user_id, user_ids=[user.user_id], include_user_enabled=True
        )

        if not db_user or not db_user[0]:
            return False

        # Check if feature flag has been disabled for user (giving them access) or check their db permission
        return not is_database_access_check_enabled_for_user(user_id=user.user_id) or (
            bool(db_user[0].has_alfa_access) and bool(db_user[0].cognito_enabled)  # type: ignore
        )
