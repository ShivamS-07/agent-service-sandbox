import asyncio
import datetime
import json
import logging
import re
import time
import traceback
import uuid
from collections import defaultdict
from copy import deepcopy
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple, cast
from uuid import uuid4

import pandoc
from fastapi import HTTPException, Request, UploadFile, status
from gpt_service_proto_v1.service_grpc import GPTServiceStub

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
    CustomNotification,
    Debug,
    DeleteAgentResponse,
    DeleteMemoryResponse,
    DisableAgentAutomationResponse,
    EnableAgentAutomationResponse,
    ExecutionPlanTemplate,
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
    ListMemoryItemsResponse,
    MarkNotificationsAsReadResponse,
    MarkNotificationsAsUnreadResponse,
    MediaType,
    MemoryItem,
    NotificationEmailsResponse,
    NotificationEvent,
    PlanRunToolDebugInfo,
    PlanTemplateTask,
    RenameMemoryResponse,
    RestoreAgentResponse,
    SetAgentFeedBackRequest,
    SetAgentFeedBackResponse,
    SetAgentScheduleRequest,
    SetAgentScheduleResponse,
    SharePlanRunResponse,
    TerminateAgentResponse,
    ToolMetadata,
    ToolPromptInfo,
    Tooltips,
    UnsharePlanRunResponse,
    UpdateAgentDraftStatusResponse,
    UpdateAgentRequest,
    UpdateAgentResponse,
    UpdateUserResponse,
    UploadFileResponse,
)
from agent_service.endpoints.utils import get_agent_hierarchical_worklogs
from agent_service.external.pa_svc_client import get_all_watchlists, get_all_workspaces
from agent_service.external.user_svc_client import (
    get_users,
    list_team_members,
    update_user,
)
from agent_service.io_type_utils import load_io_type
from agent_service.io_types.citations import CitationDetailsType, CitationType
from agent_service.io_types.text import Text, TextOutput
from agent_service.planner.action_decide import FirstActionDecider
from agent_service.planner.constants import FirstAction
from agent_service.planner.planner_types import Variable
from agent_service.slack.slack_sender import SlackSender, get_user_info_slack_string
from agent_service.tool import ToolCategory, ToolRegistry
from agent_service.types import ChatContext, MemoryType, Message
from agent_service.uploads import UploadHandler
from agent_service.utils.agent_event_utils import publish_agent_name, send_chat_message
from agent_service.utils.agent_name import generate_name_for_agent
from agent_service.utils.async_db import AsyncDB
from agent_service.utils.async_utils import (
    async_wrap,
    gather_with_concurrency,
    run_async_background,
)
from agent_service.utils.clickhouse import Clickhouse
from agent_service.utils.constants import MEDIA_TO_MIMETYPE
from agent_service.utils.date_utils import get_now_utc
from agent_service.utils.event_logging import log_event
from agent_service.utils.feature_flags import (
    get_custom_user_dict,
    get_secure_mode_hash,
    get_user_context,
)
from agent_service.utils.logs import async_perf_logger
from agent_service.utils.memory_handler import MemoryHandler, get_handler
from agent_service.utils.output_utils.output_construction import get_output_from_io_type
from agent_service.utils.postgres import DEFAULT_AGENT_NAME
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
        await self.pg.delete_agent_by_id(agent_id)
        return DeleteAgentResponse(success=True)

    async def restore_agent(self, agent_id: str) -> RestoreAgentResponse:
        await self.pg.restore_agent_by_id(agent_id)
        return RestoreAgentResponse(success=True)

    async def update_agent(self, agent_id: str, req: UpdateAgentRequest) -> UpdateAgentResponse:
        await self.pg.update_agent_name(agent_id, req.agent_name)
        return UpdateAgentResponse(success=True)

    async def get_all_agent_notification_criteria(self, agent_id: str) -> List[CustomNotification]:
        return await self.pg.get_all_agent_custom_notifications(agent_id=agent_id)

    async def create_agent_notification_criteria(self, req: CreateCustomNotificationRequest) -> str:
        cn = CustomNotification(
            agent_id=req.agent_id, notification_prompt=req.notification_prompt, auto_generated=False
        )
        await self.pg.insert_agent_custom_notification(cn=cn)
        return cn.custom_notification_id

    async def delete_agent_notification_criteria(
        self, agent_id: str, custom_notification_id: str
    ) -> None:
        await self.pg.delete_agent_custom_notification_prompt(
            agent_id=agent_id, custom_notification_id=custom_notification_id
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
                # not user_email.endswith("@boosted.ai")
                # and not user_email.endswith("@gradientboostedinvestments.com")
                user.user_id
                == user.real_user_id
            ):
                if user.fullstory_link:
                    user_info_slack_string += f"\nfullstory_link: {user.fullstory_link}"
                six_hours_from_now = int(time.time() + (10))
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

        # Now automatically subscribe the agent owner to email notifications
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
        return EnableAgentAutomationResponse(success=True, next_run=next_run)

    async def disable_agent_automation(
        self, agent_id: str, user_id: str
    ) -> DisableAgentAutomationResponse:
        await self.pg.set_agent_automation_enabled(agent_id=agent_id, enabled=False)
        return DisableAgentAutomationResponse(success=True)

    async def set_agent_schedule(self, req: SetAgentScheduleRequest) -> SetAgentScheduleResponse:
        schedule, success, error_msg = await get_schedule_from_user_description(
            agent_id=req.agent_id, user_desc=req.user_schedule_description
        )
        if success:
            await self.pg.update_agent_schedule(agent_id=req.agent_id, schedule=schedule)
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

        resp = GetPlanRunDebugInfoResponse(plan_run_tools=[])
        for node in execution_plan.nodes:
            task_id = node.tool_task_id

            tool_name = node.tool_name
            tool = ToolRegistry.get_tool(tool_name)

            # if the tool hasn't been run yet, there won't be `tool_call`
            tool_call = task_id_to_tool_call.get(task_id)
            arg_values = json.loads(tool_call["args"]) if tool_call else None
            start_time_utc = tool_call["start_time_utc"] if tool_call else None
            end_time_utc = tool_call["end_time_utc"] if tool_call else None
            duration_seconds = tool_call["duration_seconds"] if tool_call else None

            # remove the constant args from `node.args` to avoid duplicates with `arg_values`
            # keep consistent with `ToolExecutionNode.resolve_arguments`
            arg_names = deepcopy(node.args)
            for arg in list(arg_names.keys()):
                val = arg_names[arg]
                if not isinstance(val, Variable | list):
                    del arg_names[arg]
                elif isinstance(val, list):
                    var_val = [v for v in val if isinstance(v, Variable)]
                    if var_val:
                        arg_names[arg] = var_val
                    else:
                        del arg_names[arg]

            tool_prompts = tool_prompt_infos.get(task_id, [])
            tool_prompt_objs = [ToolPromptInfo(**prompt) for prompt in tool_prompts]

            resp.plan_run_tools.append(
                PlanRunToolDebugInfo(
                    tool_id=task_id,
                    tool_name=tool_name,
                    tool_description=tool.description,
                    tool_comment=node.description,
                    arg_names=arg_names,
                    args=arg_values,
                    output_variable_name=node.output_variable_name,
                    start_time_utc=start_time_utc,
                    end_time_utc=end_time_utc,
                    duration_seconds=duration_seconds,
                    prompt_infos=tool_prompt_objs,
                )
            )

        return resp

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

    async def get_prompt_tempaltes(self) -> List[PromptTemplate]:
        return await self.pg.get_prompt_templates()

    async def copy_agent(self, src_agent_id: str, dst_user_ids: List[str]) -> Dict[str, str]:
        res = {}
        for user_id in dst_user_ids:
            new_agent_id = str(uuid.uuid4())
            await self.pg.copy_agent(
                src_agent_id=src_agent_id, dst_agent_id=new_agent_id, dst_user_id=user_id
            )
            res[user_id] = new_agent_id
        return res
