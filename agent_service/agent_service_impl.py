import asyncio
import datetime
import json
import logging
import traceback
from collections import defaultdict
from typing import Any, AsyncGenerator, Dict, List, Optional
from uuid import uuid4

from fastapi import HTTPException, Request, UploadFile, status
from gpt_service_proto_v1.service_grpc import GPTServiceStub

from agent_service.chatbot.chatbot import Chatbot
from agent_service.endpoints.authz_helper import User
from agent_service.endpoints.models import (
    AgentEvent,
    AgentMetadata,
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
    GetAgentDebugInfoResponse,
    GetAgentOutputResponse,
    GetAgentTaskOutputResponse,
    GetAgentWorklogBoardResponse,
    GetAllAgentsResponse,
    GetAutocompleteItemsResponse,
    GetChatHistoryResponse,
    GetMemoryContentResponse,
    GetPlanRunOutputResponse,
    GetSecureUserResponse,
    GetTestCaseInfoResponse,
    GetTestCasesResponse,
    GetTestSuiteRunInfoResponse,
    GetTestSuiteRunsIdsResponse,
    ListMemoryItemsResponse,
    MarkNotificationsAsReadResponse,
    MarkNotificationsAsUnreadResponse,
    MemoryItem,
    NotificationEvent,
    PlanTemplateTask,
    RenameMemoryResponse,
    SetAgentScheduleRequest,
    SetAgentScheduleResponse,
    SharePlanRunResponse,
    Tooltips,
    UnsharePlanRunResponse,
    UpdateAgentRequest,
    UpdateAgentResponse,
    UploadFileResponse,
)
from agent_service.endpoints.utils import get_agent_hierarchical_worklogs
from agent_service.external.pa_svc_client import (
    get_all_watchlists,
    get_all_workspaces,
    rename_watchlist,
    rename_workspace,
)
from agent_service.io_type_utils import TableColumnType, load_io_type
from agent_service.io_types.table import (
    STOCK_ID_COL_NAME_DEFAULT,
    TableOutput,
    TableOutputColumn,
)
from agent_service.types import ChatContext, MemoryType, Message
from agent_service.uploads import UploadHandler
from agent_service.utils.agent_event_utils import send_chat_message
from agent_service.utils.agent_name import generate_name_for_agent
from agent_service.utils.async_db import AsyncDB
from agent_service.utils.clickhouse import Clickhouse
from agent_service.utils.date_utils import get_now_utc
from agent_service.utils.feature_flags import (
    get_custom_user_dict,
    get_secure_mode_hash,
    get_user_context,
)
from agent_service.utils.output_utils.output_construction import get_output_from_io_type
from agent_service.utils.postgres import DEFAULT_AGENT_NAME
from agent_service.utils.redis_queue import (
    get_agent_event_channel,
    get_notification_event_channel,
    wait_for_messages,
)
from agent_service.utils.scheduling import (
    AgentSchedule,
    get_schedule_from_user_description,
)
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
    ):
        self.pg = async_db
        self.ch = clickhouse_db
        self.task_executor = task_executor
        self.gpt_service_stub = gpt_service_stub

    async def create_agent(self, user: User) -> CreateAgentResponse:
        """Create an agent entry in the DB and return ID immediately"""

        now = get_now_utc()
        agent = AgentMetadata(
            agent_id=str(uuid4()),
            user_id=user.user_id,
            agent_name=DEFAULT_AGENT_NAME,
            created_at=now,
            last_updated=now,
        )
        await self.pg.create_agent(agent)
        return CreateAgentResponse(success=True, allow_retry=False, agent_id=agent.agent_id)

    async def get_all_agents(self, user: User) -> GetAllAgentsResponse:
        agents = await self.pg.get_user_all_agents(user.user_id)
        return GetAllAgentsResponse(agents=agents)

    async def get_agent(self, user: User, agent_id: str) -> AgentMetadata:
        agents = await self.pg.get_user_all_agents(user_id=user.user_id, agent_ids=[agent_id])
        if not agents:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=f"No agent found for {agent_id=}"
            )
        return agents[0]

    async def delete_agent(self, agent_id: str) -> DeleteAgentResponse:
        await self.pg.delete_agent_by_id(agent_id)
        return DeleteAgentResponse(success=True)

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

    async def chat_with_agent(self, req: ChatWithAgentRequest, user: User) -> ChatWithAgentResponse:
        agent_id = req.agent_id
        user_msg = Message(agent_id=agent_id, message=req.prompt, is_user_message=True)
        name = None

        # TODO should clean this up to prevent duplication
        if req.skip_agent_response:
            LOGGER.info(f"Inserting user's new message to DB for {agent_id=} WITHOUT A RESPONSE")
            await self.pg.insert_chat_messages(messages=[user_msg])

            if req.is_first_prompt:
                try:
                    LOGGER.info("Generating name for agent")
                    existing_agents = await self.pg.get_existing_agents_names(user.user_id)
                    name = await generate_name_for_agent(
                        agent_id=agent_id,
                        chat_context=ChatContext(messages=[user_msg]),
                        existing_names=existing_agents,
                        gpt_service_stub=self.gpt_service_stub,
                    )
                    await self.pg.update_agent_name(agent_id=agent_id, agent_name=name)
                except Exception as e:
                    LOGGER.exception(
                        f"Failed to generate name for agent from GPT with exception: {e}"
                    )

            return ChatWithAgentResponse(success=True, allow_retry=False, name=name)

        if not req.is_first_prompt:
            try:
                LOGGER.info(f"Inserting user's new message to DB for {agent_id=}")
                await self.pg.insert_chat_messages(messages=[user_msg])
            except Exception as e:
                LOGGER.exception(f"Failed to insert user message into DB with exception: {e}")
                return ChatWithAgentResponse(success=False, allow_retry=True)
            try:
                LOGGER.info(f"Updating execution plan after user's new message for {req.agent_id=}")
                await self.task_executor.update_execution_after_input(
                    agent_id=req.agent_id, user_id=user.user_id, chat_context=None
                )
            except Exception as e:
                LOGGER.exception((f"Failed to update {agent_id=} execution plan: {e}"))
                return ChatWithAgentResponse(success=False, allow_retry=False)
        else:
            try:
                LOGGER.info("Generating initial response from GPT (first prompt)")
                chatbot = Chatbot(agent_id, gpt_service_stub=self.gpt_service_stub)
                gpt_resp = await chatbot.generate_initial_preplan_response(
                    chat_context=ChatContext(messages=[user_msg])
                )

                LOGGER.info("Inserting user's and GPT's messages to DB")
                gpt_msg = Message(agent_id=agent_id, message=gpt_resp, is_user_message=False)
                await self.pg.insert_chat_messages(messages=[user_msg, gpt_msg])
            except Exception as e:
                # FE should retry if this fails
                LOGGER.exception(
                    f"Failed to generate initial response from GPT with exception: {e}"
                )
                return ChatWithAgentResponse(success=False, allow_retry=True)
            try:
                LOGGER.info("Generating name for agent")
                existing_agents = await self.pg.get_existing_agents_names(user.user_id)
                name = await generate_name_for_agent(
                    agent_id=agent_id,
                    chat_context=ChatContext(messages=[user_msg]),
                    existing_names=existing_agents,
                    gpt_service_stub=self.gpt_service_stub,
                )
                await self.pg.update_agent_name(agent_id=agent_id, agent_name=name)
            except Exception as e:
                LOGGER.exception(f"Failed to generate name for agent from GPT with exception: {e}")
            try:
                LOGGER.info("Publishing GPT response to Redis")
                await send_chat_message(gpt_msg, self.pg, insert_message_into_db=False)
            except Exception as e:
                LOGGER.exception(f"Failed to publish GPT response to Redis: {e}")
                return ChatWithAgentResponse(success=False, allow_retry=False)

            plan_id = str(uuid4())
            LOGGER.info(f"Creating execution plan {plan_id} for {agent_id=}")
            try:
                await self.task_executor.create_execution_plan(
                    agent_id=agent_id,
                    plan_id=plan_id,
                    user_id=user.user_id,
                    run_plan_in_prefect_immediately=True,
                )
            except Exception:
                LOGGER.exception("Failed to kick off execution plan creation")
                return ChatWithAgentResponse(success=False, allow_retry=False)

        return ChatWithAgentResponse(success=True, allow_retry=False, name=name)

    async def get_chat_history(
        self, agent_id: str, start: Optional[datetime.datetime], end: Optional[datetime.datetime]
    ) -> GetChatHistoryResponse:
        chat_context = await self.pg.get_chats_history_for_agent(agent_id, start, end)
        return GetChatHistoryResponse(messages=chat_context.messages)

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

        run_history = await get_agent_hierarchical_worklogs(
            agent_id, self.pg, start_date, end_date, most_recent_num_run
        )

        # TODO: For now just get the latest plan. Later we can switch to LIVE plan
        (
            plan_id,
            execution_plan,
            _,
            status,
            upcoming_plan_run_id,
        ) = await self.pg.get_latest_execution_plan(agent_id)
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

    async def get_agent_output(self, agent_id: str) -> GetAgentOutputResponse:
        outputs = await self.pg.get_agent_outputs(agent_id=agent_id)
        if not outputs:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=f"No output found for {agent_id=}"
            )

        if outputs:
            # Will be the same for all of these outputs
            metadata = outputs[0].run_metadata
            return GetAgentOutputResponse(
                outputs=outputs,
                run_summary_long=metadata.run_summary_long if metadata else None,
                run_summary_short=metadata.run_summary_short if metadata else None,
                newly_updated_outputs=(metadata.updated_output_ids or []) if metadata else [],
            )
        return GetAgentOutputResponse(outputs=outputs)

    async def get_agent_plan_output(
        self, agent_id: str, plan_run_id: str
    ) -> GetAgentOutputResponse:
        outputs = await self.pg.get_agent_outputs(agent_id=agent_id, plan_run_id=plan_run_id)
        if not outputs:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No output found for {agent_id=} and {plan_run_id=}",
            )

        final_outputs = [output for output in outputs if not output.is_intermediate]
        if final_outputs:
            return GetAgentOutputResponse(outputs=final_outputs)

        return GetAgentOutputResponse(outputs=outputs)

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

    async def enable_agent_automation(self, agent_id: str) -> EnableAgentAutomationResponse:
        await self.pg.set_agent_automation_enabled(agent_id=agent_id, enabled=True)
        schedule = await self.pg.get_agent_schedule(agent_id=agent_id)
        if not schedule:
            schedule = AgentSchedule.default()
            await self.pg.update_agent_schedule(agent_id=agent_id, schedule=schedule)
        next_run = schedule.get_next_run()
        return EnableAgentAutomationResponse(success=True, next_run=next_run)

    async def disable_agent_automation(self, agent_id: str) -> DisableAgentAutomationResponse:
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
        cols = [TableOutputColumn(name=STOCK_ID_COL_NAME_DEFAULT, col_type=TableColumnType.STOCK)]
        if type == MemoryType.PORTFOLIO:
            cols.append(TableOutputColumn(name="Weight", col_type=TableColumnType.FLOAT))

        table = TableOutput(title="Memory Content", columns=cols, rows=[])

        return GetMemoryContentResponse(output=table)

    async def delete_memory(self, user_id: str, type: str, id: str) -> DeleteMemoryResponse:
        return DeleteMemoryResponse(success=True)

    async def rename_memory(
        self, user_id: str, type: str, id: str, new_name: str
    ) -> RenameMemoryResponse:
        if new_name == "":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="name cannot be empty"
            )
        try:
            if type == MemoryType.PORTFOLIO:
                return RenameMemoryResponse(success=await rename_workspace(user_id, id, new_name))
            if type == MemoryType.WATCHLIST:
                return RenameMemoryResponse(success=await rename_watchlist(user_id, id, new_name))
        # catch rpc error and raise as HTTPException
        except ValueError as e:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=repr(e))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=f"type {type} is not supported"
        )

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

    async def get_agent_debug_info(self, agent_id: str) -> GetAgentDebugInfoResponse:
        agent_owner_id: Optional[str] = await self.pg.get_agent_owner(agent_id)
        plan_selections: List[Dict[str, Any]] = self.ch.get_agent_debug_plan_selections(
            agent_id=agent_id
        )
        all_generated_plans: List[Dict[str, Any]] = self.ch.get_agent_debug_plans(agent_id=agent_id)
        worker_sqs_log: Dict[str, Any] = self.ch.get_agent_debug_worker_sqs_log(agent_id=agent_id)
        tool_calls: Dict[str, Any] = self.ch.get_agent_debug_tool_calls(agent_id=agent_id)
        cost_info: Dict[str, Any] = self.ch.get_agent_debug_cost_info(agent_id=agent_id)
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
        )
        return GetAgentDebugInfoResponse(tooltips=tool_tips, debug=debug)

    async def get_info_for_test_suite_run(self, test_run_id: str) -> GetTestSuiteRunInfoResponse:
        infos = self.ch.get_info_for_test_suite_run(test_run_id=test_run_id)
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

    def get_test_suite_runs(self) -> GetTestSuiteRunsIdsResponse:
        return GetTestSuiteRunsIdsResponse(test_suite_run_ids=self.ch.get_test_suite_run_ids())

    def get_test_cases(self) -> GetTestCasesResponse:
        return GetTestCasesResponse(test_cases=self.ch.get_test_cases())

    async def get_info_for_test_case(self, test_name: str) -> GetTestCaseInfoResponse:
        infos = self.ch.get_info_for_test_case(test_name=test_name)
        for info in infos:
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
                        f"version={info['service_version']}: {traceback.format_exc()}"
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
