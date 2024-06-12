import datetime
import logging
from typing import AsyncGenerator, Optional
from uuid import uuid4

from fastapi import HTTPException, status
from gpt_service_proto_v1.service_grpc import GPTServiceStub

from agent_service.chatbot.chatbot import Chatbot
from agent_service.endpoints.authz_helper import User
from agent_service.endpoints.models import (
    AgentEvent,
    AgentMetadata,
    ChatWithAgentRequest,
    ChatWithAgentResponse,
    CreateAgentResponse,
    DeleteAgentResponse,
    ExecutionPlanTemplate,
    GetAgentOutputResponse,
    GetAgentTaskOutputResponse,
    GetAgentWorklogBoardResponse,
    GetAllAgentsResponse,
    GetChatHistoryResponse,
    GetPlanRunOutputResponse,
    SharePlanRunResponse,
    UnsharePlanRunResponse,
    UpdateAgentRequest,
    UpdateAgentResponse,
)
from agent_service.endpoints.utils import get_agent_hierarchical_worklogs
from agent_service.types import ChatContext, Message
from agent_service.utils.agent_event_utils import send_chat_message
from agent_service.utils.async_db import AsyncDB
from agent_service.utils.date_utils import get_now_utc
from agent_service.utils.output_construction import get_output_from_io_type
from agent_service.utils.postgres import DEFAULT_AGENT_NAME
from agent_service.utils.redis_queue import get_agent_event_channel, wait_for_messages
from agent_service.utils.task_executor import TaskExecutor

LOGGER = logging.getLogger(__name__)


class AgentServiceImpl:
    def __init__(
        self, task_executor: TaskExecutor, gpt_service_stub: GPTServiceStub, async_db: AsyncDB
    ):
        self.pg = async_db
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

    async def delete_agent(self, agent_id: str) -> DeleteAgentResponse:
        await self.pg.delete_agent_by_id(agent_id)
        return DeleteAgentResponse(success=True)

    async def update_agent(self, agent_id: str, req: UpdateAgentRequest) -> UpdateAgentResponse:
        await self.pg.update_agent_name(agent_id, req.agent_name)
        return UpdateAgentResponse(success=True)

    async def chat_with_agent(self, req: ChatWithAgentRequest, user: User) -> ChatWithAgentResponse:
        agent_id = req.agent_id
        user_msg = Message(agent_id=agent_id, message=req.prompt, is_user_message=True)

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

        return ChatWithAgentResponse(success=True, allow_retry=False)

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
        plan_id, execution_plan, _ = await self.pg.get_latest_execution_plan(agent_id)
        if plan_id is None or execution_plan is None:
            execution_plan_template = None
        else:
            execution_plan_template = ExecutionPlanTemplate(
                plan_id=plan_id, task_names=[node.description for node in execution_plan.nodes]
            )

        return GetAgentWorklogBoardResponse(
            run_history=run_history, execution_plan_template=execution_plan_template
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

        final_outputs = [output for output in outputs if not output.is_intermediate]
        if final_outputs:
            return GetAgentOutputResponse(outputs=final_outputs)

        return GetAgentOutputResponse(outputs=outputs)

    async def stream_agent_events(self, agent_id: str) -> AsyncGenerator[AgentEvent, None]:
        LOGGER.info(f"Listening to events on channel for {agent_id=}")
        async with get_agent_event_channel(agent_id=agent_id) as channel:
            async for message in wait_for_messages(channel):
                resp = AgentEvent.model_validate_json(message)
                LOGGER.info(
                    f"Got event on channel for {agent_id=} of type '{resp.event.event_type.value}'"
                )
                yield resp

    async def share_plan_run(self, plan_run_id: str) -> SharePlanRunResponse:
        await self.pg.set_plan_run_share_status(plan_run_id=plan_run_id, status=True)
        return SharePlanRunResponse(success=True)

    async def unshare_plan_run(self, plan_run_id: str) -> UnsharePlanRunResponse:
        await self.pg.set_plan_run_share_status(plan_run_id=plan_run_id, status=False)
        return UnsharePlanRunResponse(success=True)

    async def get_plan_run_output(self, plan_run_id: str) -> GetPlanRunOutputResponse:
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
