import datetime
import logging
from typing import Optional
from uuid import uuid4

from fastapi import HTTPException, status
from gpt_service_proto_v1.service_grpc import GPTServiceStub

from agent_service.chatbot.chatbot import Chatbot
from agent_service.endpoints.authz_helper import User
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
from agent_service.utils.postgres import DEFAULT_AGENT_NAME, Postgres
from agent_service.utils.task_executor import TaskExecutor

LOGGER = logging.getLogger(__name__)


class AgentServiceImpl:
    def __init__(self, pg: Postgres, task_executor: TaskExecutor, gpt_service_stub: GPTServiceStub):
        self.pg = pg
        self.task_executor = task_executor
        self.gpt_service_stub = gpt_service_stub

    async def create_agent(self, req: CreateAgentRequest, user: User) -> CreateAgentResponse:
        """Create an agent - Client should send the first prompt from user
        1. Generate initial response from GPT -> Allow retry if fails
        2. Insert agent and messages into DB -> Allow retry if fails
        3. Kick off Prefect job -> retry is forbidden since it's a major system issue
        4. Return success or failure to client. If success, return agent ID
        """

        now = get_now_utc()
        agent_id = str(uuid4())
        try:
            LOGGER.info("Generating initial response from GPT...")
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
            chatbot = Chatbot(agent.agent_id, gpt_service_stub=self.gpt_service_stub)
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
            LOGGER.exception(f"Failed to generate initial response from GPT with exception: {e}")
            return CreateAgentResponse(success=False, allow_retry=True)

        try:
            LOGGER.info(f"Inserting agent and messages into DB for agent {agent.agent_id}...")
            self.pg.insert_agent_and_messages(agent_metadata=agent, messages=[user_msg, gpt_msg])
        except Exception as e:
            LOGGER.exception(f"Failed to insert agent and messages into DB with exception: {e}")
            return CreateAgentResponse(success=False, allow_retry=True)

        plan_id = str(uuid4())
        LOGGER.info(f"Creating execution plan {plan_id} for {agent_id=}")
        try:
            await self.task_executor.create_execution_plan(
                agent_id=agent_id, plan_id=plan_id, user_id=user.user_id, run_plan_immediately=True
            )
        except Exception:
            LOGGER.exception("Failed to kick off execution plan creation")
            return CreateAgentResponse(success=False, allow_retry=False)

        return CreateAgentResponse(success=True, allow_retry=False, agent_id=agent.agent_id)

    async def get_all_agents(self, user: User) -> GetAllAgentsResponse:
        return GetAllAgentsResponse(agents=self.pg.get_user_all_agents(user.user_id))

    async def delete_agent(self, agent_id: str) -> DeleteAgentResponse:
        self.pg.delete_agent_by_id(agent_id)
        return DeleteAgentResponse(success=True)

    async def update_agent(self, agent_id: str, req: UpdateAgentRequest) -> UpdateAgentResponse:
        self.pg.update_agent_name(agent_id, req.agent_name)
        return UpdateAgentResponse(success=True)

    async def chat_with_agent(self, req: ChatWithAgentRequest, user: User) -> ChatWithAgentResponse:
        now = get_now_utc()
        try:
            LOGGER.info("Generating initial response from GPT...")
            user_msg = Message(
                agent_id=req.agent_id,
                message=req.prompt,
                is_user_message=True,
                message_time=now,
            )
            chatbot = Chatbot(req.agent_id, gpt_service_stub=self.gpt_service_stub)
            gpt_resp = await chatbot.generate_initial_preplan_response(
                chat_context=ChatContext(messages=[user_msg])
            )
            gpt_msg = Message(
                agent_id=req.agent_id,
                message=gpt_resp,
                is_user_message=False,
            )
        except Exception as e:
            LOGGER.exception(f"Failed to generate initial response from GPT with exception: {e}")
            return ChatWithAgentResponse(success=False, allow_retry=True)

        try:
            LOGGER.info(
                f"Inserting user message and GPT response into DB for agent {req.agent_id}..."
            )
            self.pg.insert_chat_messages(messages=[user_msg, gpt_msg])
        except Exception as e:
            LOGGER.exception(f"Failed to insert messages into DB with exception: {e}")
            return ChatWithAgentResponse(success=False, allow_retry=True)

        # kick off Prefect job -> retry is forbidden
        # TODO we should check if we NEED to create a new plan
        plan_id = str(uuid4())
        LOGGER.info(f"Creating execution plan {plan_id} for {req.agent_id=}")
        try:
            await self.task_executor.create_execution_plan(
                agent_id=req.agent_id,
                plan_id=plan_id,
                user_id=user.user_id,
                run_plan_immediately=True,
            )
        except Exception:
            LOGGER.exception("Failed to kick off execution plan creation")
            return ChatWithAgentResponse(success=False, allow_retry=False)

        return ChatWithAgentResponse(success=True, allow_retry=False)

    async def get_chat_history(
        self, agent_id: str, start: Optional[datetime.datetime], end: Optional[datetime.datetime]
    ) -> GetChatHistoryResponse:
        chat_context = self.pg.get_chats_history_for_agent(agent_id, start, end)
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
        plan_id, execution_plan = self.pg.get_latest_execution_plan(agent_id)
        if plan_id is None or execution_plan is None:
            execution_plan_template = None
        else:
            execution_plan_template = ExecutionPlanTemplate(
                plan_id=plan_id, task_names=[node.description for node in execution_plan.nodes]
            )

        return GetAgentWorklogBoardResponse(
            run_history=run_history, execution_plan_template=execution_plan_template
        )

    async def get_agent_worklog_output(
        self, agent_id: str, log_id: str
    ) -> GetAgentWorklogOutputResponse:
        rows = self.pg.get_log_data_from_log_id(agent_id, log_id)
        if not rows:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"{log_id} not found")
        log_data = rows[0]["log_data"]
        output = load_io_type(log_data) if log_data is not None else None

        return GetAgentWorklogOutputResponse(output=output)

    async def get_agent_task_output(
        self, agent_id: str, plan_run_id: str, task_id: str
    ) -> GetAgentTaskOutputResponse:
        task_output = self.pg.get_task_output(
            agent_id=agent_id, plan_run_id=plan_run_id, task_id=task_id
        )

        return GetAgentTaskOutputResponse(log_data=task_output)

    async def get_agent_output(self, agent_id: str) -> GetAgentOutputResponse:
        outputs = self.pg.get_agent_outputs(agent_id=agent_id)
        if not outputs:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=f"No output found for {agent_id=}"
            )

        final_outputs = [output for output in outputs if not output.is_intermediate]
        if final_outputs:
            return GetAgentOutputResponse(outputs=final_outputs)

        return GetAgentOutputResponse(outputs=outputs)
