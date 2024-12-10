from typing import Optional

from agent_service.planner.constants import FollowupAction
from agent_service.planner.plan_creation import update_execution_after_input
from agent_service.planner.planner_types import ExecutionPlan
from agent_service.q_and_a.qa_agent import QAAgent
from agent_service.types import ChatContext, Message, PlanRunContext
from agent_service.utils.agent_event_utils import send_chat_message
from agent_service.utils.async_db import AsyncDB, get_async_db
from agent_service.utils.feature_flags import get_ld_flag_async
from agent_service.utils.prefect import (
    kick_off_create_execution_plan,
    kick_off_run_execution_plan,
)
from agent_service.utils.task_executor import TaskExecutor


class DefaultTaskExecutor(TaskExecutor):
    async def create_execution_plan(
        self,
        agent_id: str,
        plan_id: str,
        user_id: str,
        skip_db_commit: bool = False,
        skip_task_cache: bool = False,
        run_plan_in_prefect_immediately: bool = True,
    ) -> None:
        await kick_off_create_execution_plan(
            agent_id=agent_id,
            plan_id=plan_id,
            user_id=user_id,
            skip_db_commit=skip_db_commit,
            skip_task_cache=skip_task_cache,
            run_plan_in_prefect_immediately=run_plan_in_prefect_immediately,
        )

    async def run_execution_plan(
        self, plan: ExecutionPlan, context: PlanRunContext, do_chat: bool = True
    ) -> None:
        await kick_off_run_execution_plan(plan=plan, context=context, do_chat=do_chat)

    async def update_execution_after_input(
        self,
        agent_id: str,
        user_id: str,
        skip_db_commit: bool = False,
        skip_task_cache: bool = False,
        run_plan_in_prefect_immediately: bool = True,
        run_tasks_without_prefect: bool = False,
        do_chat: bool = True,
        chat_context: Optional[ChatContext] = None,
        async_db: Optional[AsyncDB] = None,
    ) -> Optional[FollowupAction]:
        if not async_db:
            async_db = get_async_db()

        is_live = await async_db.get_agent_automation_enabled(agent_id=agent_id)
        if is_live and await get_ld_flag_async(
            flag_name="run-q-a-for-live-agents", user_id=user_id, async_db=async_db, default=False
        ):
            if not chat_context:
                chat_context = await async_db.get_chats_history_for_agent(agent_id=agent_id)
            qa_agent = QAAgent(agent_id=agent_id, chat_context=chat_context, user_id=user_id)
            query = chat_context.get_latest_user_message()
            if not query:
                # Should never get here
                return None
            latest_plan_run, _ = await async_db.get_agent_plan_runs(agent_id=agent_id, limit_num=1)
            response = await qa_agent.query(
                query=str(query.message), about_plan_run_id=latest_plan_run[0][0]
            )
            if do_chat:
                await send_chat_message(
                    message=Message(agent_id=agent_id, message=response, is_user_message=False)
                )
        else:
            res = await update_execution_after_input(
                agent_id=agent_id,
                user_id=user_id,
                skip_db_commit=skip_db_commit,
                skip_task_cache=skip_task_cache,
                run_plan_in_prefect_immediately=run_plan_in_prefect_immediately,
                run_tasks_without_prefect=run_tasks_without_prefect,
                do_chat=do_chat,
                chat_context=chat_context,
                async_db=async_db,
            )
            if not res:
                return None
            return res[2]

        return None
