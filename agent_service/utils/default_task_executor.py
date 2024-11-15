from typing import Optional

from agent_service.planner.constants import FollowupAction
from agent_service.planner.executor import update_execution_after_input
from agent_service.planner.planner_types import ExecutionPlan
from agent_service.types import ChatContext, PlanRunContext
from agent_service.utils.async_db import AsyncDB
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
