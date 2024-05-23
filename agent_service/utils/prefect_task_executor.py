from typing import Optional

from agent_service.planner.executor import update_execution_after_input
from agent_service.planner.planner_types import ExecutionPlan
from agent_service.types import ChatContext, PlanRunContext
from agent_service.utils.prefect import (
    prefect_create_execution_plan,
    prefect_run_execution_plan,
)
from agent_service.utils.task_executor import TaskExecutor


class PrefectTaskExecutor(TaskExecutor):
    async def create_execution_plan(
        self,
        agent_id: str,
        plan_id: str,
        user_id: str,
        skip_db_commit: bool = False,
        skip_task_cache: bool = False,
        run_plan_immediately: bool = True,
    ) -> None:
        await prefect_create_execution_plan(
            agent_id=agent_id,
            plan_id=plan_id,
            user_id=user_id,
            skip_db_commit=skip_db_commit,
            skip_task_cache=skip_task_cache,
            run_plan_immediately=run_plan_immediately,
        )

    async def run_execution_plan(
        self, plan: ExecutionPlan, context: PlanRunContext, send_chat_when_finished: bool = True
    ) -> None:
        await prefect_run_execution_plan(
            plan=plan, context=context, send_chat_when_finished=send_chat_when_finished
        )

    async def update_execution_after_input(
        self,
        agent_id: str,
        user_id: str,
        skip_db_commit: bool = False,
        skip_task_cache: bool = False,
        run_plan_in_prefect_immediately: bool = True,
        run_tasks_without_prefect: bool = False,
        send_chat_when_finished: bool = True,
        chat_context: Optional[ChatContext] = None,
    ) -> None:
        await update_execution_after_input(
            agent_id=agent_id,
            user_id=user_id,
            skip_db_commit=skip_db_commit,
            skip_task_cache=skip_task_cache,
            run_plan_in_prefect_immediately=run_plan_in_prefect_immediately,
            run_tasks_without_prefect=run_tasks_without_prefect,
            send_chat_when_finished=send_chat_when_finished,
            chat_context=chat_context,
        )
