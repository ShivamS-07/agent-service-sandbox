from abc import ABC, abstractmethod
from typing import Optional

from agent_service.planner.planner_types import ExecutionPlan
from agent_service.types import ChatContext, PlanRunContext


class TaskExecutor(ABC):
    @abstractmethod
    async def create_execution_plan(
        self,
        agent_id: str,
        plan_id: str,
        user_id: str,
        skip_db_commit: bool = False,
        skip_task_cache: bool = False,
        run_plan_in_prefect_immediately: bool = True,
    ) -> None:
        pass

    @abstractmethod
    async def run_execution_plan(
        self, plan: ExecutionPlan, context: PlanRunContext, do_chat: bool = True
    ) -> None:
        pass

    @abstractmethod
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
    ) -> None:
        pass
