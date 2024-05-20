from abc import ABC, abstractmethod

from agent_service.planner.planner_types import ExecutionPlan
from agent_service.types import PlanRunContext


class TaskExecutor(ABC):
    @abstractmethod
    async def create_execution_plan(
        self,
        agent_id: str,
        plan_id: str,
        user_id: str,
        skip_db_commit: bool = False,
        skip_task_cache: bool = False,
        run_plan_immediately: bool = True,
    ) -> None:
        pass

    @abstractmethod
    async def run_execution_plan(
        self, plan: ExecutionPlan, context: PlanRunContext, send_chat_when_finished: bool = True
    ) -> None:
        pass
