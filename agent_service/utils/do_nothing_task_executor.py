import logging

from agent_service.planner.planner_types import ExecutionPlan
from agent_service.types import PlanRunContext
from agent_service.utils.task_executor import TaskExecutor

LOGGER = logging.getLogger(__name__)


class DoNothingTaskExecutor(TaskExecutor):
    async def create_execution_plan(
        self,
        agent_id: str,
        plan_id: str,
        user_id: str,
        skip_db_commit: bool = False,
        skip_task_cache: bool = False,
        run_plan_immediately: bool = True,
    ) -> None:
        LOGGER.info("create_execution_plan called")

    async def run_execution_plan(
        self, plan: ExecutionPlan, context: PlanRunContext, send_chat_when_finished: bool = True
    ) -> None:
        LOGGER.info("run_execution_plan called")