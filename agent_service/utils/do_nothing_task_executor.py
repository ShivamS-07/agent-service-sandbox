import logging
from typing import Optional

from agent_service.planner.constants import FollowupAction
from agent_service.planner.planner_types import ExecutionPlan
from agent_service.types import ChatContext, PlanRunContext
from agent_service.utils.async_db import AsyncDB
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
        run_plan_in_prefect_immediately: bool = True,
    ) -> None:
        LOGGER.info("create_execution_plan called")

    async def run_execution_plan(
        self, plan: ExecutionPlan, context: PlanRunContext, do_chat: bool = True
    ) -> None:
        LOGGER.info("run_execution_plan called")

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
        return FollowupAction.NONE
