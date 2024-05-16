import datetime
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from prefect import get_client
from prefect.client.schemas import TaskRun
from prefect.client.schemas.filters import FlowRunFilter, FlowRunFilterName
from prefect.deployments import run_deployment

from agent_service.endpoints.models import Status
from agent_service.planner.constants import (
    CREATE_EXECUTION_PLAN_FLOW_NAME,
    RUN_EXECUTION_PLAN_FLOW_NAME,
)
from agent_service.planner.planner_types import ExecutionPlan
from agent_service.types import PlanRunContext
from agent_service.utils.logs import async_perf_logger


@async_perf_logger
async def prefect_create_execution_plan(
    agent_id: str,
    plan_id: str,
    user_id: str,
    skip_db_commit: bool = False,
    skip_task_cache: bool = False,
    run_plan_immediately: bool = True,
) -> None:
    # Deployment name has the format "flow_name/deployment_name". For use
    # they're the same.
    await run_deployment(
        name=f"{CREATE_EXECUTION_PLAN_FLOW_NAME}/{CREATE_EXECUTION_PLAN_FLOW_NAME}",
        timeout=0,
        parameters={
            "agent_id": agent_id,
            "plan_id": plan_id,
            "user_id": user_id,
            "skip_db_commit": skip_db_commit,
            "skip_task_cache": skip_task_cache,
            "run_plan_immediately": run_plan_immediately,
        },
    )


@async_perf_logger
async def prefect_run_execution_plan(
    plan: ExecutionPlan, context: PlanRunContext, send_chat_when_finished: bool = True
) -> None:
    await run_deployment(
        name=f"{RUN_EXECUTION_PLAN_FLOW_NAME}/{RUN_EXECUTION_PLAN_FLOW_NAME}",
        timeout=0,
        parameters={
            "plan": plan.model_dump(),
            "context": context.model_dump(),
            "send_chat_when_finished": send_chat_when_finished,
        },
    )


def get_task_run_name(ctx: PlanRunContext) -> str:
    return f"{ctx.plan_run_id}:{ctx.task_id}"


@dataclass(frozen=True)
class PrefectStatus:
    status: Status
    start_time: Optional[datetime.datetime]
    end_time: Optional[datetime.datetime]


@async_perf_logger
async def get_prefect_task_statuses(
    plan_run_ids: List[str],
) -> Dict[Tuple[str, str], TaskRun]:
    """
    Given a list of plan run ID's, returns a mapping from (plan_run_id, task_id)
    to a on object holding prefect-related info.

    TaskRun.state_type: Optional[StateType]
    TaskRun.start_time: Optional[datetime.datetime]
    TaskRun.end_time: Optional[datetime.datetime]
    """
    if not plan_run_ids:
        return {}

    async with get_client() as client:
        runs: List[TaskRun] = await client.read_task_runs(
            flow_run_filter=FlowRunFilter(name=FlowRunFilterName(any_=plan_run_ids))
        )
    output = {}
    for run in runs:
        plan_run_id, task_id = run.name.split(":")
        output[(plan_run_id, task_id)] = run

    return output


@async_perf_logger
async def get_prefect_plan_run_statuses(plan_run_ids: List[str]) -> Dict[str, TaskRun]:
    """
    TaskRun.state_type: Optional[StateType]
    TaskRun.start_time: Optional[datetime.datetime]
    TaskRun.end_time: Optional[datetime.datetime]
    """
    if not plan_run_ids:
        return {}

    async with get_client() as client:
        runs = await client.read_flow_runs(
            flow_run_filter=FlowRunFilter(name=FlowRunFilterName(any_=plan_run_ids))
        )
    return {run.name: run for run in runs}
