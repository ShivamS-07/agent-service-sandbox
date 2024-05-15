import datetime
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

from prefect import get_client
from prefect.client.schemas.filters import FlowRunFilter, FlowRunFilterName
from prefect.client.schemas.objects import StateType
from prefect.deployments import run_deployment

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


# TODO resolve conflict here
class Status(str, Enum):
    NOT_STARTED = "NOT_STARTED"
    RUNNING = "RUNNING"
    COMPLETE = "COMPLETE"
    CANCELLED = "CANCELLED"
    ERROR = "ERROR"


def _map_prefect_state_to_status(prefect_state: Optional[StateType]) -> Status:
    if prefect_state == StateType.RUNNING:
        return Status.RUNNING
    elif prefect_state == StateType.COMPLETED:
        return Status.COMPLETE
    elif prefect_state in (StateType.FAILED, StateType.CRASHED):
        return Status.ERROR
    elif prefect_state in (StateType.CANCELLING, StateType.CANCELLED):
        return Status.CANCELLED
    else:
        return Status.NOT_STARTED


@dataclass(frozen=True)
class PrefectStatus:
    status: Status
    start_time: Optional[datetime.datetime]
    end_time: Optional[datetime.datetime]


@async_perf_logger
async def get_prefect_task_statuses(
    plan_run_ids: List[str],
) -> Dict[Tuple[str, str], PrefectStatus]:
    """
    Given a list of plan run ID's, returns a mapping from (plan_run_id, task_id)
    to a on object holding prefect-related info.
    """
    async with get_client() as client:
        runs = await client.read_task_runs(
            flow_run_filter=FlowRunFilter(name=FlowRunFilterName(any_=plan_run_ids))
        )
    output = {}
    for run in runs:
        plan_run_id, task_id = run.name.split(":")
        output[(plan_run_id, task_id)] = PrefectStatus(
            status=_map_prefect_state_to_status(run.state_type),
            start_time=run.start_time,
            end_time=run.end_time,
        )

    return output


@async_perf_logger
async def get_prefect_plan_run_statuses(plan_run_ids: List[str]) -> Dict[str, PrefectStatus]:
    async with get_client() as client:
        runs = await client.read_flow_runs(
            flow_run_filter=FlowRunFilter(name=FlowRunFilterName(any_=plan_run_ids))
        )
    output = {}
    for run in runs:
        output[run.name] = PrefectStatus(
            status=_map_prefect_state_to_status(run.state_type),
            start_time=run.start_time,
            end_time=run.end_time,
        )

    return output
