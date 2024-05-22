import datetime
import enum
import logging
from dataclasses import dataclass
from logging import Logger, LoggerAdapter
from typing import Dict, List, Optional, Tuple, Union
from uuid import UUID

from async_lru import alru_cache
from prefect import get_client
from prefect.client.schemas import TaskRun
from prefect.client.schemas.filters import (
    FlowRunFilter,
    FlowRunFilterName,
    FlowRunFilterState,
    FlowRunFilterStateType,
    FlowRunFilterTags,
)
from prefect.client.schemas.objects import State, StateType
from prefect.deployments import run_deployment
from prefect.engine import pause_flow_run, resume_flow_run
from prefect.logging.loggers import get_run_logger

from agent_service.endpoints.models import Status
from agent_service.planner.constants import (
    CREATE_EXECUTION_PLAN_FLOW_NAME,
    RUN_EXECUTION_PLAN_FLOW_NAME,
    Action,
)
from agent_service.planner.planner_types import ExecutionPlan
from agent_service.types import PlanRunContext
from agent_service.utils.logs import async_perf_logger

logger = logging.getLogger(__name__)


@async_perf_logger
async def prefect_create_execution_plan(
    agent_id: str,
    plan_id: str,
    user_id: str,
    action: Action = Action.CREATE,
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
            "run_plan_in_prefect_immediately": run_plan_immediately,
            "action": action,
        },
        tags=[agent_id],
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
        tags=[context.agent_id],
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
        # We split by a colon since we used one in "get_task_run_name"
        try:
            plan_run_id, task_id = run.name.split(":")
        except ValueError:
            logger.warning(
                f"Failed to split ID string: '{run.name}', got pieces: {run.name.split(':')}"
            )
            raise
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


def get_prefect_logger(name: str) -> Union[Logger, LoggerAdapter]:
    try:
        return get_run_logger()
    except Exception:
        return logging.getLogger(name)


@alru_cache(maxsize=32)
async def _get_prefect_flow_uuid_from_plan_run_id(plan_run_id: str) -> Optional[UUID]:
    async with get_client() as client:
        runs: List[TaskRun] = await client.read_task_runs(
            flow_run_filter=FlowRunFilter(name=FlowRunFilterName(any_=[plan_run_id]))
        )
    if not runs:
        return None
    return runs[0].id


class FlowRunType(enum.Enum):
    PLAN_EXECUTION = 1
    PLAN_CREATION = 2


@dataclass(frozen=True)
class PrefectFlowRun:
    flow_run_id: UUID
    flow_run_type: FlowRunType


async def prefect_pause_current_agent_flow(agent_id: str) -> Optional[PrefectFlowRun]:
    async with get_client() as client:
        # Get runs that are in progress with this agent, and pause them. Note
        # that this is *technically* a race condition, but it should be so
        # unlikely that hopefully it won't have any impact.
        runs = await client.read_flow_runs(
            flow_run_filter=FlowRunFilter(
                tags=FlowRunFilterTags(all_=[agent_id]),
                state=FlowRunFilterState(  # type: ignore
                    type=FlowRunFilterStateType(
                        any_=[StateType.PENDING, StateType.RUNNING, StateType.SCHEDULED]
                    )
                ),
            )
        )
        if not runs:
            return None
        # There should only be one
        run = runs[0]
        await pause_flow_run(flow_run_id=run.id)

    # TODO find a better way to do this.
    # If the run has a parent, it's an execution, otherwise it's a creation.
    if run.parent_task_run_id:
        return PrefectFlowRun(flow_run_id=run.id, flow_run_type=FlowRunType.PLAN_EXECUTION)
    else:
        return PrefectFlowRun(flow_run_id=run.id, flow_run_type=FlowRunType.PLAN_CREATION)


async def prefect_resume_agent_flow(run: PrefectFlowRun) -> None:
    await resume_flow_run(flow_run_id=run.flow_run_id)


async def prefect_cancel_agent_flow(run: PrefectFlowRun) -> None:
    cancelling_state: State = State(type=StateType.CANCELLING)
    async with get_client() as client:
        await client.set_flow_run_state(flow_run_id=run.flow_run_id, state=cancelling_state)


async def prefect_pause_plan_run(plan_run_id: str) -> None:
    """
    Given an plan run ID, pauses the associated prefect flow.
    """
    logger = get_prefect_logger(__name__)
    flow_run_id = await _get_prefect_flow_uuid_from_plan_run_id(plan_run_id)
    if not flow_run_id:
        logger.error(f"Tried to pause a non-existant plan run {plan_run_id}")
        return
    await pause_flow_run(flow_run_id=flow_run_id)
    logger.info(f"Paused plan run {plan_run_id}")


async def prefect_resume_plan_run(plan_run_id: str) -> None:
    logger = get_prefect_logger(__name__)
    flow_run_id = await _get_prefect_flow_uuid_from_plan_run_id(plan_run_id)
    if not flow_run_id:
        logger.error(f"Tried to resume a non-existant plan run {plan_run_id}")
        return
    await resume_flow_run(flow_run_id=flow_run_id)
    logger.info(f"Resumed plan run {plan_run_id}")


async def prefect_cancel_plan_run(plan_run_id: str) -> None:
    logger = get_prefect_logger(__name__)
    flow_run_id = await _get_prefect_flow_uuid_from_plan_run_id(plan_run_id)
    if not flow_run_id:
        logger.error(f"Tried to cancel a non-existant plan run {plan_run_id}")
        return
    async with get_client() as client:
        cancelling_state: State = State(type=StateType.CANCELLING)
        await client.set_flow_run_state(flow_run_id=flow_run_id, state=cancelling_state)
    logger.info(f"Scheduled plan run {plan_run_id} for cancellation")


async def prefect_get_current_plan_run_task_id(plan_run_id: str) -> Optional[str]:
    """
    Given a plan run ID, fetch the task ID that is in progress and return it. If
    the plan has not been started, is already complete, or has failed then None
    will be returned.
    """
    task_status_map = await get_prefect_task_statuses(plan_run_ids=[plan_run_id])
    for (_, task_id), task_run in task_status_map.items():
        if task_run.state_type in (StateType.RUNNING, StateType.PAUSED):
            return task_id

    return None
