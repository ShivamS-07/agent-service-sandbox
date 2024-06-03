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
    FlowRunFilterId,
    FlowRunFilterName,
    FlowRunFilterState,
    FlowRunFilterStateType,
    FlowRunFilterTags,
)
from prefect.client.schemas.objects import State, StateType
from prefect.context import FlowRunContext, get_run_context
from prefect.deployments import run_deployment
from prefect.logging.loggers import get_run_logger

from agent_service.endpoints.models import Status
from agent_service.planner.constants import (
    CREATE_EXECUTION_PLAN_FLOW_NAME,
    RUN_EXECUTION_PLAN_FLOW_NAME,
    Action,
)
from agent_service.planner.planner_types import ErrorInfo, ExecutionPlan
from agent_service.types import PlanRunContext
from agent_service.utils.logs import async_perf_logger

logger = logging.getLogger(__name__)


@async_perf_logger
async def prefect_create_execution_plan(
    agent_id: str,
    plan_id: str,
    user_id: str,
    action: Action = Action.CREATE,
    error_info: Optional[ErrorInfo] = None,
    skip_db_commit: bool = False,
    skip_task_cache: bool = False,
    run_plan_in_prefect_immediately: bool = True,
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
            "run_plan_in_prefect_immediately": run_plan_in_prefect_immediately,
            "action": action,
            "error_info": error_info.model_dump() if error_info else error_info,
        },
        tags=[agent_id],
    )


@async_perf_logger
async def prefect_run_execution_plan(
    plan: ExecutionPlan, context: PlanRunContext, do_chat: bool = True
) -> None:
    await run_deployment(
        name=f"{RUN_EXECUTION_PLAN_FLOW_NAME}/{RUN_EXECUTION_PLAN_FLOW_NAME}",
        timeout=0,
        parameters={
            "plan": plan.model_dump(),
            "context": context.model_dump(),
            "do_chat": do_chat,
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

    async with get_client() as client:  # type: ignore
        runs: List[TaskRun] = await client.read_task_runs(
            flow_run_filter=FlowRunFilter(
                name=FlowRunFilterName(any_=plan_run_ids),
            )
        )
    output = {}
    for run in runs:
        # We split by a colon since we used one in "get_task_run_name"
        try:
            plan_run_id, task_id = run.name.split(":")
        except ValueError:
            # If the task name can't be split, the task probably has not be
            # truly started yet. We can skip it for now and it'll appear in the
            # next status call.
            logger.warning(
                f"Failed to split ID string: '{run.name}', got pieces: {run.name.split(':')}"
            )
            continue
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

    async with get_client() as client:  # type: ignore
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
    async with get_client() as client:  # type: ignore
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
    # When we pause a run, store its prior state here for resuming.
    prior_state: State


async def prefect_pause_current_agent_flow(agent_id: str) -> Optional[PrefectFlowRun]:
    async with get_client() as client:  # type: ignore
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
            logger.info(f"No flow runs found to pause for {agent_id=}")
            return None
        # There should only be one
        run = runs[0]
        prior_state = run.state
        if not prior_state:
            logger.info(f"Flow run {run.id} was not running for {agent_id=}, skipped")
            return None
        try:
            # Try to pause, it will error if it's not in progress
            logger.info(f"Pausing flow run {run.id} for {agent_id=}")
            paused_state: State = State(type=StateType.PAUSED)
            await client.set_flow_run_state(flow_run_id=run.id, state=paused_state)
        except RuntimeError:
            logger.info(f"Flow run {run.id} was not running for {agent_id=}, skipped")
            return None

    # TODO find a better way to do this.
    # If the run has a parent, it's an execution, otherwise it's a creation.
    if run.parent_task_run_id:
        return PrefectFlowRun(
            flow_run_id=run.id, flow_run_type=FlowRunType.PLAN_EXECUTION, prior_state=prior_state
        )
    else:
        return PrefectFlowRun(
            flow_run_id=run.id, flow_run_type=FlowRunType.PLAN_CREATION, prior_state=prior_state
        )


async def prefect_resume_agent_flow(run: PrefectFlowRun) -> None:
    logger.info(f"Resetting flow run {run.flow_run_id} to state {run.prior_state.type}")
    async with get_client() as client:  # type: ignore
        await client.set_flow_run_state(flow_run_id=run.flow_run_id, state=run.prior_state)


async def prefect_cancel_agent_flow(run: PrefectFlowRun) -> None:
    logger.info(f"Cancelling flow run {run.flow_run_id}")
    cancelling_state: State = State(type=StateType.CANCELLED)
    async with get_client() as client:  # type: ignore
        await client.set_flow_run_state(flow_run_id=run.flow_run_id, state=cancelling_state)


async def prefect_get_current_plan_run_task_id(run: PrefectFlowRun) -> Optional[str]:
    """
    Given a plan run ID, fetch the task ID that is in progress and return it. If
    the plan has not been started, is already complete, or has failed then None
    will be returned.
    """
    async with get_client() as client:  # type: ignore
        runs = await client.read_task_runs(
            flow_run_filter=FlowRunFilter(id=FlowRunFilterId(any_=[run.flow_run_id]))
        )
    task_status_map = {r.name: r for r in runs}
    for task_id, task_run in task_status_map.items():
        if task_run.state_type in (StateType.RUNNING, StateType.PAUSED):
            return task_id

    return None


async def prefect_cancel_current_flow() -> None:
    try:
        context = get_run_context()
    except Exception:
        # Ignore any issues to handle local runs
        return
    logger = get_prefect_logger(__name__)
    if not isinstance(context, FlowRunContext):
        logger.error("Cannot cancel flow from within a task...")
        return

    if not context.flow_run:
        logger.error("Cannot cancel flow with no flow running...")
        return

    cancelling_state: State = State(type=StateType.CANCELLED)
    async with get_client() as client:  # type: ignore
        await client.set_flow_run_state(flow_run_id=context.flow_run.id, state=cancelling_state)
