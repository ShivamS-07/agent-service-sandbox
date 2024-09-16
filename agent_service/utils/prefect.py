import asyncio
import datetime
import enum
import json
import logging
from dataclasses import dataclass
from logging import Logger, LoggerAdapter
from typing import Dict, List, Optional, Tuple, Union
from uuid import UUID

import boto3
from async_lru import alru_cache
from gbi_common_py_utils.utils.event_logging import json_serial
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
from prefect.context import TaskRunContext, get_run_context
from prefect.logging.loggers import get_run_logger

from agent_service.endpoints.models import Status
from agent_service.planner.constants import FollowupAction
from agent_service.planner.planner_types import ErrorInfo, ExecutionPlan
from agent_service.types import PlanRunContext
from agent_service.unit_test_util import RUNNING_IN_UNIT_TEST
from agent_service.utils.agent_event_utils import publish_agent_execution_status
from agent_service.utils.async_utils import run_async_background
from agent_service.utils.constants import AGENT_WORKER_QUEUE, BOOSTED_DAG_QUEUE
from agent_service.utils.date_utils import get_now_utc
from agent_service.utils.feature_flags import use_boosted_dag_for_run_execution_plan
from agent_service.utils.logs import async_perf_logger
from agent_service.utils.postgres import Postgres
from agent_service.utils.s3_upload import upload_string_to_s3

logger = logging.getLogger(__name__)


@async_perf_logger
async def prefect_create_execution_plan(
    agent_id: str,
    plan_id: str,
    user_id: str,
    action: FollowupAction = FollowupAction.CREATE,
    error_info: Optional[ErrorInfo] = None,
    skip_db_commit: bool = False,
    skip_task_cache: bool = False,
    run_plan_in_prefect_immediately: bool = True,
) -> None:
    sqs = boto3.resource("sqs", region_name="us-west-2")
    arguments = {
        "agent_id": agent_id,
        "plan_id": plan_id,
        "user_id": user_id,
        "skip_db_commit": skip_db_commit,
        "skip_task_cache": skip_task_cache,
        "run_plan_in_prefect_immediately": run_plan_in_prefect_immediately,
        "action": action,
        "error_info": error_info.model_dump() if error_info else error_info,
    }
    message = {
        "method": "create_execution_plan",
        "arguments": arguments,
        "send_time_utc": get_now_utc().isoformat(),
    }
    queue = sqs.get_queue_by_name(QueueName=AGENT_WORKER_QUEUE)
    queue.send_message(MessageBody=json.dumps(message, default=json_serial))


@async_perf_logger
async def prefect_run_execution_plan(
    plan: ExecutionPlan,
    context: PlanRunContext,
    do_chat: bool = True,
    override_task_output_id_lookup: Optional[Dict[str, str]] = None,
    replan_execution_error: bool = True,
) -> None:
    sqs = boto3.resource("sqs", region_name="us-west-2")
    arguments = {
        "plan": plan.model_dump(),
        "context": context.model_dump(),
        "do_chat": do_chat,
        "override_task_output_id_lookup": override_task_output_id_lookup,
        "replan_execution_error": replan_execution_error,
    }
    message_contents = {
        "method": "run_execution_plan",
        "arguments": arguments,
        "send_time_utc": get_now_utc().isoformat(),
    }
    message = {
        "s3_path": upload_string_to_s3(data=json.dumps(message_contents, default=json_serial)),
        "method": "run_execution_plan",
        "agent_id": context.agent_id,
        "plan_id": context.plan_id,
        "plan_run_id": context.plan_run_id,
        "user_id": context.user_id,
    }
    queue_name = (
        BOOSTED_DAG_QUEUE if use_boosted_dag_for_run_execution_plan() else AGENT_WORKER_QUEUE
    )
    queue = sqs.get_queue_by_name(QueueName=queue_name)
    queue.send_message(MessageBody=json.dumps(message, default=json_serial))


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
    if RUNNING_IN_UNIT_TEST:
        return logging.getLogger(name)

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
    name: str


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
            flow_run_id=run.id,
            flow_run_type=FlowRunType.PLAN_EXECUTION,
            prior_state=prior_state,
            name=run.name,
        )
    else:
        return PrefectFlowRun(
            flow_run_id=run.id,
            flow_run_type=FlowRunType.PLAN_CREATION,
            prior_state=prior_state,
            name=run.name,
        )


async def prefect_resume_agent_flow(run: PrefectFlowRun) -> None:
    logger.info(f"Resetting flow run {run.flow_run_id} to state {run.prior_state.type}")
    async with get_client() as client:  # type: ignore
        await client.set_flow_run_state(flow_run_id=run.flow_run_id, state=run.prior_state)


async def prefect_cancel_agent_flow(
    db: Postgres,
    agent_id: str,
    plan_id: Optional[str],
    plan_run_id: Optional[str],
    flow_run: Optional[PrefectFlowRun],
) -> None:
    """
    A few steps:
    1. Try to cancel the flow run in Prefect
    2. Insert the plan run ID into the cancelled_ids table (also plan_id if we decide to not use it)
    3. Publish the execution status to FE

    The reason to do 2) and 3) is because when the task is cancelled by Prefect, FE won't know the
    status immediately. So we need to publish the event to update FE
    """
    logger.info(
        f"Inserting {plan_run_id=} into cancelled_ids table and publishing execution status"
    )

    cancelled_ids = [_id for _id in (plan_run_id, plan_id) if _id]
    if flow_run and flow_run.name not in cancelled_ids:
        cancelled_ids.append(flow_run.name)

    tasks = []
    if cancelled_ids:
        tasks.append(
            run_async_background(
                asyncio.to_thread(
                    db.multi_row_insert,
                    table_name="agent.cancelled_ids",
                    rows=[{"cancelled_id": _id} for _id in cancelled_ids],
                )
            )
        )
    if plan_run_id and plan_id:
        tasks.append(
            run_async_background(
                publish_agent_execution_status(agent_id, plan_run_id, plan_id, Status.CANCELLED)
            )
        )

    if flow_run:
        try:
            logger.info(f"Cancelling Prefect flow run {flow_run.flow_run_id}")
            async with get_client() as client:  # type: ignore
                await client.set_flow_run_state(
                    flow_run_id=flow_run.flow_run_id, state=State(type=StateType.CANCELLED)
                )
            logger.info(f"Cancelled Prefect flow run {flow_run.flow_run_id}")
        except Exception as e:
            logger.error(f"Failed to cancel Prefect flow run {flow_run.flow_run_id}: {e}")

    await asyncio.gather(*tasks)
    logger.info(f"Inserted {plan_run_id=} into cancelled_ids table and published execution status")


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


def is_inside_prefect_task() -> bool:
    try:
        context = get_run_context()
    except Exception:
        return False
    if isinstance(context, TaskRunContext):
        return True
    return False
