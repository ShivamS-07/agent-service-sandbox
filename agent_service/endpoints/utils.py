import datetime
import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple, cast

from agent_service.endpoints.models import (
    PlanRun,
    PlanRunStatusInfo,
    PlanRunTask,
    PlanRunTaskLog,
    Status,
    TaskRunStatusInfo,
)
from agent_service.io_type_utils import load_io_type
from agent_service.planner.planner_types import (
    ExecutionPlan,
    PlanStatus,
    RunMetadata,
    ToolExecutionNode,
)
from agent_service.utils.async_db import AsyncDB
from agent_service.utils.async_utils import gather_with_concurrency
from agent_service.utils.date_utils import get_now_utc
from agent_service.utils.logs import async_perf_logger

logger = logging.getLogger(__name__)


@async_perf_logger
async def get_agent_hierarchical_worklogs(
    agent_id: str,
    db: AsyncDB,
    start_date: Optional[datetime.date] = None,  # inclusive
    end_date: Optional[datetime.date] = None,  # inclusive
    most_recent_num_runs: Optional[int] = None,
) -> List[PlanRun]:
    """
    NOTE: To get the correct results, make sure you do
    `export PREFECT_API_URL=http://prefect-dev.boosted.ai:4200/api`
    in your terminal!

    Given the agent_id, and date range/number of most recent runs, this function returns ALL the
    scheduled runs for the agent in no matter what status.
    Each run includes a list of tasks, and the finished tasks have logs

    Detailed steps:
    1. Get plan ids and plan run ids for the agent from `agent.plan_runs` table
    2. Get various types of data concurrently:
        - Get worklogs for the agent from `agent.work_logs` table given the plan runs
        - Get statuses for the plan runs
        - Get statuses for the tasks in the plan runs
        - Get execution plans for the plan ids
    3. Group worklog db rows and create lookup dictionaries
    4. Build a hierarchical structure of work logs for each plan run
        - PlanRun -> List[PlanRunTask]
        - PlanRunTask -> List[PlanRunTaskLog]
    """
    logger.info(f"Getting plan runs for agent {agent_id}...")
    end_date_exclusive = end_date + datetime.timedelta(days=1) if end_date else None
    tuples = await db.get_agent_plan_runs(
        agent_id, start_date, end_date_exclusive, most_recent_num_runs
    )
    plan_run_ids = [tup[0] for tup in tuples]
    plan_ids = list({tup[1] for tup in tuples})

    logger.info("Getting worklogs, statuses, execution plan task names, and cancelled ids")
    rows: List[Dict[str, Any]]
    plan_run_id_to_status: Dict[str, PlanRunStatusInfo]
    run_task_pair_to_status: Dict[Tuple[str, str], TaskRunStatusInfo]  # (plan_run_id, task_id)
    plan_id_to_plan: Dict[
        str, Tuple[ExecutionPlan, PlanStatus, datetime.datetime, datetime.datetime]
    ]
    cancelled_ids: Set[str]

    tasks = [
        db.get_agent_worklogs(agent_id, start_date, end_date_exclusive, plan_run_ids),
        db.get_plan_run_statuses(plan_run_ids),
        db.get_task_run_statuses(plan_run_ids),
        db.get_execution_plans(plan_ids),
        db.get_cancelled_ids(ids_to_check=plan_run_ids + plan_ids),
    ]
    results = await gather_with_concurrency(tasks)
    rows, plan_run_id_to_status, run_task_pair_to_status, plan_id_to_plan, cancelled_ids = results
    cancelled_ids = set(cancelled_ids)

    logger.info(f"Creating lookup dictionaries for agent {agent_id}...")
    plan_run_id_to_share_status: Dict[str, bool] = {}
    plan_run_id_to_run_metadata: Dict[str, Optional[RunMetadata]] = {}
    plan_run_id_task_id_to_logs: Dict[Tuple[str, str], List[PlanRunTaskLog]] = defaultdict(list)
    plan_run_id_task_id_to_task_output: Dict[Tuple[str, str], Dict[str, Any]] = defaultdict(dict)

    for row in rows:
        if row["plan_id"] not in plan_id_to_plan:
            logger.warning(f"Plan ID {row['plan_id']} not found in execution_plans table")
            continue

        plan_run_id_to_share_status[row["plan_run_id"]] = row["shared"] or False
        plan_run_id_to_run_metadata[row["plan_run_id"]] = (
            RunMetadata.model_validate(row["run_metadata"]) if row["run_metadata"] else None
        )

        if row["is_task_output"]:  # there should only be 1 task output per task
            plan_run_id_task_id_to_task_output[(row["plan_run_id"], row["task_id"])] = row
        else:
            plan_run_id_task_id_to_logs[(row["plan_run_id"], row["task_id"])].append(
                PlanRunTaskLog(
                    log_id=row["log_id"],
                    log_message=cast(str, load_io_type(row["log_message"])),
                    created_at=row["created_at"],
                    has_output=row["has_output"],
                )
            )

    logger.info(f"Creating outputs for agent {agent_id}...")
    run_history: List[PlanRun] = []
    now = get_now_utc()
    for plan_run_id, plan_id in tuples:
        plan_tup = plan_id_to_plan.get(plan_id)
        if not plan_tup:
            logger.warning(f"Plan ID {plan_id} not found in execution_plans table. Removing...")
            continue
        plan, plan_status, plan_created_at, plan_last_updated = plan_tup

        plan_run_share_status = plan_run_id_to_share_status.get(plan_run_id, False)

        run_metadata = plan_run_id_to_run_metadata.get(plan_run_id, None)
        run_description = run_metadata.run_summary_short if run_metadata else None

        # Plan with status cancelled -> run hasn't started yet
        if plan_status == PlanStatus.CANCELLED:
            run_history.append(
                PlanRun(
                    plan_id=plan_id,
                    plan_run_id=plan_run_id,
                    status=Status.CANCELLED,
                    start_time=plan_created_at,
                    end_time=plan_last_updated,
                    tasks=create_default_tasks_from_plan(plan),
                    shared=plan_run_share_status,
                    run_description=run_description,
                )
            )
            continue

        # determine the status of the plan run
        plan_run_info = plan_run_id_to_status.get(plan_run_id)
        if plan_run_info is None:
            logger.warning(f"Plan run for {plan_run_id=} not found")
            plan_run_status = Status.NOT_STARTED
            plan_run_start = None
            plan_run_end = None
        else:
            plan_run_status = plan_run_info.status or Status.COMPLETE
            plan_run_start = plan_run_info.start_time
            plan_run_end = plan_run_info.end_time

        if plan_run_id in cancelled_ids or plan_id in cancelled_ids:
            # if the id is in the cancelled_ids, override the status
            plan_run_status = Status.CANCELLED

        # create List[PlanRunTask] for the plan run
        full_tasks = get_plan_run_task_list(
            plan_run_id,
            plan.nodes,
            plan_run_id_task_id_to_logs,
            plan_run_id_task_id_to_task_output,
            run_task_pair_to_status,
        )
        plan_run_status = reset_plan_run_status_if_needed(plan_run_status, full_tasks)

        # `now` should only be used if something has gone very wrong, don't wanna fully crash.
        run_history.append(
            PlanRun(
                plan_id=plan_id,
                plan_run_id=plan_run_id,
                status=plan_run_status,
                start_time=plan_run_start or full_tasks[0].start_time or now,
                end_time=plan_run_end or full_tasks[-1].end_time or now,
                tasks=full_tasks,
                shared=plan_run_share_status,
                run_description=run_description,
            )
        )

    if most_recent_num_runs:
        # if it errors here it means the db query is wrong
        error_msg = f"Got {len(run_history)} runs than required {most_recent_num_runs=}"
        assert len(run_history) <= most_recent_num_runs, error_msg

    run_history.sort(key=lambda x: x.start_time)
    return run_history


def get_plan_run_task_list(
    plan_run_id: str,
    plan_nodes: List[ToolExecutionNode],
    plan_run_id_task_id_to_logs: Dict[Tuple[str, str], List[PlanRunTaskLog]],
    plan_run_id_task_id_to_task_output: Dict[Tuple[str, str], Dict[str, Any]],
    run_task_pair_to_status: Dict[Tuple[str, str], TaskRunStatusInfo],
) -> List[PlanRunTask]:
    # we want each run to have the full list of tasks with different statuses
    incomplete_tasks_dict: Dict[str, PlanRunTask] = {}
    complete_tasks_dict: Dict[str, PlanRunTask] = {}
    for node in plan_nodes:
        task_id = node.tool_task_id
        task_key = (plan_run_id, task_id)
        task_run_info = run_task_pair_to_status.get((plan_run_id, task_id), None)
        if task_run_info is None:
            logger.warning(f"Task info for {task_id=} not found")
            task_status = Status.COMPLETE
            task_start = None
            task_end = None
        else:
            task_status = task_run_info.status or Status.COMPLETE
            task_start = task_run_info.start_time
            task_end = task_run_info.end_time

        if task_status != Status.COMPLETE:
            incomplete_tasks_dict[task_id] = PlanRunTask(
                task_id=task_id,
                task_name=node.description,
                status=task_status,
                start_time=task_start,
                end_time=task_end,
                logs=[],
                has_output=node.store_output,
            )
            continue

        if task_key in plan_run_id_task_id_to_logs:  # this is a task that has logs
            logs = plan_run_id_task_id_to_logs[task_key]
            logs.sort(key=lambda x: x.created_at)
            task = PlanRunTask(
                task_id=task_id,
                task_name=node.description,
                status=task_status,
                start_time=task_start,
                end_time=task_end,
                logs=logs,
                has_output=node.store_output,
            )
        else:  # this is a task that has no logs, just task output
            task = PlanRunTask(
                task_id=task_id,
                task_name=node.description,
                status=task_status,
                start_time=task_start,
                end_time=task_end,
                logs=[],
                has_output=node.store_output,
            )
        complete_tasks_dict[task_id] = task
    full_tasks_dict = {**complete_tasks_dict, **incomplete_tasks_dict}
    full_tasks: List[PlanRunTask] = []
    for node in plan_nodes:
        full_tasks.append(full_tasks_dict[node.tool_task_id])
    return full_tasks


def create_default_tasks_from_plan(plan: ExecutionPlan) -> List[PlanRunTask]:
    return [
        PlanRunTask(
            task_id=node.tool_task_id,
            task_name=node.description,
            status=Status.NOT_STARTED,
            start_time=None,
            end_time=None,
            logs=[],
            has_output=node.store_output,
        )
        for node in plan.nodes
    ]


def reset_plan_run_status_if_needed(
    plan_run_status: Status, full_tasks: List[PlanRunTask]
) -> Status:
    if plan_run_status == Status.CANCELLED:
        return plan_run_status

    if any(task.status == Status.ERROR for task in full_tasks):
        plan_run_status = Status.ERROR
    elif any(task.status == Status.CANCELLED for task in full_tasks):
        plan_run_status = Status.CANCELLED
    elif any(task.status == Status.RUNNING for task in full_tasks):
        plan_run_status = Status.RUNNING
    return plan_run_status
