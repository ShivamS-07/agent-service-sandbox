import asyncio
import datetime
import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple, cast

from prefect.client.schemas import TaskRun

from agent_service.endpoints.models import PlanRun, PlanRunTask, PlanRunTaskLog, Status
from agent_service.io_type_utils import load_io_type
from agent_service.planner.planner_types import ToolExecutionNode
from agent_service.utils.async_db import AsyncDB
from agent_service.utils.logs import async_perf_logger
from agent_service.utils.prefect import (
    get_prefect_plan_run_statuses,
    get_prefect_task_statuses,
)

logger = logging.getLogger(__name__)


@async_perf_logger
async def get_agent_hierarchical_worklogs(
    agent_id: str,
    db: AsyncDB,
    start_date: Optional[datetime.date] = None,  # inclusive
    end_date: Optional[datetime.date] = None,  # inclusive
    most_recent_num_runs: Optional[int] = None,
) -> List[PlanRun]:
    plan_run_ids: Optional[List[str]] = None
    if most_recent_num_runs:
        logger.info(
            f"Getting the most recent {most_recent_num_runs} plan run IDs for agent {agent_id}..."
        )
        plan_run_ids = await db.get_agent_plan_runs(agent_id, limit_num=most_recent_num_runs)

    logger.info(f"Getting worklogs for agent {agent_id}...")
    end_date_exclusive = end_date + datetime.timedelta(days=1) if end_date else None
    rows = await db.get_agent_worklogs(agent_id, start_date, end_date_exclusive, plan_run_ids)

    logger.info("Getting runs statuses and tasks statuses from Prefect...")
    plan_ids = list({row["plan_id"] for row in rows})
    plan_run_ids = list({row["plan_run_id"] for row in rows})
    plan_run_id_to_status, run_task_pair_to_status = await asyncio.gather(
        get_prefect_plan_run_statuses(plan_run_ids), get_prefect_task_statuses(plan_run_ids)
    )
    logger.info("Getting task names from execution_plans table...")
    plan_id_to_plan = await db.get_execution_plans(plan_ids)

    logger.info(f"Grouping work logs in hierarchical structure for agent {agent_id}...")
    plan_id_to_plan_run_ids: Dict[str, Set[str]] = defaultdict(set)
    plan_run_id_to_task_ids: Dict[str, Set[str]] = defaultdict(set)
    plan_run_id_to_share_status: Dict[str, bool] = {}
    task_id_to_logs: Dict[str, List[PlanRunTaskLog]] = defaultdict(list)
    task_id_to_task_output: Dict[str, Dict[str, Any]] = defaultdict(dict)

    for row in rows:
        if row["plan_id"] not in plan_id_to_plan:
            logger.warning(f"Plan ID {row['plan_id']} not found in execution_plans table")
            continue

        plan_id_to_plan_run_ids[row["plan_id"]].add(row["plan_run_id"])
        plan_run_id_to_task_ids[row["plan_run_id"]].add(row["task_id"])
        plan_run_id_to_share_status[row["plan_run_id"]] = row["shared"] or False

        if row["is_task_output"]:  # there should only be 1 task output per task
            task_id_to_task_output[row["task_id"]] = row
        else:
            task_id_to_logs[row["task_id"]].append(
                PlanRunTaskLog(
                    log_id=row["log_id"],
                    log_message=cast(str, load_io_type(row["log_message"])),
                    created_at=row["created_at"],
                    has_output=row["has_output"],
                )
            )

    logger.info(f"Found {len(plan_id_to_plan_run_ids)} plans for agent {agent_id}...")
    run_history: List[PlanRun] = []
    all_plan_run_ids = []
    for plan_run_ids in plan_id_to_plan_run_ids.values():
        for plan_run_id in plan_run_ids:
            all_plan_run_ids.append(plan_run_id)

    plan_run_cancelled_ids = set(await db.get_cancelled_ids(ids_to_check=all_plan_run_ids))

    for plan_id, plan_run_ids in plan_id_to_plan_run_ids.items():
        logger.info(f"Processing {plan_id=}, found {len(plan_run_ids)} runs...")
        plan_nodes = plan_id_to_plan[plan_id].nodes

        for plan_run_id in plan_run_ids:
            prefect_flow_run = plan_run_id_to_status.get(plan_run_id, None)
            if prefect_flow_run is None:
                plan_run_status = Status.NOT_STARTED
                plan_run_start = None
                plan_run_end = None
            else:
                plan_run_status = Status.from_prefect_state(prefect_flow_run.state_type)
                if plan_run_id in plan_run_cancelled_ids:
                    plan_run_status = Status.CANCELLED
                plan_run_start = prefect_flow_run.start_time
                plan_run_end = prefect_flow_run.end_time

            full_tasks = get_plan_run_task_list(
                plan_run_id,
                plan_nodes,
                task_id_to_logs,
                task_id_to_task_output,
                run_task_pair_to_status,
            )

            plan_run_status = reset_plan_run_status_if_needed(plan_run_status, full_tasks)
            plan_run_share_status = plan_run_id_to_share_status.get(plan_run_id, False)

            run_history.append(
                PlanRun(
                    plan_id=plan_id,
                    plan_run_id=plan_run_id,
                    status=plan_run_status,
                    start_time=plan_run_start or full_tasks[0].start_time,  # type: ignore # noqa
                    end_time=plan_run_end or full_tasks[-1].end_time,
                    tasks=full_tasks,
                    shared=plan_run_share_status,
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
    task_id_to_logs: Dict[str, List[PlanRunTaskLog]],
    task_id_to_task_output: Dict[str, Dict[str, Any]],
    run_task_pair_to_status: Dict[Tuple[str, str], TaskRun],
) -> List[PlanRunTask]:
    # we want each run to have the full list of tasks with different statuses
    incomplete_tasks: List[PlanRunTask] = []
    complete_tasks: List[PlanRunTask] = []
    for node in plan_nodes:
        task_id = node.tool_task_id
        prefect_task_run = run_task_pair_to_status.get((plan_run_id, task_id), None)
        if prefect_task_run is None:
            task_status = Status.NOT_STARTED
            task_start = None
            task_end = None
        else:
            task_status = Status.from_prefect_state(prefect_task_run.state_type)
            task_start = prefect_task_run.start_time
            task_end = prefect_task_run.end_time

        if task_id not in task_id_to_task_output and task_id not in task_id_to_logs:
            incomplete_tasks.append(
                PlanRunTask(
                    task_id=task_id,
                    task_name=node.description,
                    status=task_status,
                    start_time=task_start,
                    end_time=task_end,
                    logs=[],
                    has_output=node.store_output,
                )
            )
            continue

        if task_id in task_id_to_logs:  # this is a task that has logs
            logs = task_id_to_logs[task_id]
            logs.sort(key=lambda x: x.created_at)
            task = PlanRunTask(
                task_id=task_id,
                task_name=node.description,
                status=task_status,
                start_time=task_start or logs[0].created_at,
                end_time=task_end or logs[-1].created_at,
                logs=logs,
                has_output=node.store_output,
            )
        else:  # this is a task that has no logs, just task output
            log_time = task_id_to_task_output[task_id]["created_at"]
            task = PlanRunTask(
                task_id=task_id,
                task_name=node.description,
                status=task_status,
                start_time=task_start or log_time,
                end_time=task_end or log_time,
                logs=[],
                has_output=node.store_output,
            )
        complete_tasks.append(task)

    full_tasks: List[PlanRunTask] = (
        sorted(complete_tasks, key=lambda x: x.start_time) + incomplete_tasks  # type: ignore # noqa
    )
    return full_tasks


def reset_plan_run_status_if_needed(
    plan_run_status: Status, full_tasks: List[PlanRunTask]
) -> Status:
    if any(task.status == Status.ERROR for task in full_tasks):
        plan_run_status = Status.ERROR
    elif any(task.status == Status.CANCELLED for task in full_tasks):
        plan_run_status = Status.CANCELLED
    elif any(task.status == Status.RUNNING for task in full_tasks):
        plan_run_status = Status.RUNNING
    elif all(task.status == Status.COMPLETE for task in full_tasks):
        # chances are that all tasks are completed but the prefect flow has some following garbage
        # collection steps that are not related to the actual tasks
        plan_run_status = Status.COMPLETE
    return plan_run_status
