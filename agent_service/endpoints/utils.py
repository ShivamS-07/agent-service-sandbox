import asyncio
import datetime
import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set

from agent_service.endpoints.models import PlanRun, PlanRunTask, PlanRunTaskLog, Status
from agent_service.utils.postgres import get_psql
from agent_service.utils.prefect import (
    get_prefect_plan_run_statuses,
    get_prefect_task_statuses,
)

logger = logging.getLogger(__name__)


async def get_agent_hierarchical_worklogs(
    agent_id: str,
    start_date: Optional[datetime.date] = None,  # inclusive
    end_date: Optional[datetime.date] = None,  # inclusive
    most_recent_num_runs: Optional[int] = None,
) -> List[PlanRun]:
    db = get_psql()

    plan_run_ids: Optional[List[str]] = None
    if most_recent_num_runs:
        logger.info(
            f"Getting the most recent {most_recent_num_runs} plan run IDs for agent {agent_id}..."
        )
        plan_run_ids = db.get_agent_plan_runs(agent_id, limit_num=most_recent_num_runs)

    logger.info(f"Getting worklogs for agent {agent_id}...")
    end_date_exclusive = end_date + datetime.timedelta(days=1) if end_date else None
    rows = db.get_agent_worklogs(agent_id, start_date, end_date_exclusive, plan_run_ids)

    logger.info("Getting runs statuses and tasks statuses from Prefect...")
    plan_ids = list({row["plan_id"] for row in rows})
    plan_run_ids = list({row["plan_run_id"] for row in rows})
    plan_run_id_to_status, run_task_pair_to_status = await asyncio.gather(
        get_prefect_plan_run_statuses(plan_run_ids), get_prefect_task_statuses(plan_run_ids)
    )

    logger.info("Getting task names from execution_plans table...")
    plan_id_to_plan = db.get_execution_plans(plan_ids)

    logger.info(f"Grouping work logs in hierarchical structure for agent {agent_id}...")
    plan_id_to_plan_run_ids: Dict[str, Set[str]] = defaultdict(set)
    plan_run_id_to_task_ids: Dict[str, Set[str]] = defaultdict(set)
    task_id_to_logs: Dict[str, List[PlanRunTaskLog]] = defaultdict(list)
    task_id_to_task_output: Dict[str, Dict[str, Any]] = defaultdict(dict)

    for row in rows:
        if row["plan_id"] not in plan_id_to_plan:
            logger.warning(f"Plan ID {row['plan_id']} not found in execution_plans table")
            continue

        plan_id_to_plan_run_ids[row["plan_id"]].add(row["plan_run_id"])
        plan_run_id_to_task_ids[row["plan_run_id"]].add(row["task_id"])

        if row["is_task_output"]:  # there should only be 1 task output per task
            task_id_to_task_output[row["task_id"]] = row
        else:
            task_id_to_logs[row["task_id"]].append(
                PlanRunTaskLog(
                    log_id=row["log_id"],
                    log_message=row["log_message"],
                    created_at=row["created_at"],
                )
            )

    logger.info(f"Found {len(plan_id_to_plan_run_ids)} plans for agent {agent_id}...")
    run_history: List[PlanRun] = []
    for plan_id, plan_run_ids in plan_id_to_plan_run_ids.items():
        logger.info(f"Processing {plan_id=}, found {len(plan_run_ids)} runs...")

        # get plan nodes for tasks' names
        plan_nodes = plan_id_to_plan[plan_id].nodes
        task_id_to_node = {node.tool_task_id: node for node in plan_nodes}

        for plan_run_id in plan_run_ids:
            prefect_flow_run = plan_run_id_to_status.get(plan_run_id, None)
            if prefect_flow_run is None:
                plan_run_status = Status.COMPLETE
                plan_run_start = None
                plan_run_end = None
            else:
                plan_run_status = Status.from_prefect_state(prefect_flow_run.state_type)
                plan_run_start = prefect_flow_run.start_time
                plan_run_end = prefect_flow_run.end_time

            # Gather all tasks under this plan run
            tasks: List[PlanRunTask] = []
            for task_id in plan_run_id_to_task_ids[plan_run_id]:
                prefect_task_run = run_task_pair_to_status.get((plan_run_id, task_id), None)
                if prefect_task_run is None:
                    task_status = Status.COMPLETE  # for now we assume it's complete
                    task_start = None
                    task_end = None
                else:
                    task_status = Status.from_prefect_state(prefect_task_run.state_type)
                    task_start = prefect_task_run.start_time
                    task_end = prefect_task_run.end_time

                if task_id in task_id_to_logs:  # this is a task that has logs
                    logs = task_id_to_logs[task_id]
                    logs.sort(key=lambda x: x.created_at)
                    task = PlanRunTask(
                        task_id=task_id,
                        task_name=task_id_to_node[task_id].description,
                        status=task_status,
                        start_time=task_start or logs[0].created_at,
                        end_time=task_end or logs[-1].created_at,
                        logs=logs,
                    )
                else:  # this is a task that has no logs, just task output
                    log_time = task_id_to_task_output[task_id]["created_at"]
                    task = PlanRunTask(
                        task_id=task_id,
                        task_name=task_id_to_node[task_id].description,
                        status=task_status,
                        start_time=task_start or log_time,
                        end_time=task_end or log_time,
                        logs=[],
                    )
                tasks.append(task)

            tasks.sort(key=lambda x: x.start_time)

            # reset plan status if any task is not COMPLETE
            if any(task.status == Status.ERROR for task in tasks):
                plan_run_status = Status.ERROR
            elif any(task.status == Status.CANCELLED for task in tasks):
                plan_run_status = Status.CANCELLED
            elif any(task.status == Status.RUNNING for task in tasks):
                plan_run_status = Status.RUNNING

            run_history.append(
                PlanRun(
                    plan_id=plan_id,
                    plan_run_id=plan_run_id,
                    status=plan_run_status,
                    start_time=plan_run_start or tasks[0].start_time,
                    end_time=plan_run_end or tasks[-1].end_time,
                    tasks=tasks,
                )
            )
    if most_recent_num_runs:
        # if it errors here it means the db query is wrong
        error_msg = f"Got {len(run_history)} runs than required {most_recent_num_runs=}"
        assert len(run_history) <= most_recent_num_runs, error_msg

    run_history.sort(key=lambda x: x.start_time)
    return run_history
