import asyncio
import datetime
import logging
from typing import Dict, List, Optional, Tuple

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
    plan_run_ids = list({row["plan_run_id"] for row in rows})
    plan_run_id_to_status, run_task_pair_to_status = await asyncio.gather(
        get_prefect_plan_run_statuses(plan_run_ids), get_prefect_task_statuses(plan_run_ids)
    )

    logger.info("Getting task names from execution_plans table...")
    plan_ids = list({row["plan_id"] for row in rows})
    plan_id_to_plan = db.get_execution_plans(plan_ids)

    logger.info(f"Grouping work logs in hierarchical structure for agent {agent_id}...")
    # (plan_id, plan_run_id) -> task_id -> (log_id, log_message)
    hierarchy_lookup: Dict[Tuple[str, str], Dict[str, List[PlanRunTaskLog]]] = {}
    for row in rows:
        plan_id, plan_run_id = row["plan_id"], row["plan_run_id"]
        if plan_run_id not in plan_run_id_to_status:
            # impossible if prefect doesn't have status but db has logs for this run
            logger.warning(f"Unable to find Prefect status for plan run ID {row['plan_run_id']}")
            continue

        task_id = row["task_id"]
        run_task_pair = (plan_run_id, task_id)
        if run_task_pair not in run_task_pair_to_status:
            # impossible if prefect doesn't have status but db has logs for this task
            logger.warning(f"Unable to find Prefect status for run-task pair {run_task_pair}")
            continue

        top_level_key = (plan_id, plan_run_id)
        if top_level_key not in hierarchy_lookup:
            hierarchy_lookup[top_level_key] = {}
        if task_id not in hierarchy_lookup[top_level_key]:
            hierarchy_lookup[top_level_key][task_id] = []

        hierarchy_lookup[top_level_key][task_id].append(
            PlanRunTaskLog(
                log_id=row["log_id"], log_message=row["log_message"], created_at=row["created_at"]
            )
        )

    run_history: List[PlanRun] = []
    for (plan_id, plan_run_id), task_dicts in hierarchy_lookup.items():
        nodes = plan_id_to_plan[plan_id].nodes
        task_id_to_node = {node.tool_task_id: node for node in nodes}

        tasks: List[PlanRunTask] = []
        for task_id, logs in task_dicts.items():
            if task_id in task_id_to_node:
                task_run = run_task_pair_to_status[(plan_run_id, task_id)]
                start_time = task_run.start_time if task_run.start_time else logs[0].created_at
                tasks.append(
                    PlanRunTask(
                        task_id=task_id,
                        task_name=task_id_to_node[task_id].description,
                        status=Status.from_prefect_state(task_run.state_type),
                        start_time=start_time,
                        end_time=task_run.end_time,
                        logs=sorted(logs, key=lambda x: x.created_at),
                    )
                )
        tasks.sort(key=lambda x: x.start_time)

        run_status = plan_run_id_to_status[plan_run_id]
        start_time = run_status.start_time if run_status.start_time else tasks[0].start_time
        run_history.append(
            PlanRun(
                plan_id=plan_id,
                status=Status.from_prefect_state(run_status.state_type),
                plan_run_id=plan_run_id,
                start_time=start_time,
                end_time=run_status.end_time,
                tasks=tasks,
            )
        )

    if most_recent_num_runs:
        # if it errors here it means the db query is wrong
        error_msg = f"Got {len(run_history)} runs than required {most_recent_num_runs=}"
        assert len(run_history) <= most_recent_num_runs, error_msg

    run_history.sort(key=lambda x: x.start_time)
    return run_history
