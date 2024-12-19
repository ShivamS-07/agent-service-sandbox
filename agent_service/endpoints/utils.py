import asyncio
import datetime
import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import sentry_sdk
from gbi_common_py_utils.utils.environment import get_environment_tag

from agent_service.endpoints.models import (
    PlanRun,
    PlanRunStatusInfo,
    PlanRunTask,
    PlanRunTaskLog,
    Status,
    TaskRunStatusInfo,
)
from agent_service.io_type_utils import load_io_type
from agent_service.io_types.graph import BarGraph, LineGraph, PieGraph
from agent_service.io_types.table import Table
from agent_service.io_types.text import Text
from agent_service.planner.planner_types import ExecutionPlan, PlanStatus, ToolExecutionNode
from agent_service.utils.async_db import AsyncDB
from agent_service.utils.date_utils import get_now_utc
from agent_service.utils.logs import async_perf_logger
from agent_service.utils.prompt_template import OutputPreview, OutputType

logger = logging.getLogger(__name__)


def get_base_url() -> str:
    env = get_environment_tag()
    return "alfa.boosted.ai" if env == "ALPHA" else "agent-dev.boosted.ai"


@async_perf_logger
async def get_agent_hierarchical_worklogs(
    agent_id: str,
    db: AsyncDB,
    start_date: Optional[datetime.date] = None,  # inclusive
    end_date: Optional[datetime.date] = None,  # inclusive
    start_index: Optional[int] = 0,
    limit_num: Optional[int] = None,
) -> Tuple[List[PlanRun], Optional[int]]:
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
    with sentry_sdk.start_span(op="db.get_agent_plan_runs", description="get_agent_plan_runs"):
        logger.info(f"Getting plan runs for agent {agent_id}...")
        end_date_exclusive = end_date + datetime.timedelta(days=1) if end_date else None
        tuples, total_plan_count = await db.get_agent_plan_runs(
            agent_id, start_date, end_date_exclusive, start_index, limit_num
        )

    plan_run_ids = [tup[0] for tup in tuples]
    plan_ids = list({tup[1] for tup in tuples})

    with sentry_sdk.start_span(op="db", description="get_agent_worklogs_fast"):
        logger.info("Getting worklogs, statuses, execution plan task names, and cancelled ids")
        (
            rows,
            plan_run_id_to_metadata,
            run_task_pair_to_status,
            plan_id_to_plan,
            cancelled_ids,
        ) = await asyncio.gather(
            db.get_agent_worklogs_fast(agent_id, start_date, end_date_exclusive, plan_run_ids),
            db.get_plan_runs_metadata(plan_run_ids),
            db.get_task_run_statuses(plan_run_ids),
            db.get_execution_plans(plan_ids),
            db.get_cancelled_ids(ids_to_check=plan_run_ids + plan_ids),
        )

    cancelled_ids = set(cancelled_ids)

    logger.info(f"Creating lookup dictionaries for agent {agent_id}...")
    plan_run_id_task_id_to_logs: Dict[Tuple[str, str], List[PlanRunTaskLog]] = defaultdict(list)
    plan_run_id_task_id_to_task_output: Dict[Tuple[str, str], Dict[str, Any]] = defaultdict(dict)

    for row in rows:
        if row["plan_id"] not in plan_id_to_plan:
            logger.warning(f"Plan ID {row['plan_id']} not found in execution_plans table")
            continue

        if row["is_task_output"]:
            plan_run_id_task_id_to_task_output[(row["plan_run_id"], row["task_id"])] = row
        else:
            message = load_io_type(row["log_message"])
            message_str = (await message.get()).val if isinstance(message, Text) else str(message)
            plan_run_id_task_id_to_logs[(row["plan_run_id"], row["task_id"])].append(
                PlanRunTaskLog(
                    log_id=row["log_id"],
                    log_message=message_str,
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

        shared, plan_run_status_info, run_metadata = plan_run_id_to_metadata.get(
            plan_run_id, (False, None, None)
        )
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
                    shared=shared,
                    run_description=run_description,
                )
            )
            continue

        plan_run_status, plan_run_start, plan_run_end = get_plan_run_info(plan_run_status_info)

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
            plan_run_status,
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
                shared=shared,
                run_description=run_description,
                preview=get_plan_preview(plan),
            )
        )

    if limit_num:
        # if it errors here it means the db query is wrong
        error_msg = f"Got {len(run_history)} runs than required {limit_num=}"
        assert len(run_history) <= limit_num, error_msg

    run_history.sort(key=lambda x: x.start_time)
    return run_history, total_plan_count


def get_plan_run_info(
    plan_run_status_info: Optional[PlanRunStatusInfo],
) -> Tuple[Status, Optional[datetime.datetime], Optional[datetime.datetime]]:
    """
    Get plan run's status, start time and end time
    - We first check if the status is stored in DB, if so, use that
    - If not, default to NOT_STARTED
    """

    plan_run_start: Optional[datetime.datetime]
    plan_run_end: Optional[datetime.datetime]

    if plan_run_status_info:
        plan_run_status = plan_run_status_info.status
        if plan_run_status is None:
            plan_run_status = Status.NOT_STARTED
        plan_run_start = plan_run_status_info.start_time
        plan_run_end = plan_run_status_info.end_time
    else:
        plan_run_status = Status.NOT_STARTED
        plan_run_start = None
        plan_run_end = None

    return plan_run_status, plan_run_start, plan_run_end


def get_plan_run_task_info(
    run_task_pair: Tuple[str, str],
    run_task_pair_to_status: Dict[Tuple[str, str], TaskRunStatusInfo],
    has_logs: bool,
    has_task_output: bool,
    plan_run_status: Status,
) -> Tuple[Status, Optional[datetime.datetime], Optional[datetime.datetime]]:
    """
    Get plan run task's status, start time and end time
    - We first check if the status is stored in DB, if so, use that
    - If not
        - If logs or task output is found, set status to COMPLETE
        - If plan is cancelled, set status to CANCELLED
        - Otherwise, set status to NOT_STARTED just in case
    """

    task_start: Optional[datetime.datetime]
    task_end: Optional[datetime.datetime]

    if run_task_pair in run_task_pair_to_status:
        task_run_info = run_task_pair_to_status[run_task_pair]
        task_status = task_run_info.status
        if task_status is None:
            task_status = Status.NOT_STARTED
        task_start = task_run_info.start_time
        task_end = task_run_info.end_time
    else:
        task_start = None
        task_end = None

        if has_logs or has_task_output:
            logger.warning(
                f"Task info for (plan_run_id, task_id)={run_task_pair} not found. "
                "But we found logs or task output, so Set task status to NOT_STARTED"
            )
            task_status = Status.NOT_STARTED
        elif plan_run_status in (Status.CANCELLED, Status.ERROR):
            logger.warning(
                f"Task info for (plan_run_id, task_id)={run_task_pair} not found. "
                f"Set task status as {plan_run_status=}"
            )
            task_status = plan_run_status
        else:
            logger.warning(
                f"Task info for (plan_run_id, task_id)={run_task_pair} not found"
                "Set task status to NOT_STARTED"
            )
            task_status = Status.NOT_STARTED

    return task_status, task_start, task_end


def get_plan_run_task_list(
    plan_run_id: str,
    plan_nodes: List[ToolExecutionNode],
    plan_run_id_task_id_to_logs: Dict[Tuple[str, str], List[PlanRunTaskLog]],
    plan_run_id_task_id_to_task_output: Dict[Tuple[str, str], Dict[str, Any]],
    run_task_pair_to_status: Dict[Tuple[str, str], TaskRunStatusInfo],
    plan_run_status: Status,
) -> List[PlanRunTask]:
    # we want each run to have the full list of tasks with different statuses
    full_tasks: List[PlanRunTask] = []
    for node in plan_nodes:
        task_id = node.tool_task_id
        task_key = (plan_run_id, task_id)

        logs = plan_run_id_task_id_to_logs.get(task_key, [])
        logs.sort(key=lambda x: x.created_at)

        task_status, task_start, task_end = get_plan_run_task_info(
            run_task_pair=task_key,
            run_task_pair_to_status=run_task_pair_to_status,
            has_logs=bool(logs),
            has_task_output=task_key in plan_run_id_task_id_to_task_output,
            plan_run_status=plan_run_status,
        )

        full_tasks.append(
            PlanRunTask(
                task_id=task_id,
                task_name=node.description,
                status=task_status,
                start_time=task_start,
                end_time=task_end,
                logs=logs,
                has_output=node.store_output,
            )
        )

    full_tasks = reset_run_task_statuses_if_needed(full_tasks)
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


def reset_run_task_statuses_if_needed(full_tasks: List[PlanRunTask]) -> List[PlanRunTask]:
    """
    If there's a non-COMPLETE task, all the following tasks should be reset to NOT_STARTED
    """
    non_complete_statuses = {Status.RUNNING, Status.ERROR, Status.CANCELLED, Status.NOT_STARTED}
    for idx, task in enumerate(full_tasks):
        if task.status in non_complete_statuses:
            logger.info(
                f"Found non-COMPLETE task {task.task_id}, "
                "resetting the following tasks to NOT_STARTED"
            )

            idx += 1
            while idx < len(full_tasks):
                full_tasks[idx].status = Status.NOT_STARTED
                idx += 1
            break

    return full_tasks


def reset_plan_run_status_if_needed(
    plan_run_status: Status, full_tasks: List[PlanRunTask]
) -> Status:
    if plan_run_status == Status.CANCELLED:
        return plan_run_status

    if any(task.status == Status.ERROR for task in full_tasks):
        plan_run_status = Status.ERROR
    elif any(task.status == Status.CANCELLED for task in full_tasks):
        plan_run_status = Status.CANCELLED
    elif plan_run_status != Status.ERROR and any(
        task.status == Status.RUNNING for task in full_tasks
    ):
        plan_run_status = Status.RUNNING
    return plan_run_status


def get_plan_preview(plan: ExecutionPlan) -> List[OutputPreview]:
    """returns the preview which is a list of OutputPreview objects"""
    from agent_service.tool import default_tool_registry

    preview = []
    output_nodes = [node for node in plan.nodes if node.is_output_node]

    output_type_mapping = {
        Table: OutputType.TABLE,
        LineGraph: OutputType.LINE_GRAPH,
        BarGraph: OutputType.BAR_GRAPH,
        PieGraph: OutputType.PIE_GRAPH,
    }

    for node in output_nodes:
        title: str = node.args.get("title", "")  # type: ignore
        try:
            var_name = node.args.get("object_to_output").var_name  # type: ignore
            matching_node = next(
                (n for n in plan.nodes if n.output_variable_name == var_name), None
            )

            if matching_node:
                tool_name = matching_node.tool_name
                tool_out_type = default_tool_registry().get_tool(tool_name).return_type
                # default to TEXT
                output_type = OutputType.TEXT
                # Find the corresponding output type or default to TEXT
                for tool_class, out_type in output_type_mapping.items():
                    if issubclass(tool_out_type, tool_class):
                        output_type = out_type
                        continue

                preview.append(OutputPreview(output_type=output_type, title=title))
        except Exception:
            preview.append(OutputPreview(output_type=OutputType.TEXT, title=title))

    return preview
