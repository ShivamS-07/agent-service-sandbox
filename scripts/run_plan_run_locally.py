import argparse
import asyncio
import sys
from typing import Dict, List, Tuple
from unittest.mock import AsyncMock, patch

from gbi_common_py_utils.utils.clickhouse_base import ClickhouseBase

from agent_service.io_type_utils import IOType, load_io_type
from agent_service.io_types import *  # noqa
from agent_service.planner.executor import run_execution_plan_local
from agent_service.planner.planner_types import ExecutionPlan
from agent_service.types import PlanRunContext
from agent_service.utils.logs import init_stdout_logging


def fetch_task_outputs_from_clickhouse(
    plan_run_id: str, task_ids: List[str], env: str
) -> Dict[str, IOType]:
    """
    Returns a mapping from task ID to its output.
    """
    c = ClickhouseBase(environment=env)
    sql = """
    SELECT result AS output, task_id
    FROM agent.tool_calls
    WHERE plan_run_id = %(plan_run_id)s
      AND has(%(task_ids)s, task_id)
    """
    rows = c.generic_read(sql, params={"plan_run_id": plan_run_id, "task_ids": task_ids})
    output = {}
    for row in rows:
        output[row["task_id"]] = load_io_type(row["output"])

    return output


def fetch_plan_and_context_from_clickhouse(
    plan_run_id: str, env: str
) -> Tuple[ExecutionPlan, PlanRunContext]:
    c = ClickhouseBase(environment=env)
    sql = """
    SELECT simpleJSONExtractRaw(arguments, 'plan') AS plan,
      simpleJSONExtractRaw(arguments, 'context') AS context
    FROM agent.worker_sqs_log
    WHERE method='run_execution_plan'
    AND simpleJSONExtractString(context, 'plan_run_id') = %(plan_run_id)s
    LIMIT 1
    """
    rows = c.generic_read(sql, params={"plan_run_id": plan_run_id})
    if not rows:
        raise RuntimeError("Nothing found for plan run ID!")
    row = rows[0]
    return (
        ExecutionPlan.model_validate_json(row["plan"]),
        PlanRunContext.model_validate_json(row["context"]),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--plan-run-id", type=str, required=True)
    parser.add_argument("-t", "--start-with-task-id", type=str)
    parser.add_argument("-e", "--env", type=str, default="DEV")
    return parser.parse_args()


async def main() -> IOType:
    args = parse_args()
    init_stdout_logging()
    plan, context = fetch_plan_and_context_from_clickhouse(
        plan_run_id=args.plan_run_id, env=args.env
    )
    context.run_tasks_without_prefect = True
    context.skip_db_commit = True

    # Fill in the variables dict with things already computed
    override_output_dict = None
    if args.start_with_task_id:
        task_ids_to_lookup = []
        override_output_dict = {}
        var_name_to_task_id = {}
        for step in plan.nodes:
            if step.tool_task_id == args.start_with_task_id:
                break
            task_ids_to_lookup.append(step.tool_task_id)
            var_name_to_task_id[step.output_variable_name] = step.tool_task_id

        task_id_output_map = fetch_task_outputs_from_clickhouse(
            plan_run_id=args.plan_run_id, task_ids=task_ids_to_lookup, env=args.env
        )
        for task_id, output in task_id_output_map.items():
            override_output_dict[task_id] = output

    with (
        patch(target="agent_service.planner.executor.get_psql"),
        patch(target="agent_service.planner.executor.AsyncDB") as adb,
        patch(target="agent_service.planner.executor.get_agent_output"),
        patch(target="agent_service.planner.executor.check_cancelled") as cc,
        patch(target="agent_service.planner.executor.publish_agent_execution_plan"),  # noqa
        patch(target="agent_service.planner.executor.publish_agent_execution_status"),  # noqa
        patch(target="agent_service.planner.executor.publish_agent_task_status"),  # noqa
        patch(target="agent_service.planner.executor.publish_agent_plan_status"),  # noqa
        patch(target="agent_service.planner.executor.publish_agent_output"),  # noqa
        patch(target="agent_service.planner.executor.send_agent_emails"),  # noqa
        patch(target="agent_service.planner.executor.send_chat_message"),  # noqa
    ):

        cc.return_value = False
        adb().set_plan_run_metadata = AsyncMock()
        adb().update_plan_run = AsyncMock()
        adb().update_task_statuses = AsyncMock()
        result, _ = await run_execution_plan_local(
            plan=plan, context=context, override_task_output_lookup=override_output_dict
        )
    return result


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
