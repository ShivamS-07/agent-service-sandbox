# flake8: noqa
import datetime

from agent_service.utils.date_utils import (
    MockDate,
    enable_mock_time,
    get_now_utc,
    increment_mock_time,
    set_mock_time,
)

# this must be monkeypatched before most of our imports
# override today() to use our mocked current time
datetime.date = MockDate  # type:ignore

import argparse
import asyncio
import logging
import sys
from typing import Any, Dict, List, Tuple
from unittest.mock import AsyncMock, patch

from gbi_common_py_utils.utils.clickhouse_base import ClickhouseBase

from agent_service.io_type_utils import IOType, load_io_type
from agent_service.io_types import *  # noqa
from agent_service.planner.executor import run_execution_plan_local
from agent_service.planner.planner_types import ExecutionPlan
from agent_service.types import PlanRunContext
from agent_service.utils.logs import init_stdout_logging
from agent_service.utils.postgres import Postgres

logger = logging.getLogger(__name__)


def get_plan_run_info(plan_run_id: str, env: str) -> Dict[str, Any]:
    # check the create and update times for the plan id
    # we do this because currently the create date is generated on db server
    # at row insert but we supply a last_updated datetime in code when updating status.
    # return any plan runs within the time period
    sql = """
    SELECT plan_run_id::VARCHAR, created_at, agent_id::VARCHAR, 
    plan_id::VARCHAR , last_updated
    FROM agent.plan_runs
    WHERE plan_run_id = %(plan_run_id)s
    """

    db = Postgres(environment=env)
    params = {
        "plan_run_id": plan_run_id,
    }
    rows = db.generic_read(sql, params=params)
    return rows[0]


def get_plan_context_postgres(plan_run_id: str, env: str) -> Tuple[ExecutionPlan, PlanRunContext]:
    db = Postgres(environment=env)

    # agent_id::VARCHAR, plan_id::VARCHAR, created_at,
    plan_info = db.get_plan_run(plan_run_id)
    if not plan_info:
        raise Exception(f"{plan_run_id=} not found")

    sql = """
    SELECT DISTINCT ON (ag.agent_id) ag.agent_id::TEXT, ag.user_id::TEXT, ep.plan_id::TEXT, ep.plan,
    ag.schedule
    FROM agent.agents ag JOIN agent.execution_plans ep ON ag.agent_id = ep.agent_id
    WHERE ag.agent_id = %(agent_id)s and ep.plan_id = %(plan_id)s
    -- ORDER BY ag.agent_id, ep.created_at DESC
    """
    params = {"agent_id": plan_info["agent_id"], "plan_id": plan_info["plan_id"]}

    rows = db.generic_read(sql, params=params)
    agent_info = rows[0]
    plan = ExecutionPlan(**agent_info["plan"])

    plan_run_info = get_plan_run_info(plan_run_id, env=env)

    # get all the chat history from the plan run and everything prior
    chat_context = db.get_chats_history_for_agent(
        agent_id=plan_info["agent_id"], end=plan_run_info["last_updated"]
    )
    context = PlanRunContext(
        agent_id=agent_info["agent_id"],
        plan_id=agent_info["plan_id"],
        user_id=agent_info["user_id"],
        plan_run_id=plan_run_id,
        chat=chat_context,
        run_tasks_without_prefect=True,
        as_of_date=plan_run_info["last_updated"],
    )

    return plan, context


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
    # comment this out if you want to run the plan_run_id but for today()
    # instead of the same day it was originally tun
    enable_mock_time()
    args = parse_args()
    init_stdout_logging()
    try:
        plan, context = fetch_plan_and_context_from_clickhouse(
            plan_run_id=args.plan_run_id, env=args.env
        )
    except Exception as e:
        logger.warning(f"falling back to postgres for plan/context retrieval: {repr(e)}")
        plan, context = get_plan_context_postgres(plan_run_id=args.plan_run_id, env=args.env)

    context.run_tasks_without_prefect = True
    context.skip_db_commit = True

    print("==================================")
    if context.as_of_date:
        set_mock_time(context.as_of_date)
        print(f"overriding current time to: {context.as_of_date=}")
    else:
        plan_run_info = get_plan_run_info(plan_run_id=args.plan_run_id, env=args.env)
        last_update = plan_run_info["last_update"]
        context.as_of_date = last_update
        set_mock_time(last_update)
        print(f"overriding current time to: {last_update=}")
    print("==================================")

    # move time slightly forward
    increment_mock_time()

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
