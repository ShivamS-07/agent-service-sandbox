import argparse
import asyncio
import sys
from typing import List

from agent_service.io_type_utils import load_io_type
from agent_service.io_types import *  # noqa
from agent_service.planner.planner_types import ExecutionPlan, OutputWithID
from agent_service.tools.category import Category  # noqa
from agent_service.types import PlanRunContext
from agent_service.utils.async_db import AsyncDB
from agent_service.utils.async_postgres_base import AsyncPostgresBase
from agent_service.utils.logs import init_stdout_logging
from agent_service.utils.output_utils.output_diffs import OutputDiff, OutputDiffer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--agent-id", type=str, required=True)
    parser.add_argument("-l", "--last-plan-run-id", type=str)
    parser.add_argument("-c", "--current-plan-run-id", type=str)
    parser.add_argument("-n", "--custom-notification", type=str)
    return parser.parse_args()


async def main() -> List[OutputDiff]:
    init_stdout_logging()
    args = parse_args()
    db = AsyncDB(AsyncPostgresBase())
    plan_run_ids = [args.current_plan_run_id, args.last_plan_run_id]
    if not args.last_plan_run_id or not args.current_plan_run_id:
        print("No plan run ID's passed in, using most recent two for agent...")
        tuples = await db.get_agent_plan_runs(agent_id=args.agent_id, limit_num=2)
        plan_run_ids = [tup[0] for tup in tuples[0]]

    if len(plan_run_ids) < 2:
        print("Fewer than 2 plan runs found! Exiting early...")
        return []

    plan_sql = """
    SELECT ep.plan_id::TEXT, ep.plan FROM agent.plan_runs pr
    JOIN agent.execution_plans ep
      ON ep.plan_id = pr.plan_id
    WHERE pr.plan_run_id = ANY(%(plan_run_ids)s)
    AND pr.agent_id = %(agent_id)s
    """
    rows = await db.pg.generic_read(
        plan_sql, {"plan_run_ids": plan_run_ids, "agent_id": args.agent_id}
    )
    if rows[0]["plan_id"] != rows[1]["plan_id"]:
        print(
            (
                "Plan runs did not share an underlying plan! "
                "Diff won't be able to do anything! Exiting early..."
            )
        )
        return []

    sql = """
            SELECT ao.plan_run_id::TEXT, ao.output, ao.created_at
            FROM agent.agent_outputs ao
            LEFT JOIN agent.plan_runs pr
            ON ao.plan_run_id = pr.plan_run_id
            WHERE ao.plan_run_id = ANY(%(plan_run_ids)s) AND ao.output NOTNULL
            ORDER BY ao.created_at ASC;"""

    outputs_raw = await db.pg.generic_read(sql, {"plan_run_ids": plan_run_ids})
    latest_outputs = []
    prev_outputs = []
    prev_output_date = None
    for row in outputs_raw:
        if row["plan_run_id"] == plan_run_ids[0]:
            latest_outputs.append(load_io_type(row["output"]))
        else:
            prev_outputs.append(load_io_type(row["output"]))
            prev_output_date = prev_output_date or row["created_at"]

    custom_notification = args.custom_notification or "\n".join(
        (
            cn.notification_prompt
            for cn in await db.get_all_agent_custom_notifications(agent_id=args.agent_id)
        )
    )
    plan = ExecutionPlan.from_dict(rows[0]["plan"])
    od = OutputDiffer(
        plan=plan,
        custom_notifications=custom_notification,
        context=PlanRunContext(
            agent_id=args.agent_id,
            plan_id=rows[0]["plan_id"],
            user_id="",
            plan_run_id=plan_run_ids[0],
            skip_db_commit=True,
            run_tasks_without_prefect=True,
        ),
    )
    diffs = await od.diff_outputs(
        latest_outputs_with_ids=[
            OutputWithID(output=output, output_id="") for output in latest_outputs
        ],
        db=db.pg,
        prev_outputs=prev_outputs,
        prev_run_time=prev_output_date,
    )
    return diffs


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    result = asyncio.run(main())
    print("GOT RESULT DIFFS:")
    print(result)
