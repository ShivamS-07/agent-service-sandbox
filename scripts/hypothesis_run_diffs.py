# type: ignore

import datetime
import logging
from typing import List

from gbi_common_py_utils.utils.clickhouse_base import ClickhouseBase

from agent_service.io_type_utils import load_io_type
from agent_service.planner.constants import NO_CHANGE_MESSAGE
from agent_service.planner.planner_types import ExecutionPlan, OutputWithID, RunMetadata
from agent_service.tools.category import Category  # noqa
from agent_service.types import PlanRunContext
from agent_service.utils.async_db import AsyncDB
from agent_service.utils.output_utils.output_diffs import (
    OutputDiffer,
    generate_full_diff_summary,
)
from agent_service.utils.postgres import Postgres, SyncBoostedPG

logger = logging.getLogger(__name__)

ch = ClickhouseBase()


def build_plan_object(agent_id: str, plan_id: str) -> ExecutionPlan:
    sql = """
        SELECT plan
        FROM agent.execution_plans
        WHERE agent_id = %(agent_id)s AND plan_id = %(plan_id)s
    """
    db = Postgres()
    rows = db.generic_read(sql, {"agent_id": agent_id, "plan_id": plan_id})
    return ExecutionPlan.model_validate(rows[0]["plan"])


def get_run_outputs(agent_id: str, plan_run_id: str) -> List[OutputWithID]:
    sql = """
        SELECT output_id::VARCHAR, task_id::VARCHAR, output
        FROM agent.agent_outputs
        WHERE agent_id = %(agent_id)s AND plan_run_id = %(plan_run_id)s
    """
    db = Postgres()
    rows = db.generic_read(sql, {"agent_id": agent_id, "plan_run_id": plan_run_id})
    return [
        OutputWithID(
            output_id=row["output_id"], task_id=row["task_id"], output=load_io_type(row["output"])
        )
        for row in rows
    ]


async def generate_diffs(
    context: PlanRunContext,
    curr_plan_run_id: str,
    past_plan_run_id: str,
    prev_run_time: datetime.datetime,
):
    plan = build_plan_object(context.agent_id, context.plan_id)

    async_db = AsyncDB(pg=SyncBoostedPG(skip_commit=context.skip_db_commit))

    # get outputs
    past_run_outputs = get_run_outputs(context.agent_id, past_plan_run_id)
    curr_run_outputs = get_run_outputs(context.agent_id, curr_plan_run_id)

    # generate diffs
    custom_notifications = await async_db.get_all_agent_custom_notifications(context.agent_id)
    custom_notification_str = "\n".join((cn.notification_prompt for cn in custom_notifications))
    output_differ = OutputDiffer(
        plan=plan, context=context, custom_notifications=custom_notification_str
    )
    output_diffs = await output_differ.diff_outputs(
        latest_outputs_with_ids=curr_run_outputs,
        db=SyncBoostedPG(skip_commit=context.skip_db_commit),
        prev_outputs=[output.output for output in past_run_outputs],
        prev_run_time=prev_run_time,
    )

    # determine if need to send notifications
    should_notify = any([diff.should_notify for diff in output_diffs])
    updated_output_ids = [
        diff.output_id for diff in output_diffs if diff.should_notify and diff.output_id
    ]

    full_diff_summary = generate_full_diff_summary(output_diffs)

    if full_diff_summary:
        logger.info(
            f"Full diff summary:\n{full_diff_summary.val}\ncitations:{len(full_diff_summary.history[0].citations)}"
        )
    else:
        logger.info("No changes")
    if not should_notify:
        logger.info("No notification necessary")
        short_diff_summary = NO_CHANGE_MESSAGE
    else:
        filtered_diff_summary = "\n".join(
            (
                f"- {diff.title}: {diff.diff_summary_message}"
                if diff.title
                else f"- {diff.diff_summary_message}"
            )
            for diff in output_diffs
            if diff.should_notify
        )

        short_diff_summary = await output_differ.generate_short_diff_summary(
            filtered_diff_summary, notification_criteria=custom_notification_str
        )
        logger.info(f"Short diff summary:\n{short_diff_summary}")

    await async_db.set_plan_run_metadata(
        context=context,
        metadata=RunMetadata(
            run_summary_long=full_diff_summary,
            run_summary_short=short_diff_summary,
            updated_output_ids=updated_output_ids,
        ),
    )


async def main(agent_id: str, user_id: str) -> None:
    db = Postgres()

    # get all plan runs for the most recent plan
    sql = """
        SELECT plan_run_id::VARCHAR, created_at, plan_id::VARCHAR
        FROM agent.plan_runs
        WHERE plan_id IN (
            SELECT plan_id
            FROM agent.plan_runs
            WHERE agent_id = %(agent_id)s
            ORDER BY created_at DESC
            LIMIT 1
        )
        ORDER BY created_at DESC
    """
    rows = db.generic_read(sql, {"agent_id": agent_id})

    logger.info(f"Found {len(rows)} plan runs")

    for i in range(len(rows) - 1):
        curr_run_id = rows[i]["plan_run_id"]
        curr_run_time = rows[i]["created_at"]
        past_run_id = rows[i + 1]["plan_run_id"]
        prev_run_time = rows[i + 1]["created_at"]

        logger.info(f"############# {curr_run_time} #############")

        run_context = PlanRunContext(
            agent_id=agent_id,
            user_id=user_id,
            plan_run_id=curr_run_id,
            plan_id=rows[i]["plan_id"],
            skip_db_commit=False,
        )
        await generate_diffs(run_context, curr_run_id, past_run_id, prev_run_time)


if __name__ == "__main__":
    import asyncio

    from agent_service.utils.logs import init_stdout_logging

    init_stdout_logging()

    # agent_id = "cd1010c0-f51d-4a3c-9ddf-4c134bd6a0f1"
    agent_id = "76a9a703-d76e-4b06-9810-a9e48158b2e8"

    asyncio.run(
        main(
            agent_id=agent_id,  # the existing agent_id
            user_id="6953b640-16f9-4757-914e-02de6b79fab4",
        )
    )
