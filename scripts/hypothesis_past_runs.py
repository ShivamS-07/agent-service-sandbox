# type: ignore

import datetime
import json
import logging
from typing import List, Optional, Tuple
from uuid import uuid4

from gbi_common_py_utils.utils.clickhouse_base import ClickhouseBase

from agent_service.io_type_utils import (
    IOType,
    dump_io_type,
    load_io_type,
    split_io_type_into_components,
)
from agent_service.io_types.dates import DateRange
from agent_service.io_types.stock import StockID
from agent_service.io_types.text import StockText
from agent_service.planner.planner_types import ExecutionPlan
from agent_service.tools.category_hypothesis import (
    DoCompetitiveAnalysisInput,
    GenerateSummaryForCompetitiveAnalysisInput,
    do_competitive_analysis,
    generate_summary_for_competitive_analysis,
)
from agent_service.tools.other_text import (
    GetAllTextDataForStocksInput,
    get_all_text_data_for_stocks,
)
from agent_service.tools.output import OutputArgs, prepare_output
from agent_service.types import PlanRunContext
from agent_service.utils.postgres import Postgres

logger = logging.getLogger(__name__)

ch = ClickhouseBase(environment="DEV")


def get_origin_hypothesis_and_target_stock(
    agent_id: str, plan_run_id: str, tool_name: str
) -> Tuple[str, Optional[StockID]]:
    sql = """
        SELECT args
        FROM agent.tool_calls
        WHERE agent_id = %(agent_id)s AND plan_run_id = %(plan_run_id)s
            AND tool_name = %(tool_name)s
    """
    result = ch.generic_read(
        sql=sql, params={"agent_id": agent_id, "plan_run_id": plan_run_id, "tool_name": tool_name}
    )

    d = json.loads(result[0]["args"])
    hypothesis = d["prompt"]

    if "target_stock" not in d or not d["target_stock"]:
        return hypothesis, None

    target_stock = load_io_type(json.dumps(d["target_stock"]))
    return hypothesis, target_stock


def get_tool_call_output(agent_id: str, plan_run_id: str, tool_name: str) -> IOType:
    sql = """
        SELECT result AS output
        FROM agent.tool_calls
        WHERE agent_id = %(agent_id)s AND plan_run_id = %(plan_run_id)s
            AND tool_name = %(tool_name)s
    """
    result = ch.generic_read(
        sql=sql, params={"agent_id": agent_id, "plan_run_id": plan_run_id, "tool_name": tool_name}
    )

    print({"agent_id": agent_id, "plan_run_id": plan_run_id, "tool_name": tool_name})

    row = result[0]
    return load_io_type(row["output"])


def insert_fake_past_run_entry(context: PlanRunContext, created_at: datetime.datetime) -> None:
    db = Postgres()

    # create a `plan_run` record and insert, fake `created_at`
    sql = """
        INSERT INTO agent.plan_runs (plan_run_id, agent_id, plan_id, created_at)
        VALUES (
            %(plan_run_id)s, %(agent_id)s, %(plan_id)s, %(created_at)s
        )
    """
    db.generic_write(
        sql,
        {
            "plan_run_id": context.plan_run_id,
            "agent_id": context.agent_id,
            "plan_id": context.plan_id,
            "created_at": created_at,
        },
    )


def build_plan_object(agent_id: str, plan_id: str) -> ExecutionPlan:
    sql = """
        SELECT plan
        FROM agent.execution_plans
        WHERE agent_id = %(agent_id)s AND plan_id = %(plan_id)s
    """
    db = Postgres()
    rows = db.generic_read(sql, {"agent_id": agent_id, "plan_id": plan_id})
    return ExecutionPlan.model_validate(rows[0]["plan"])


def insert_fake_past_run_worklogs(
    context: PlanRunContext, latest_run_id: str, created_at: datetime.datetime
) -> None:
    db = Postgres()

    # get the existing worklogs
    sql1 = """
        SELECT agent_id::VARCHAR, plan_id::VARCHAR, plan_run_id::VARCHAR, task_id::VARCHAR, log_message,
            log_data, is_task_output, created_at
        FROM agent.work_logs
        WHERE agent_id = %(agent_id)s AND plan_run_id = %(plan_run_id)s
        ORDER BY created_at ASC
    """
    rows = db.generic_read(sql1, {"agent_id": context.agent_id, "plan_run_id": latest_run_id})

    # replace the plan_run_id with the fake one
    ts = created_at
    for row in rows:
        ts += datetime.timedelta(seconds=1)

        row["plan_run_id"] = context.plan_run_id
        row["created_at"] = ts

    sql2 = """
        INSERT INTO agent.work_logs (
            agent_id, plan_id, plan_run_id, task_id, log_message, log_data, is_task_output, created_at
        ) VALUES (
            %(agent_id)s, %(plan_id)s, %(plan_run_id)s, %(task_id)s, %(log_message)s, %(log_data)s,
            %(is_task_output)s, %(created_at)s
        )
    """

    # we need to insert extra entries for "output" steps so we can make this run "complete" on UI
    # because run and task status are all from prefect where the local run doesn't have that
    plan = build_plan_object(context.agent_id, context.plan_id)
    output_rows = []
    for node in plan.nodes:
        if node.tool_name == "prepare_output":
            ts += datetime.timedelta(seconds=1)

            output_rows.append(
                {
                    "agent_id": context.agent_id,
                    "plan_id": context.plan_id,
                    "plan_run_id": context.plan_run_id,
                    "task_id": node.tool_task_id,
                    "log_message": node.description,
                    "log_data": "Ignore - placeHolder for testing",
                    "is_task_output": True,
                    "created_at": ts,
                }
            )

    try:
        with db.connection.cursor() as cursor:
            all_rows = rows + output_rows
            for i in range(0, len(all_rows), 10):
                cursor.executemany(sql2, all_rows[i : i + 10])
    except Exception:
        db = Postgres()
        with db.connection.cursor() as cursor:
            all_rows = rows + output_rows
            for i in range(0, len(all_rows), 10):
                cursor.executemany(sql2, all_rows[i : i + 10])


def write_past_agent_output(
    output_id: str, output: IOType, context: PlanRunContext, created_at: datetime.datetime
) -> None:
    sql = """
        INSERT INTO agent.agent_outputs
          (output_id, agent_id, plan_id, plan_run_id, output, created_at)
        VALUES
          (
             %(output_id)s, %(agent_id)s, %(plan_id)s, %(plan_run_id)s, %(output)s, %(created_at)s
          )
        """
    db = Postgres()
    db.generic_write(
        sql,
        params={
            "output_id": output_id,
            "agent_id": context.agent_id,
            "plan_id": context.plan_id,
            "plan_run_id": context.plan_run_id,
            "output": dump_io_type(output),
            "created_at": created_at,
        },
    )


def drop_plan_run(agent_id: str, plan_run_id: str) -> None:
    sql1 = """
        DELETE FROM agent.plan_runs
        WHERE agent_id = %(agent_id)s AND plan_run_id = %(plan_run_id)s
    """
    sql2 = """
        DELETE FROM agent.agent_outputs
        WHERE agent_id = %(agent_id)s AND plan_run_id = %(plan_run_id)s
    """
    sql3 = """
        DELETE FROM agent.work_logs
        WHERE agent_id = %(agent_id)s AND plan_run_id = %(plan_run_id)s
    """
    db = Postgres()
    db.generic_write(sql1, {"agent_id": agent_id, "plan_run_id": plan_run_id})
    db.generic_write(sql2, {"agent_id": agent_id, "plan_run_id": plan_run_id})
    db.generic_write(sql3, {"agent_id": agent_id, "plan_run_id": plan_run_id})


async def create_fake_past_run(
    context: PlanRunContext,
    latest_plan_run_id: str,
    start_date: datetime.datetime,  # the start date you want the agent to run
    end_date: datetime.datetime,  # the end date you want the agent to run
) -> None:
    ##########################################
    # Prepare inputs that are no need to rerun
    ##########################################
    hypothesis, target_stock = get_origin_hypothesis_and_target_stock(
        context.agent_id, latest_plan_run_id, tool_name="do_competitive_analysis"
    )

    # update with original plan run id (even after it is deleted), only uncommented in main run
    latest_plan_run_id = "0dd6cd8a-297f-4e32-9db6-0c9f7b71a6c9"

    stocks = get_tool_call_output(
        context.agent_id, latest_plan_run_id, tool_name="filter_stocks_by_product_or_service"
    )

    categories = get_tool_call_output(
        context.agent_id, latest_plan_run_id, tool_name="get_criteria_for_competitive_analysis"
    )

    temp_task_id = context.task_id
    context.task_id = None

    ####################################
    # Prepare inputs that need to rerun
    ####################################
    all_texts: List[StockText] = await get_all_text_data_for_stocks(  # type: ignore
        GetAllTextDataForStocksInput(
            stock_ids=stocks,
            date_range=DateRange(start_date=start_date.date(), end_date=end_date.date()),
        ),
        context,
    )

    context.task_id = temp_task_id

    ############################
    # Run agent as of past date
    ############################
    category_deepdives = await do_competitive_analysis(
        DoCompetitiveAnalysisInput(
            prompt=hypothesis,
            criteria=categories,  # type: ignore
            stocks=stocks,  # type: ignore
            all_text_data=all_texts,
            target_stock=target_stock,
        ),
        context,
    )
    summary = await generate_summary_for_competitive_analysis(
        GenerateSummaryForCompetitiveAnalysisInput(
            prompt=hypothesis, competitive_analysis=category_deepdives
        ),
        context,
    )

    categories = get_tool_call_output(
        context.agent_id, latest_plan_run_id, tool_name="generate_summary_for_competitive_analysis"
    )

    # write outputs

    insert_fake_past_run_entry(context, created_at=end_date)

    insert_fake_past_run_worklogs(context, latest_plan_run_id, created_at=end_date)

    summary_output = await prepare_output(
        OutputArgs(object_to_output=summary, title="Summary"), context
    )
    split_summary_outputs = await split_io_type_into_components(summary_output)
    for obj in split_summary_outputs:
        output_id = str(uuid4())
        write_past_agent_output(output_id, obj, context, created_at=end_date)

    categories_output = await prepare_output(
        OutputArgs(object_to_output=categories, title="Categories"), context
    )
    split_categories_output = await split_io_type_into_components(categories_output)
    for obj in split_categories_output:
        output_id = str(uuid4())
        write_past_agent_output(output_id, obj, context, created_at=end_date)

    deepdive_output = await prepare_output(
        OutputArgs(object_to_output=category_deepdives, title=""), context
    )
    split_deepdive_output = await split_io_type_into_components(deepdive_output)
    for obj in split_deepdive_output:
        output_id = str(uuid4())
        write_past_agent_output(output_id, obj, context, created_at=end_date)

    logger.info(f"Fake past run created: {context.plan_run_id=}")


async def main(
    agent_id: str,
    user_id: str,
    window_start: datetime.datetime,  # the start date you want the agent to run
    window_end: datetime.datetime,  # the end date you want the agent to run
) -> None:
    db = Postgres()

    # get the most recent plan_run_id
    sql = """
        SELECT agent_id::VARCHAR, plan_id::VARCHAR, plan_run_id::VARCHAR
        FROM agent.agent_outputs
        WHERE agent_id = %s
        ORDER BY created_at DESC
        LIMIT 1
    """
    row = db.generic_read(sql, (agent_id,))[0]
    plan_id = row["plan_id"]
    latest_plan_run_id = row["plan_run_id"]

    _, plan, _, _, _ = db.get_latest_execution_plan(agent_id)

    if plan:
        for step in plan.nodes:
            if step.tool_name == "do_competitive_analysis":
                task_id = step.tool_task_id
                break

    data_end = window_start
    while data_end <= window_end:
        data_start = data_end - datetime.timedelta(days=90)

        fake_past_run_id = str(uuid4())
        fake_run_context = PlanRunContext(
            agent_id=agent_id,
            user_id=user_id,
            plan_run_id=fake_past_run_id,
            plan_id=plan_id,
            task_id=task_id,
            skip_db_commit=True,  # don't want to insert weird worklogs
        )

        # comment out this code for first run
        if data_end == window_start:
            data_end += datetime.timedelta(days=1)
            continue
        fake_run_context.diff_info = {}

        await create_fake_past_run(fake_run_context, latest_plan_run_id, data_start, data_end)

        # uncomment this for first run
        # break

        data_end += datetime.timedelta(days=1)


if __name__ == "__main__":
    import asyncio

    from agent_service.utils.logs import init_stdout_logging

    agent_id = "cd1010c0-f51d-4a3c-9ddf-4c134bd6a0f1"

    # Drop the unneeded local generated runs
    # drop_plan_run(
    #     agent_id=agent_id,
    #     plan_run_id="0ae9e953-d732-46cc-8228-a0037ae8d8c8",
    # )
    # print(kjfdlsjf)

    # Explanation of how I made this work for new competitive analysis diffing
    # First, build a agent for today
    # Run this tool only for the first day of the period, see associated comments
    # in main, also need to uncomment plan_run_id in create_fake_past_run
    # Then, delete the original (current day) plan run and make opposite changes
    # in main and create_fake_past_run (must use original run id, stuff is still in clickhouse!)
    # and run for rest.

    today = datetime.datetime.today()
    window_end = today - datetime.timedelta(days=1)
    # window_end = datetime.datetime(2024, 7, 27)
    window_start = today - datetime.timedelta(days=16)
    # window_start = datetime.datetime(2024, 7, 15)

    import time

    s = time.time()

    init_stdout_logging()

    asyncio.run(
        main(
            agent_id=agent_id,  # the existing agent_id
            user_id="6953b640-16f9-4757-914e-02de6b79fab4",
            window_start=window_start,
            window_end=window_end,
        )
    )

    print(f"Total time: {time.time() - s}")
