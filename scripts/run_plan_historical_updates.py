# flake8: noqa
# E402 module level import not at top of file

# example run:
# pipenv run python -m scripts.run_plan_historical_updates --agent '67f99b79-c786-47c9-b139-a86a883addc0' --start-date '2024-10-22'  --end-date '2024-10-25'

######################################################################
# Runs the most recent plan for the agent on the dates specified
#
# WARNING THIS SCRIPT WILL DELETE ANY PLAN RUNS DURING THE SPECIFIED
# TIME PERIOD AND REPLACE THEM WITH NEW RUNS LOCALLY GENERATED
#
######################################################################

import argparse
import asyncio
import datetime
import json
import logging
import os
import sys
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple, Union
from uuid import uuid4

from agent_service.io_type_utils import (
    IOType,
    dump_io_type,
    load_io_type,
    split_io_type_into_components,
)
from agent_service.io_types.dates import DateRange
from agent_service.io_types.stock import StockID
from agent_service.io_types.text import StockText
from agent_service.planner.executor import run_execution_plan
from agent_service.planner.planner_types import ExecutionPlan
from agent_service.tools.output import OutputArgs, prepare_output
from agent_service.types import PlanRunContext
from agent_service.utils.date_utils import (
    enable_mock_time,
    get_now_utc,
    increment_mock_time,
    set_mock_time,
)
from agent_service.utils.logs import init_stdout_logging
from agent_service.utils.postgres import Postgres, get_psql

# override today() to use our mocked current time
# enable_mock_time()


logger = logging.getLogger(__name__)


def get_plan_runs(
    db: Postgres, agent: str, plan_id: str, start_date: datetime.date, end_date: datetime.date
) -> List[Dict[str, Any]]:
    # check the create and update times for the plan id
    # we do this because currently the create date is generated on db server
    # at row insert but we supply a last_updated datetime in code when updating status.
    # return any plan runs within the time period
    sql = """
    SELECT plan_run_id::VARCHAR, created_at, agent_id::VARCHAR, plan_id::VARCHAR 
    FROM agent.plan_runs
    WHERE agent_id = %(agent)s
    AND plan_id = %(plan_id)s
    AND 
    (
    (created_at::DATE >= %(start_date)s AND created_at::DATE <=  %(end_date)s)
    OR
    (last_updated::DATE >= %(start_date)s AND last_updated::DATE <=  %(end_date)s)
    )
    ORDER BY created_at asc
    """

    params = {
        "agent": agent,
        "plan_id": plan_id,
        "start_date": start_date,
        "end_date": end_date,
    }
    rows = db.generic_read(sql, params=params)
    return rows


def delete_plan_run(db: Postgres, agent_id: str, plan_run_id: str) -> None:
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
    sql4 = """
        DELETE FROM agent.task_runs
        WHERE agent_id = %(agent_id)s AND plan_run_id = %(plan_run_id)s
    """
    sql5 = """
        DELETE FROM agent.task_run_info
        WHERE agent_id = %(agent_id)s AND plan_run_id = %(plan_run_id)s
    """
    sql6 = """
        DELETE FROM agent.chat_messages
        WHERE agent_id = %(agent_id)s AND plan_run_id = %(plan_run_id)s
        AND is_user_message = FALSE
    """

    db.generic_write(sql1, {"agent_id": agent_id, "plan_run_id": plan_run_id})
    db.generic_write(sql2, {"agent_id": agent_id, "plan_run_id": plan_run_id})
    db.generic_write(sql3, {"agent_id": agent_id, "plan_run_id": plan_run_id})
    db.generic_write(sql4, {"agent_id": agent_id, "plan_run_id": plan_run_id})
    db.generic_write(sql5, {"agent_id": agent_id, "plan_run_id": plan_run_id})
    db.generic_write(sql6, {"agent_id": agent_id, "plan_run_id": plan_run_id})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--agent", type=str)

    parser.add_argument(
        "--start-date",
        type=datetime.date.fromisoformat,
        help="specify a start date for running the agent",
    )
    parser.add_argument(
        "--end-date",
        type=datetime.date.fromisoformat,
        help="specify an end date for running the agent",
    )

    parser.add_argument(
        "--delta",
        type=int,
        default=1,
        help="how many y/m/d/w/h",
    )

    """
    parser.add_argument(
        "--unit",
        type=str,
        default="d",
        help="unit to time delta, years months days weeks hours",
    )
    """

    return parser.parse_args()


async def main() -> None:
    os.environ["FORCE_LOGGING"] = "1"  # enable clickhouse logs
    args = parse_args()

    end_date = args.end_date
    start_date = args.start_date

    days_increment = max(1, args.delta)

    await run_plan_historical_updates(
        agent_id=args.agent,
        start_date=args.start_date,
        end_date=args.end_date,
        days_increment=days_increment,
        skip_commit=False,
    )


async def run_plan_historical_updates(
    agent_id: str,
    skip_commit: bool,
    start_date: Optional[datetime.date] = None,
    end_date: Optional[datetime.date] = None,
    days_increment: int = 1,
) -> None:
    if not start_date:
        start_date = datetime.date.today()

    if not end_date:
        end_date = start_date

    assert start_date <= end_date

    # pick an odd looking time to make it easier to spot
    start_time = datetime.datetime.combine(start_date, datetime.time(hour=7, minute=35))

    set_mock_time(start_time)
    enable_mock_time()

    time_increment = datetime.timedelta(days=days_increment)

    db = get_psql(skip_commit=skip_commit)
    agent_infos = db.get_live_agents_info(agent_ids=[agent_id])
    agent_info = agent_infos[0]
    plan = ExecutionPlan(**agent_info["plan"])
    user_id = agent_info["user_id"]

    chat_contexts = db.get_chat_contexts(agent_ids=[agent_id])
    # we will assume all of the chat context is valid for all runs
    chat_context = chat_contexts[agent_id]

    # cleanup old runs for same time period first
    plan_runs = get_plan_runs(
        db, agent_info["agent_id"], agent_info["plan_id"], start_date, end_date
    )
    print(
        "found",
        len(plan_runs),
        "plan runs for agent: ",
        agent_info["agent_id"],
        "plan:",
        agent_info["plan_id"],
    )
    for r in plan_runs:
        print("deleting:", r)
        delete_plan_run(db, agent_id=agent_info["agent_id"], plan_run_id=r["plan_run_id"])

    print("start/end", start_date, end_date)
    current_date = start_date

    plan_id = agent_info["plan_id"]
    print("==================================")
    print(f"about to run {plan_id=}:", plan)
    print("==================================")

    while current_date <= end_date:
        new_plan_run_id = str(uuid.uuid4())
        print("==================================")
        print("starting:", current_date.isoformat(), "plan_run_id", new_plan_run_id)
        print("==================================")

        context = PlanRunContext(
            agent_id=agent_info["agent_id"],
            plan_id=agent_info["plan_id"],
            user_id=agent_info["user_id"],
            # new plan run id
            plan_run_id=new_plan_run_id,
            chat=chat_context,
            run_tasks_without_prefect=True,
            as_of_date=get_now_utc(),
            skip_db_commit=skip_commit,
        )

        print("about to run context:", context)
        print("==================================")
        print("about to run plan:", plan)
        print("==================================")
        # Tuple[List[IOType], Optional[DefaultDict[str, List[dict]]]]
        try:
            run_start = time.time()
            plan_run_outputs = await run_execution_plan(
                plan=plan,  # ExecutionPlan,
                context=context,  #: PlanRunContext,
                do_chat=True,
                log_all_outputs=False,
                replan_execution_error=False,
                run_plan_in_prefect_immediately=False,
                # This is meant for testing, basically we can fill in the lookup table to
                # make sure we only run the plan starting from a certain point while passing
                # in precomputed outputs for prior tasks.
                override_task_output_lookup=None,  # : Optional[Dict[str, IOType]]
                scheduled_by_automation=True,  # must be true to simulate updates for scheduled automations
                execution_log=None,  #: Optional[DefaultDict[str, List[dict]]] = None,
                # Map task ID's to "replay ID's", which uniquely identify rows in
                # clickhouse's tool_calls table.
                override_task_output_id_lookup=None,  #: Optional[Dict[str, str]] = None,
                # Map task ID's to log_ids, which uniquely identify rows in in the work_logs
                # table. This is a more reliable alternative to the clickhouse version
                # above, but they can be used together, with this map taking precedence over
                # the above map.
                override_task_work_log_id_lookup=None,  #: Optional[Dict[str, str]] = None,
            )
            print(f"Total time to run {current_date=}: {time.time() - run_start}")
        except Exception as e:
            logger.exception(f"Threw exception on  {current_date=}")
            break

        # roll over to the next date/time increment
        increment_mock_time(time_increment)
        current_date = datetime.date.today()

    return


if __name__ == "__main__":
    s = time.time()

    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    init_stdout_logging()
    asyncio.run(main())

    print(f"Total time: {time.time() - s}")
