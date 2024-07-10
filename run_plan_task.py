import argparse
import asyncio
from typing import Tuple

from gbi_common_py_utils.utils.clickhouse_base import ClickhouseBase

from agent_service.tool import ToolArgs, ToolRegistry
from agent_service.tools import *  # noqa
from agent_service.types import PlanRunContext
from agent_service.utils.logs import init_stdout_logging


def fetch_args_from_clickhouse(
    plan_run_id: str, tool_name: str, task_id: str, env: str
) -> Tuple[ToolArgs, PlanRunContext, str]:
    c = ClickhouseBase(environment=env)
    sql = """
    SELECT agent_id, user_id, tool_name, task_id, plan_id, plan_run_id, args, context
    FROM agent.tool_calls
    WHERE plan_run_id = %(plan_run_id)s
    AND (tool_name = %(tool_name)s OR task_id = %(task_id)s)
    """

    params = {"plan_run_id": plan_run_id, "tool_name": tool_name, "task_id": task_id}
    result = c.generic_read(sql=sql, params=params)
    if not result:
        raise RuntimeError("No args found!")
    row = result[0]
    tool = ToolRegistry.get_tool(row["tool_name"])

    return (
        tool.input_type.model_validate_json(row["args"]),
        (
            PlanRunContext.model_validate_json(row["context"])
            if row["context"]
            else PlanRunContext(
                agent_id=row["agent_id"],
                plan_id=row["plan_id"],
                plan_run_id=row["plan_run_id"],
                user_id=row["user_id"],
                task_id=row["task_id"],
                run_tasks_without_prefect=True,
            )
        ),
        row["tool_name"],
    )


def fetch_args_from_clickhouse_replay_id(
    env: str, replay_id: str
) -> Tuple[ToolArgs, PlanRunContext, str]:
    c = ClickhouseBase(environment=env)
    sql = """
    SELECT agent_id, user_id, tool_name, task_id, plan_id, plan_run_id, args, context
    FROM agent.tool_calls
    WHERE replay_id = %(replay_id)s
    """

    params = {"replay_id": replay_id}
    result = c.generic_read(sql=sql, params=params)
    if not result:
        raise RuntimeError("No args found!")
    row = result[0]
    tool = ToolRegistry.get_tool(row["tool_name"])

    return (
        tool.input_type.model_validate_json(row["args"]),
        (
            PlanRunContext.model_validate_json(row["context"])
            if row["context"]
            else PlanRunContext(
                agent_id=row["agent_id"],
                plan_id=row["plan_id"],
                plan_run_id=row["plan_run_id"],
                user_id=row["user_id"],
                task_id=row["task_id"],
                run_tasks_without_prefect=True,
            )
        ),
        row["tool_name"],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--plan-run-id", type=str)
    parser.add_argument(
        "-t",
        "--tool-name",
        type=str,
        help="Will run for the FIRST instance of the tool. Otherwise use task-id.",
    )
    parser.add_argument("-i", "--task-id", type=str)
    parser.add_argument("-r", "--replay-id", type=str)
    parser.add_argument("-e", "--env", type=str, default="DEV")
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    init_stdout_logging()
    print("Fetching args from clickhouse...")
    if args.replay_id:
        tool_args, context, tool_name = fetch_args_from_clickhouse_replay_id(
            replay_id=args.replay_id,
            env=args.env,
        )
    else:
        tool_args, context, tool_name = fetch_args_from_clickhouse(
            env=args.env,
            plan_run_id=args.plan_run_id,
            task_id=args.task_id,
            tool_name=args.tool_name,
        )
    # Don't connect to prefect
    context.run_tasks_without_prefect = True
    print("Fetched args, running tool...\n--------------------")
    tool = ToolRegistry.get_tool(tool_name)
    result = await tool.func(args=tool_args, context=context)
    print("--------------------\nGot Result:\n")
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
