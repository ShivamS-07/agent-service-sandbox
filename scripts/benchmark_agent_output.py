import argparse
import asyncio

from agent_service.utils.async_db import AsyncDB
from agent_service.utils.async_postgres_base import AsyncPostgresBase
from agent_service.utils.logs import init_stdout_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--agent-id", type=str, required=True)
    parser.add_argument("-p", "--plan-run-id", type=str)
    return parser.parse_args()


async def main() -> None:
    init_stdout_logging()
    args = parse_args()
    db = AsyncDB(pg=AsyncPostgresBase())
    await db.get_agent_outputs(agent_id=args.agent_id, plan_run_id=args.plan_run_id)


asyncio.run(main())
