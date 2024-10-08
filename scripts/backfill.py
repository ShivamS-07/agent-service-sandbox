import asyncio
import logging
import typing
from collections import defaultdict
from typing import Optional

from agent_service.endpoints.models import GetAgentOutputResponse
from agent_service.utils.async_db import AsyncDB
from agent_service.utils.async_postgres_base import AsyncPostgresBase
from agent_service.utils.async_utils import gather_with_concurrency
from agent_service.utils.cache_utils import CacheBackend, RedisCacheBackend
from agent_service.utils.get_agent_outputs import get_agent_output
from agent_service.utils.logs import init_stdout_logging

LOGGER = logging.getLogger(__name__)


async def wrap_get_agent_output(
    pg: AsyncDB, cache: CacheBackend, agent_id: str, plan_run_id: Optional[str] = None
) -> None:
    try:
        await get_agent_output(pg=pg, cache=cache, agent_id=agent_id, plan_run_id=plan_run_id)
    except Exception as e:
        LOGGER.error(e)
        LOGGER.error(f"error in getting outputs for {agent_id=} and {plan_run_id=}")


async def backfill_cache() -> None:
    pg = AsyncDB(pg=AsyncPostgresBase())
    cache = RedisCacheBackend(
        namespace="agent-output-cache",
        serialize_func=lambda x: x.model_dump_json(serialize_as_any=True),
        deserialize_func=lambda x: GetAgentOutputResponse.model_validate_json(x),
    )
    agents = """
    select agent_id::VARCHAR from agent.agents where created_at >= DATE '2024-09-01'
    """
    agent_ids = [row["agent_id"] for row in (await pg.pg.generic_read(agents))]
    latest_plan_runs = """
    SELECT DISTINCT ON (agent_id)
    agent_id::VARCHAR,
    plan_run_id::VARCHAR,
    created_at AS max_created_at
    FROM agent.agent_outputs ao
    WHERE agent_id = ANY(%(agent_ids)s)
    ORDER BY agent_id, created_at DESC
    """

    rows = await pg.pg.generic_read(latest_plan_runs, params={"agent_ids": agent_ids})
    d: typing.DefaultDict = defaultdict(list)
    for row in rows:
        if len(d[row["agent_id"]]) < 10:
            d[row["agent_id"]].append(row["plan_run_id"])
    LOGGER.info("processing")
    tasks = [
        wrap_get_agent_output(pg=pg, cache=cache, agent_id=agent_id, plan_run_id=plan_run_id)
        for agent_id, plan_run_ids in d.items()
        for plan_run_id in plan_run_ids
    ]
    await gather_with_concurrency(tasks, n=4)
    LOGGER.info("completed")
    await pg.pg.close()  # type: ignore


if __name__ == "__main__":
    init_stdout_logging()
    asyncio.run(backfill_cache())
