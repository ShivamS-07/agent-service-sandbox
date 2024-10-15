import asyncio
import logging

from agent_service.utils.clickhouse import Clickhouse

logger = logging.getLogger(__name__)


async def main() -> None:
    ch = Clickhouse()
    sql = """
    ALTER TABLE events DELETE WHERE timestamp < now() - INTERVAL 1 HOUR
    """
    await (await ch.get_or_create_client()).query(sql)
    logger.info("Deleting rows in events table")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s.%(funcName)s:%(lineno)d - %(levelname)s - %(message)s",
        force=True,
    )
    asyncio.run(main())
