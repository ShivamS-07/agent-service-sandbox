"""
This script is to warm up the stock search cache in WebServer and SecurityMetadataService
Currently, the first load is extremely slow and easy to timeout, so the idea is to call the endpoint
to warm up the cache before the first user request
"""

import asyncio
import logging
import time

from agent_service.external.webserver import get_ordered_securities
from agent_service.utils.logs import init_stdout_logging
from agent_service.utils.sentry_utils import init_sentry

logger = logging.getLogger(__name__)


async def main() -> None:
    """
    The cache key is (includeDepositary, includeForeign)
    """

    # Warm up the cache for the params used in Boosted1 top search bar
    count = 1
    while count < 5:
        logger.info(f"Warming up the stock search cache (True/True). Attempt {count}")

        try:
            start = time.perf_counter()
            await get_ordered_securities(
                user_id="2ac3b860-faa6-4ffe-8588-875837b5ce6b",  # Cypress2 Admin User
                searchText="NVDA",
                preferEtfs=False,
                includeDepositary=True,
                includeForeign=True,
                order=["volume"],
                maxItems=10,
            )
            end = time.perf_counter()
            if end - start < 10:
                logger.info(f"Completed warming up the cache. Time taken: {end - start}")
                break
        except Exception as e:
            logger.info(f"Error in warm up stock search: {e}")

        count += 1
        time.sleep(5)

    # Warm up the cache for the params used in other places
    count = 1
    while count < 5:
        logger.info(f"Warming up the stock search cache (False/False). Attempt {count}")

        try:
            start = time.perf_counter()
            await get_ordered_securities(
                user_id="2ac3b860-faa6-4ffe-8588-875837b5ce6b",  # Cypress2 Admin User
                searchText="TSLA",
                preferEtfs=False,
                includeDepositary=False,
                includeForeign=False,
                order=["country", "market_cap"],
                priorityCountry="USA",
                maxItems=10,
            )
            end = time.perf_counter()
            if end - start < 10:
                logger.info(f"Completed warming up the cache. Time taken: {end - start}")
                break
        except Exception as e:
            logger.info(f"Error in warm up stock search: {e}")

        count += 1
        time.sleep(5)


if __name__ == "__main__":
    init_stdout_logging()
    init_sentry()
    asyncio.run(main())
