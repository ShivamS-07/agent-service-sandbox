import asyncio
import logging
import time
from typing import Any, Dict

from agent_service.slack.slack_sender import SlackSender
from agent_service.utils.clickhouse import Clickhouse

logger = logging.getLogger(__name__)


async def get_latest_version(ch: Clickhouse) -> str:
    sql = """
    SELECT service_version, max(timestamp) as completed_time from agent.regression_test rt
    where service_version ilike '%0.0%' and `timestamp`  < toStartOfDay(today())
    GROUP BY service_version order by completed_time desc LIMIT 1
    """
    return (await ch.generic_read(sql))[0]["service_version"]


async def get_results_for_latest_version(ch: Clickhouse, service_version: str) -> Dict[str, Any]:
    sql = """
    SELECT SUM(CASE WHEN error_msg <> '' THEN 1 ELSE 0 END) as error_count,
    SUM(CASE WHEN error_msg = '' and warning_msg = '' THEN 1 ELSE 0 END) as success_count,
    SUM(CASE WHEN warning_msg <> '' and error_msg = '' THEN 1 ELSE 0 END) as warning_count
    FROM agent.regression_test rt
    WHERE service_version = %(service_version)s
    """
    results = await ch.generic_read(sql, {"service_version": service_version})
    return results[0]


async def main() -> None:
    ch_dev = Clickhouse(environment="DEV")
    latest_version = await get_latest_version(ch_dev)
    results = await get_results_for_latest_version(ch_dev, latest_version)
    version_short = latest_version.split(":")[-1]
    msg_content = (
        f"Regression Test Result for Latest Version {version_short}:\n"
        f"    Success Count: {results['success_count']}\n"
        f"    Error Count: {results['error_count']}\n"
        f"    Warning Count: {results['warning_count']}\n"
        f"To see detailed results, visit this link: "
        f"https://agent-dev.boosted.ai/regression-test/{version_short}"
    )

    SlackSender("regression-tests").send_message_at(msg_content, int(time.time()) + 60)


if __name__ == "__main__":
    asyncio.run(main())
