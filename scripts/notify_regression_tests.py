import argparse
import asyncio
import logging
import time
from typing import Any, Dict, Optional

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
    SUM(CASE WHEN warning_msg <> '' and error_msg = '' THEN 1 ELSE 0 END) as warning_count,
    (toUnixTimestamp64Milli(MAX(execution_finished_at_utc)) -
    toUnixTimestamp64Milli(MIN(execution_plan_started_at_utc))) / 1000 as total_duration
    FROM agent.regression_test rt
    WHERE service_version = %(service_version)s
    """
    results = await ch.generic_read(sql, {"service_version": service_version})
    return results[0]


async def main(agent_service_version: Optional[str] = None) -> None:
    ch_dev = Clickhouse(environment="DEV")
    if not agent_service_version:
        agent_service_version = await get_latest_version(ch_dev)
    results = await get_results_for_latest_version(ch_dev, agent_service_version)
    version_short = agent_service_version.split(":")[-1]
    msg_content = (
        f"Regression Test Result for Version {version_short}:\n"
        f"    Success Count: {results['success_count']}\n"
        f"    Error Count: {results['error_count']}\n"
        f"    Warning Count: {results['warning_count']}\n"
        f"    Took {results['total_duration'] // 60} minutes and {round(results['total_duration'] % 60, 3)} seconds \n"
        f"To see detailed results, visit this link: "
        f"https://agent-dev.boosted.ai/regression-test/{version_short}"
    )
    SlackSender("regression-tests").send_message_at(msg_content, int(time.time()) + 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-a",
        "--agent-service-version",
        default=None,
        type=str,
        required=False,
    )
    args = parser.parse_args()
    asyncio.run(main(agent_service_version=args.agent_service_version))
