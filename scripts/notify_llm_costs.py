import asyncio
import logging
import time
from datetime import datetime, timedelta

from agent_service.slack.slack_sender import SlackSender
from agent_service.utils.clickhouse import Clickhouse

logger = logging.getLogger(__name__)


async def main() -> None:
    ch_dev = Clickhouse(environment="DEV")
    ch_prod = Clickhouse(environment="ALPHA")

    cost_sql = """
    SELECT round(sum(cost_usd), 2) as total_cost_usd
    FROM llm.queries
    WHERE timestamp >= toStartOfDay(yesterday())
    AND timestamp < toStartOfDay(today())
    """

    dev_cost_yesterday = (await ch_dev.generic_read(sql=cost_sql))[0]["total_cost_usd"]
    prod_cost_yesterday = (await ch_prod.generic_read(sql=cost_sql))[0]["total_cost_usd"]

    total_cost_yesterday = dev_cost_yesterday + prod_cost_yesterday
    date_yesterday = (datetime.today() - timedelta(days=1)).strftime("%Y-%m-%d")
    msg_content = (
        f"Total cost for {date_yesterday} was ${total_cost_yesterday}:\n"
        f"    Dev cost: ${dev_cost_yesterday}\n"
        f"    Prod cost: ${prod_cost_yesterday}\n"
        f"For detailed breakdown of cost, visit this link:\n"
        f"http://grafana.boosted.ai/d/edwbon46t3hfkc/cost-monitoring?from=now-1d%2Fd&to=now-1d%2Fd&orgId=1&tab=query"
    )

    SlackSender("llm-costs").send_message(msg_content, int(time.time()) + 60)


if __name__ == "__main__":
    asyncio.run(main())
