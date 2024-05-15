from unittest import IsolatedAsyncioTestCase

from agent_service.tools.statistic_identifier_lookup import (
    StatisticsIdentifierLookupInput,
    statistic_identifier_lookup,
)
from agent_service.types import PlanRunContext


class TestStatisticsIdentifierLookup(IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.context = PlanRunContext.get_dummy()

    async def test_statistic_identifier_lookup_highprice(self):
        self.args = StatisticsIdentifierLookupInput(statistic_name="High Price")
        result = await statistic_identifier_lookup(self.args, self.context)
        self.assertEqual(result, "spiq_high")

    async def test_statistic_identifier_lookup_basiceps(self):
        self.args = StatisticsIdentifierLookupInput(statistic_name="Basic EPS")
        result = await statistic_identifier_lookup(self.args, self.context)
        self.assertEqual(result, "spiq_9")

    async def test_statistic_identifier_lookup_bollinger(self):
        self.args = StatisticsIdentifierLookupInput(statistic_name="Bid Price")
        result = await statistic_identifier_lookup(self.args, self.context)
        self.assertEqual(result, "spiq_bid")
