from unittest import IsolatedAsyncioTestCase

from agent_service.tools.feature_identifier_lookup import (
    FeatureIdentifierLookupInput,
    feature_identifier_lookup,
)
from agent_service.types import PlanRunContext


class TestFeatureIdentifierLookup(IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.context = PlanRunContext.get_dummy()

    async def test_feature_identifier_lookup_highprice(self):
        self.args = FeatureIdentifierLookupInput(feature_str="High Price")
        result = await feature_identifier_lookup(self.args, self.context)
        self.assertEqual(result, "spiq_high")

    async def test_feature_identifier_lookup_basiceps(self):
        self.args = FeatureIdentifierLookupInput(feature_str="Basic EPS")
        result = await feature_identifier_lookup(self.args, self.context)
        self.assertEqual(result, "spiq_9")

    async def test_feature_identifier_lookup_bollinger(self):
        self.args = FeatureIdentifierLookupInput(feature_str="Bollinger")
        result = await feature_identifier_lookup(self.args, self.context)
        self.assertEqual(result, "spiq_bb")
