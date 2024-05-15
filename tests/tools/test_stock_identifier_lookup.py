from unittest import IsolatedAsyncioTestCase

from agent_service.tools.stock_identifier_lookup import (
    StockIdentifierLookupInput,
    StockIDsToTickerInput,
    convert_stock_identifiers_to_tickers,
    stock_identifier_lookup,
)
from agent_service.types import PlanRunContext


class TestStockIdentifierLookup(IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.context = PlanRunContext.get_dummy()

    async def test_stock_identifier_lookup_meta(self):
        self.args = StockIdentifierLookupInput(stock_name="Meta")
        result = await stock_identifier_lookup(self.args, self.context)
        self.assertEqual(result, 26794)

    async def test_stock_identifier_lookup_tesla(self):
        self.args = StockIdentifierLookupInput(stock_name="Tesla Inc")
        result = await stock_identifier_lookup(self.args, self.context)
        self.assertEqual(result, 25508)

    async def test_stock_identifier_lookup_apple(self):
        self.args = StockIdentifierLookupInput(stock_name="AAPL")
        result = await stock_identifier_lookup(self.args, self.context)
        self.assertEqual(result, 714)

    async def test_stock_identifier_lookup_nvidia(self):
        self.args = StockIdentifierLookupInput(stock_name="NVIDIA")
        result = await stock_identifier_lookup(self.args, self.context)
        self.assertEqual(result, 7555)

    async def test_stock_identifier_lookup_netflix(self):
        self.args = StockIdentifierLookupInput(stock_name="Netflix Inc")
        result = await stock_identifier_lookup(self.args, self.context)
        self.assertEqual(result, 7558)

    async def test_convert_stock_identifiers_to_tickers(self):
        result = await convert_stock_identifiers_to_tickers(
            args=StockIDsToTickerInput(stock_ids=[714, 715, 716]),
            context=PlanRunContext.get_dummy(),
        )
        self.assertEqual(result, ["AAPL", "APOG", "ANSS"])
