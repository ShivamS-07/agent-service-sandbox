import datetime
from unittest import IsolatedAsyncioTestCase

from agent_service.tools.stock_identifier_lookup import (
    StockIdentifierLookupInput,
    StockIDsToTickerInput,
    convert_stock_identifiers_to_tickers,
    stock_identifier_lookup,
)
from agent_service.types import ChatContext, Message, PlanRunContext


class TestStockIdentifierLookup(IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.context = PlanRunContext(
            plan_id="test",
            plan_run_id="test",
            user_id="test",
            task_id="test",
            agent_id="test",
            chat=ChatContext(
                messages=[Message(content="test", is_user=True, timestamp=datetime.datetime.now())]
            ),
        )

    async def test_stock_identifier_lookup_meta(self):
        self.args = StockIdentifierLookupInput(stock_str="Meta")
        result = await stock_identifier_lookup(self.args, self.context)
        self.assertEqual(result, 26794)

    async def test_stock_identifier_lookup_tesla(self):
        self.args = StockIdentifierLookupInput(stock_str="Tesla Inc")
        result = await stock_identifier_lookup(self.args, self.context)
        self.assertEqual(result, 25508)

    async def test_stock_identifier_lookup_apple(self):
        self.args = StockIdentifierLookupInput(stock_str="AAPL")
        result = await stock_identifier_lookup(self.args, self.context)
        self.assertEqual(result, 714)

    async def test_stock_identifier_lookup_nvidia(self):
        self.args = StockIdentifierLookupInput(stock_str="NVIDIA")
        result = await stock_identifier_lookup(self.args, self.context)
        self.assertEqual(result, 7555)

    async def test_stock_identifier_lookup_netflix(self):
        self.args = StockIdentifierLookupInput(stock_str="Netflix Inc")
        result = await stock_identifier_lookup(self.args, self.context)
        self.assertEqual(result, 7558)

    async def test_convert_stock_identifiers_to_tickers(self):
        result = await convert_stock_identifiers_to_tickers(
            args=StockIDsToTickerInput(stock_ids=[714, 715, 716]),
            context=PlanRunContext.get_dummy(),
        )
        self.assertEqual(result, ["AAPL", "APOG", "ANSS"])
