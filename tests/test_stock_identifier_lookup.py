import datetime
from unittest import IsolatedAsyncioTestCase

from agent_service.tools.stock_identifier_lookup import (
    StockIdentifierLookupInput,
    stock_identifier_lookup,
)
from agent_service.types import ChatContext, Message, PlanRunContext


class TestStockIdentifierLookup(IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.context = PlanRunContext(
            plan_id="test",
            plan_run_id="test",
            user_id="test",
            user_email="test",
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
        self.args = StockIdentifierLookupInput(stock_str="Apple")
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
