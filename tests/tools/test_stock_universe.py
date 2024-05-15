from unittest import IsolatedAsyncioTestCase

from agent_service.tools.stock_universe import GetStockUniverseInput, get_stock_universe
from agent_service.types import PlanRunContext


class TestStockUniverse(IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.context = PlanRunContext.get_dummy()

    async def test_get_stock_universe_sp500(self):
        self.args = GetStockUniverseInput(universe_name="S&P 500")
        result = await get_stock_universe(self.args, self.context)
        self.assertEqual(len(result), 503)