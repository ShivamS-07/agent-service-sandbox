import unittest

from agent_service.GPT.requests import set_use_global_stub
from agent_service.io_types.stock import StockID
from agent_service.tools.regions import FilterStockRegionInput, filter_stocks_by_region
from agent_service.types import PlanRunContext

MSFT = StockID(gbi_id=6963, isin="", symbol="MSFT", company_name="Microsoft")
FORD = StockID(gbi_id=4579, isin="", symbol="F", company_name="Ford")
RBC = StockID(gbi_id=12838, isin="", symbol="RY", company_name="RBC")


class TestRegions(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        set_use_global_stub(False)
        self.context = PlanRunContext.get_dummy()

        # uncomment for easier debugging
        from agent_service.utils.logs import init_test_logging

        init_test_logging()

    async def test_region_filter(self):
        stock_ids = [MSFT, FORD, RBC]
        args = FilterStockRegionInput(region_name="CAN", stock_ids=stock_ids)

        stocks = await filter_stocks_by_region(args=args, context=self.context)
        self.assertEqual(1, len(stocks))
        self.assertEqual(stocks[0].gbi_id, RBC.gbi_id)

        args = FilterStockRegionInput(region_name="USA", stock_ids=stock_ids)
        stocks = await filter_stocks_by_region(args=args, context=self.context)

        self.assertEqual(2, len(stocks))
        self.assertNotEqual(stocks[0].gbi_id, RBC.gbi_id)
        self.assertNotEqual(stocks[1].gbi_id, RBC.gbi_id)
