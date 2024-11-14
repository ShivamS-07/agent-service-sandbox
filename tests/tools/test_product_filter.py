import unittest
from unittest import IsolatedAsyncioTestCase

from agent_service.io_types.stock import StockID
from agent_service.tools.product_filter import (
    FilterStocksByProductOrServiceInput,
    filter_stocks_by_product_or_service,
)
from agent_service.tools.stocks import GetStockUniverseInput, get_stock_universe
from agent_service.types import PlanRunContext

AAPL = StockID(gbi_id=714, isin="", symbol="AAPL", company_name="Apple")
GOOG = StockID(gbi_id=30336, isin="", symbol="GOOG", company_name="Alphabet Inc.")
ERGB = StockID(
    gbi_id=434782, isin="", symbol="ERGB", company_name="ErgoBilt, Inc."
)  # ergonomic chairs
TRQ = StockID(
    gbi_id=19694, isin="", symbol="TRQ", company_name="Turquoise Hill Resources Ltd."
)  # mining company
TSLA = StockID(gbi_id=25508, isin="", symbol="TSLA", company_name="Tesla Inc.")
BWA = StockID(gbi_id=1070, isin="", symbol="BWA", company_name="BorgWarner Inc.")
ALB = StockID(gbi_id=85, isin="", symbol="ALB", company_name="Albemarle Corporation")
NVDA = StockID(gbi_id=7555, isin="", symbol="NVDA", company_name="NVIDIA Corporation")


class TestFilterStocksByProductOrService(IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.context = PlanRunContext.get_dummy()
        self.stock_ids = [AAPL, ERGB, TRQ]
        self.sp500_stock_ids = await get_stock_universe(
            GetStockUniverseInput(universe_name="SP500"), self.context
        )

    @unittest.skip("slow test")
    async def test_filter_by_product1(self):
        result = await filter_stocks_by_product_or_service(
            FilterStocksByProductOrServiceInput(
                stock_ids=self.stock_ids, product_str="smartphone", texts=[], filter_only=True
            ),
            self.context,
        )
        self.assertEqual(result, [AAPL])

    async def test_filter_by_product2(self):
        result = await filter_stocks_by_product_or_service(
            FilterStocksByProductOrServiceInput(
                stock_ids=self.stock_ids,
                product_str="gold and copper mine",
                texts=[],
                filter_only=True,
            ),
            self.context,
        )
        self.assertEqual(result, [TRQ])

    @unittest.skip("slow test")
    async def test_filter_by_product3(self):
        result = await filter_stocks_by_product_or_service(
            FilterStocksByProductOrServiceInput(
                stock_ids=self.stock_ids, product_str="chairs", texts=[], filter_only=True
            ),
            self.context,
        )
        self.assertEqual(result, [ERGB])

    @unittest.skip("slow test")
    async def test_filter_by_product4(self):
        products = [
            # ("Electric Vehicles", ["Tesla"]),
            ("AI Chips", [NVDA]),
            # ("Solar Power", [TSLA]),
            # ("cloud infrastructure", [GOOG]),
            # ("online retailing and marketspace", []),
            # ("Video Gaming", []),
        ]
        for product, companies in products:
            print("====================================")
            print("product: ", product)
            print("====================================")
            result = await filter_stocks_by_product_or_service(
                FilterStocksByProductOrServiceInput(
                    stock_ids=self.sp500_stock_ids,
                    product_str=product,
                    texts=[],
                    must_include_stocks=companies if companies else None,
                    filter_only=True,
                ),
                self.context,
            )
            print([stock.company_name for stock in result])
            for _ in range(2):
                new_result = await filter_stocks_by_product_or_service(
                    FilterStocksByProductOrServiceInput(
                        stock_ids=self.sp500_stock_ids,
                        product_str=product,
                        must_include_stocks=companies,
                        filter_only=True,
                    ),
                    self.context,
                )

                if result != new_result:
                    print([stock.company_name for stock in new_result])
                    print([stock.company_name for stock in result])
                # assert the difference be less that 20% + 1
                self.assertLess(len(set(result) - set(new_result)), 3)
                self.assertLess(len(set(new_result) - set(result)), 3)

    @unittest.skip("slow test")
    async def test_filter_by_product5(self):
        new_result = await filter_stocks_by_product_or_service(
            FilterStocksByProductOrServiceInput(
                stock_ids=self.sp500_stock_ids, product_str="Solar Power", texts=[]
            ),
            self.context,
        )
        print([stock.company_name for stock in new_result])
        self.assertIn(TSLA, new_result)

    @unittest.skip("slow test")
    async def test_filter_by_product6(self):
        new_result = await filter_stocks_by_product_or_service(
            FilterStocksByProductOrServiceInput(
                stock_ids=self.sp500_stock_ids, product_str="AI Chips", texts=[]
            ),
            self.context,
        )

        print([stock.company_name for stock in new_result])
        self.assertNotIn(AAPL, new_result)
        self.assertNotIn(GOOG, new_result)
        self.assertIn(NVDA, new_result)
