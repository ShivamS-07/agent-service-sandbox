import logging
import unittest
from unittest import IsolatedAsyncioTestCase

from agent_service.io_types.stock import StockID
from agent_service.io_types.table import Table
from agent_service.tools.stocks import (
    GetRiskExposureForStocksInput,
    GetStockUniverseInput,
    StockIdentifierLookupInput,
    get_risk_exposure_for_stocks,
    get_stock_info_for_universe,
    get_stock_universe,
    stock_identifier_lookup,
)
from agent_service.types import PlanRunContext

AAPL = StockID(gbi_id=714, isin="", symbol="AAPL", company_name="Apple")


class TestStockIdentifierLookup(IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        logging.getLogger("asyncio").setLevel(logging.ERROR)

    async def asyncSetUp(self):
        self.context = PlanRunContext.get_dummy()
        # uncomment for easier debugging
        from agent_service.utils.logs import init_test_logging

        init_test_logging()

    async def test_stock_identifier_lookup_ford(self):
        self.args = StockIdentifierLookupInput(stock_name="Ford")
        result = await stock_identifier_lookup(self.args, self.context)
        self.assertEqual(result.gbi_id, 4579)

    async def test_stock_identifier_lookup_brkb(self):
        self.args = StockIdentifierLookupInput(stock_name="BRK/B")
        result = await stock_identifier_lookup(self.args, self.context)
        self.assertEqual(result.gbi_id, 1666)

    async def test_stock_identifier_lookup_hilton(self):
        self.args = StockIdentifierLookupInput(stock_name="Hilton")
        result = await stock_identifier_lookup(self.args, self.context)
        self.assertEqual(result.gbi_id, 30094)

    async def test_stock_identifier_lookup_meta(self):
        self.args = StockIdentifierLookupInput(stock_name="Meta")
        result = await stock_identifier_lookup(self.args, self.context)
        self.assertEqual(result.gbi_id, 26794)

    async def test_stock_identifier_lookup_tesla(self):
        self.args = StockIdentifierLookupInput(stock_name="Tesla Inc")
        result = await stock_identifier_lookup(self.args, self.context)
        self.assertEqual(result.gbi_id, 25508)

    async def test_stock_identifier_lookup_apple(self):
        self.args = StockIdentifierLookupInput(stock_name="AAPL")
        result = await stock_identifier_lookup(self.args, self.context)
        self.assertEqual(result.gbi_id, 714)

    async def test_stock_identifier_lookup_nvidia(self):
        self.args = StockIdentifierLookupInput(stock_name="NVIDIA")
        result = await stock_identifier_lookup(self.args, self.context)
        self.assertEqual(result.gbi_id, 7555)

    async def test_stock_identifier_lookup_netflix(self):
        self.args = StockIdentifierLookupInput(stock_name="Netflix Inc")
        result = await stock_identifier_lookup(self.args, self.context)
        self.assertEqual(result.gbi_id, 7558)

    async def test_stock_identifier_lookup_isin(self):
        self.args = StockIdentifierLookupInput(stock_name="JP3633400001")
        result = await stock_identifier_lookup(self.args, self.context)
        self.assertEqual(result.gbi_id, 389721)  # 7203 toyota japan/jpy

    async def test_stock_identifier_alt_company_names(self):
        self.args = StockIdentifierLookupInput(stock_name="Google")
        result = await stock_identifier_lookup(self.args, self.context)
        # GOOG or GOOGL are both acceptable
        self.assertTrue(result.gbi_id in [30336, 10096])

        self.args = StockIdentifierLookupInput(stock_name="Facebook")
        result = await stock_identifier_lookup(self.args, self.context)
        self.assertEqual(result.gbi_id, 26794)

        self.args = StockIdentifierLookupInput(stock_name="RBC")
        result = await stock_identifier_lookup(self.args, self.context)
        self.assertEqual(result.gbi_id, 12838)

    async def test_stock_identifier_lookup_netflix_similarity(self):
        self.args = StockIdentifierLookupInput(stock_name="Net flix")
        result = await stock_identifier_lookup(self.args, self.context)
        self.assertEqual(result.gbi_id, 7558)

    async def test_stock_identifier_lookup_tesla2(self):
        self.args = StockIdentifierLookupInput(stock_name="Tesla")
        result = await stock_identifier_lookup(self.args, self.context)
        self.assertEqual(result.gbi_id, 25508)

    async def test_stock_identifier_lookup_apple_similarity(self):
        self.args = StockIdentifierLookupInput(stock_name="apple")
        result = await stock_identifier_lookup(self.args, self.context)
        self.assertEqual(result.gbi_id, 714)

    async def test_stock_identifier_lookup_multiple_symbol_matches(self):
        self.args = StockIdentifierLookupInput(stock_name="CAT")
        result = await stock_identifier_lookup(self.args, self.context)
        self.assertEqual(result.gbi_id, 2242)

    @unittest.skip("failing")
    async def test_stock_identifier_lookup_multiple_isin_matches_no_data(self):
        self.args = StockIdentifierLookupInput(stock_name="CA31811L1076")
        result = await stock_identifier_lookup(self.args, self.context)
        self.assertEqual(result.gbi_id, 40888)

    async def test_stock_identifier_lookup_multiple_isin_matches(self):
        self.args = StockIdentifierLookupInput(stock_name="US9773723096")
        result = await stock_identifier_lookup(self.args, self.context)
        # it was hard to find an example of this case, this test failing may not mean much
        self.assertEqual(result.gbi_id, 209866)

    async def test_stock_identifier_lookup_multiple_name_matches(self):
        self.args = StockIdentifierLookupInput(stock_name="toyota")
        result = await stock_identifier_lookup(self.args, self.context)
        self.assertEqual(result.gbi_id, 389721)

    async def test_stock_identifier_lookup_small_company_close_match(self):
        self.args = StockIdentifierLookupInput(stock_name="Toyokumo")
        result = await stock_identifier_lookup(self.args, self.context)
        self.assertEqual(result.gbi_id, 498552)

    async def test_stock_identifier_lookup_bad_spelling(self):
        self.args = StockIdentifierLookupInput(stock_name="Toyotaa")
        result = await stock_identifier_lookup(self.args, self.context)
        self.assertEqual(result.gbi_id, 389721)

    async def test_stock_identifier_lookup_bloomberg(self):
        self.args = StockIdentifierLookupInput(stock_name="IBM US Equity")
        result = await stock_identifier_lookup(self.args, self.context)
        self.assertEqual(result.gbi_id, 5721)

        self.args = StockIdentifierLookupInput(stock_name="AZN LN")
        result = await stock_identifier_lookup(self.args, self.context)
        self.assertEqual(result.gbi_id, 41478)  # astrazeneca

        self.args = StockIdentifierLookupInput(stock_name="6758 JP")
        result = await stock_identifier_lookup(self.args, self.context)
        self.assertEqual(result.gbi_id, 391887)  # sony


class TestStockUniverse(IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.context = PlanRunContext.get_dummy()

        # uncomment for easier debugging
        # from agent_service.utils.logs import init_test_logging
        # init_test_logging()

    async def test_get_stock_info_tsx(self):
        self.args = GetStockUniverseInput(universe_name="TSX")
        result = await get_stock_info_for_universe(self.args, self.context)
        self.assertEqual(result.get("symbol"), "XIU")
        self.assertEqual(result.get("gbi_security_id"), 5056)

    async def test_get_stock_info_nasdaq(self):
        self.args = GetStockUniverseInput(universe_name="Nasdaq")
        result = await get_stock_info_for_universe(self.args, self.context)
        self.assertEqual(result.get("symbol"), "QQQ")

    async def test_get_stock_universe_sp500(self):
        self.args = GetStockUniverseInput(universe_name="S&P 500")
        result = await get_stock_universe(self.args, self.context)
        self.assertEqual(len(result), 503)

    @unittest.skip("failing - stock search flaky")
    async def test_get_stock_universe_tsx(self):
        self.args = GetStockUniverseInput(universe_name="TSX")
        result = await get_stock_universe(self.args, self.context)
        self.assertEqual(len(result), 60)


class TestRiskExposure(IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.context = PlanRunContext.get_dummy()

        # uncomment for easier debugging
        # from agent_service.utils.logs import init_test_logging
        # init_test_logging()

    async def test_get_risk_exposure_for_stocks(self):
        self.args = GetRiskExposureForStocksInput(stock_list=[AAPL])
        result: Table = await get_risk_exposure_for_stocks(self.args, self.context)
        # make sure we got the table for the  right STOCK
        self.assertEqual(result.to_df().values[0][0], AAPL)
        # Ensure the data in the table is the right type
        for value in result.to_df().values[0][1:]:  # skip the first entry because it's the stock
            self.assertTrue(isinstance(value, float))
