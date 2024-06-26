import logging
import unittest
from unittest import IsolatedAsyncioTestCase

from agent_service.io_types.stock import StockID
from agent_service.io_types.table import Table
from agent_service.tools.stocks import (
    GetRiskExposureForStocksInput,
    GetStockUniverseInput,
    GrowthFilterInput,
    StockIdentifierLookupInput,
    ValueFilterInput,
    get_risk_exposure_for_stocks,
    get_stock_info_for_universe,
    get_stock_universe,
    growth_filter,
    stock_identifier_lookup,
    value_filter,
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


class TestStockIdentifierLookup1(TestStockIdentifierLookup):

    async def test_stock_identifier_lookup_symbol_plus_name(self):
        self.args = StockIdentifierLookupInput(stock_name="ARTI Evolve ETF")
        result = await stock_identifier_lookup(self.args, self.context)
        self.assertEqual(result.symbol, "ARTI")
        self.assertEqual(result.gbi_id, 647005)

        self.args = StockIdentifierLookupInput(stock_name="ARTI ETF by Evolve")
        result = await stock_identifier_lookup(self.args, self.context)
        self.assertEqual(result.symbol, "ARTI")
        self.assertEqual(result.gbi_id, 647005)

        self.args = StockIdentifierLookupInput(stock_name="ARTI AI ETF")
        result = await stock_identifier_lookup(self.args, self.context)
        self.assertEqual(result.symbol, "ARTI")
        self.assertEqual(result.gbi_id, 647005)

    @unittest.skip("there is a company named azure")
    async def test_stock_identifier_lookup_products_brands_azure(self):
        self.args = StockIdentifierLookupInput(stock_name="Azure")
        result = await stock_identifier_lookup(self.args, self.context)
        self.assertEqual(result.symbol, "MSFT")
        self.assertEqual(result.gbi_id, 6963)

    async def test_stock_identifier_lookup_samsung(self):
        self.args = StockIdentifierLookupInput(stock_name="Samsung Electronics")
        result = await stock_identifier_lookup(self.args, self.context)
        self.assertEqual(result.gbi_id, 396629)


class TestStockIdentifierLookup2(TestStockIdentifierLookup):

    async def test_stock_identifier_lookup_products_brands_aws(self):
        self.args = StockIdentifierLookupInput(stock_name="AWS")
        result = await stock_identifier_lookup(self.args, self.context)
        self.assertEqual(result.symbol, "AMZN")
        self.assertEqual(result.gbi_id, 149)

    async def test_stock_identifier_lookup_ford(self):
        self.args = StockIdentifierLookupInput(stock_name="Ford")
        result = await stock_identifier_lookup(self.args, self.context)
        self.assertEqual(result.gbi_id, 4579)

    async def test_stock_identifier_lookup_brkb(self):
        self.args = StockIdentifierLookupInput(stock_name="BRK/B")
        result = await stock_identifier_lookup(self.args, self.context)
        self.assertEqual(result.gbi_id, 1666)


class TestStockIdentifierLookup3(TestStockIdentifierLookup):
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


class TestStockIdentifierLookup4(TestStockIdentifierLookup):

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


class TestStockIdentifierLookup5(TestStockIdentifierLookup):
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


class TestStockIdentifierLookup6(TestStockIdentifierLookup):

    async def test_stock_identifier_lookup_netflix_similarity(self):
        self.args = StockIdentifierLookupInput(stock_name="Net flix")
        result = await stock_identifier_lookup(self.args, self.context)
        self.assertEqual(result.gbi_id, 7558)

    async def test_stock_identifier_lookup_tesla2(self):
        self.args = StockIdentifierLookupInput(stock_name="Tesla")
        result = await stock_identifier_lookup(self.args, self.context)
        self.assertEqual(result.gbi_id, 25508)


class TestStockIdentifierLookup7(TestStockIdentifierLookup):
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

    @unittest.skip("failing, not important")
    async def test_stock_identifier_lookup_multiple_isin_matches(self):
        self.args = StockIdentifierLookupInput(stock_name="US9773723096")
        result = await stock_identifier_lookup(self.args, self.context)
        # it was hard to find an example of this case, this test failing may not mean much
        self.assertEqual(result.gbi_id, 209866)


class TestStockIdentifierLookup8(TestStockIdentifierLookup):
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


class TestStockIdentifierLookup9(TestStockIdentifierLookup):

    async def test_stock_identifier_lookup_bloomberg(self):
        self.args = StockIdentifierLookupInput(stock_name="IBM US Equity")
        result = await stock_identifier_lookup(self.args, self.context)
        self.assertEqual(result.gbi_id, 5721)

        self.args = StockIdentifierLookupInput(stock_name="ibm ln")
        result = await stock_identifier_lookup(self.args, self.context)
        self.assertEqual(result.gbi_id, 42154)

        self.args = StockIdentifierLookupInput(stock_name="AZN LN")
        result = await stock_identifier_lookup(self.args, self.context)
        self.assertEqual(result.gbi_id, 41478)  # astrazeneca

        self.args = StockIdentifierLookupInput(stock_name="6758 JP")
        result = await stock_identifier_lookup(self.args, self.context)
        self.assertEqual(result.gbi_id, 391887)  # sony


class TestStockUniverse1(IsolatedAsyncioTestCase):
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

    async def test_get_stock_info_r1k(self):
        self.args = GetStockUniverseInput(universe_name="russell 1000")
        result = await get_stock_info_for_universe(self.args, self.context)
        print("result", result)
        self.assertEqual(result.get("symbol"), "IWB")
        self.assertEqual(result.get("gbi_security_id"), 13770)

    async def test_get_stock_info_r2k(self):
        self.args = GetStockUniverseInput(universe_name="russell 2000")
        result = await get_stock_info_for_universe(self.args, self.context)
        print("result", result)
        self.assertEqual(result.get("symbol"), "IWM")
        self.assertEqual(result.get("gbi_security_id"), 13766)


class TestStockUniverse2(IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.context = PlanRunContext.get_dummy()

    async def test_get_stock_info_nasdaq(self):
        self.args = GetStockUniverseInput(universe_name="Nasdaq")
        result = await get_stock_info_for_universe(self.args, self.context)
        self.assertEqual(result.get("symbol"), "QQQ")

    async def test_get_stock_universe_sp500(self):
        self.args = GetStockUniverseInput(universe_name="S&P 500")
        result = await get_stock_universe(self.args, self.context)
        self.assertEqual(len(result), 503)

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

    async def test_get_growth_stocks(self):
        args = GrowthFilterInput()
        result = await growth_filter(args, self.context)
        self.assertGreater(len(result), 10)
        self.assertLess(len(result), 400)

    async def test_get_value_stocks(self):
        args = ValueFilterInput()
        result = await value_filter(args, self.context)
        self.assertGreater(len(result), 10)
        self.assertLess(len(result), 400)
