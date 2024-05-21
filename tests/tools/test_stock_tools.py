from unittest import IsolatedAsyncioTestCase

from agent_service.tools.stocks import (
    GetStockUniverseInput,
    StockIdentifierLookupInput,
    StockIDsToTickerInput,
    convert_stock_identifiers_to_tickers,
    get_stock_universe,
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

    async def test_stock_identifier_lookup_isin(self):
        self.args = StockIdentifierLookupInput(stock_name="JP3633400001")
        result = await stock_identifier_lookup(self.args, self.context)
        self.assertEqual(result, 389721)  # 7203 toyota japan/jpy

    async def test_stock_identifier_lookup_netflix_similarity(self):
        self.args = StockIdentifierLookupInput(stock_name="Net flix")
        result = await stock_identifier_lookup(self.args, self.context)
        self.assertEqual(result, 7558)

    async def test_stock_identifier_lookup_apple_similarity(self):
        self.args = StockIdentifierLookupInput(stock_name="apple")
        result = await stock_identifier_lookup(self.args, self.context)
        self.assertEqual(result, 714)

    async def test_stock_identifier_lookup_multiple_symbol_matches(self):
        self.args = StockIdentifierLookupInput(stock_name="CAT")
        result = await stock_identifier_lookup(self.args, self.context)
        self.assertEqual(result, 2242)

    async def test_stock_identifier_lookup_multiple_isin_matches_no_data(self):
        self.args = StockIdentifierLookupInput(stock_name="CA31811L1076")
        result = await stock_identifier_lookup(self.args, self.context)
        self.assertEqual(result, 40888)

    async def test_stock_identifier_lookup_multiple_isin_matches(self):
        self.args = StockIdentifierLookupInput(stock_name="US9773723096")
        result = await stock_identifier_lookup(self.args, self.context)
        # it was hard to find an example of this case, this test failing may not mean much
        self.assertEqual(result, 209866)

    async def test_stock_identifier_lookup_multiple_name_matches(self):
        self.args = StockIdentifierLookupInput(stock_name="toyota")
        result = await stock_identifier_lookup(self.args, self.context)
        self.assertEqual(result, 389721)

    async def test_stock_identifier_lookup_small_company_close_match(self):
        self.args = StockIdentifierLookupInput(stock_name="Toyokumo")
        result = await stock_identifier_lookup(self.args, self.context)
        self.assertEqual(result, 498552)

    async def test_stock_identifier_lookup_small_company_close_match2(self):
        self.args = StockIdentifierLookupInput(stock_name="Toyoda")
        result = await stock_identifier_lookup(self.args, self.context)
        self.assertEqual(result, 389646)

    async def test_stock_identifier_lookup_bad_spelling(self):
        self.args = StockIdentifierLookupInput(stock_name="Toyotaa")
        result = await stock_identifier_lookup(self.args, self.context)
        self.assertEqual(result, 389721)

    async def test_convert_stock_identifiers_to_tickers(self):
        result = await convert_stock_identifiers_to_tickers(
            args=StockIDsToTickerInput(stock_ids=[714, 715, 716]),
            context=PlanRunContext.get_dummy(),
        )
        self.assertEqual(result, ["AAPL", "APOG", "ANSS"])


class TestStockUniverse(IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.context = PlanRunContext.get_dummy()

    async def test_get_stock_universe_sp500(self):
        self.args = GetStockUniverseInput(universe_name="S&P 500")
        result = await get_stock_universe(self.args, self.context)
        self.assertEqual(len(result), 503)

    async def test_get_stock_universe_tsx(self):
        self.args = GetStockUniverseInput(universe_name="TSX")
        result = await get_stock_universe(self.args, self.context)
        self.assertEqual(len(result), 60)
