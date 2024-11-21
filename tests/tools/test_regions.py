import unittest

from agent_service.GPT.requests import set_use_global_stub
from agent_service.io_types.stock import StockID
from agent_service.tools.regions import (
    FilterStockContryOfDomicileInput,
    FilterStockRegionInput,
    filter_stocks_by_country_of_domicile,
    filter_stocks_by_region,
)
from agent_service.types import PlanRunContext

MSFT = StockID(gbi_id=6963, isin="", symbol="MSFT", company_name="Microsoft Corporation")  # USA
FORD = StockID(gbi_id=4579, isin="", symbol="F", company_name="Ford Motor Company")  # USA
RBC = StockID(gbi_id=12838, isin="", symbol="RY", company_name="Royal Bank of Canada")  # CAN
CHN = StockID(
    gbi_id=397877,
    isin="",
    symbol="601398",
    company_name="Industrial and Commercial Bank of China Limited",
)  # CHN
TOY = StockID(gbi_id=389721, isin="", symbol="7203", company_name="Toyota Motor Corporation")  # JPN
TCS = StockID(
    gbi_id=387282, isin="", symbol="TCS", company_name="Tata Consultancy Services Limited"
)  # IND
SIE = StockID(
    gbi_id=194452, isin="", symbol="SIE", company_name="Siemens Aktiengesellschaft"
)  # DEU
ES = StockID(gbi_id=193085, isin="", symbol="ES", company_name="Esso S.A.F.")  # FRA
RTWP = StockID(
    gbi_id=216296,
    isin="",
    symbol="RTWP",
    company_name="Legal & General UCITS ETF Plc - L&G Russell 2000 US Small Cap UCITS ETF",
)  # UK


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

    async def test_region_filter_with_region_names(self):
        stock_ids = [MSFT, FORD, RBC, CHN, TOY, TCS, SIE, ES, RTWP]

        # Test NORTHERN_AMERICA region
        args = FilterStockRegionInput(region_name="NORTH_AMERICA", stock_ids=stock_ids)
        stocks = await filter_stocks_by_region(args=args, context=self.context)
        self.assertEqual(3, len(stocks))
        self.assertTrue(any(stock.symbol == "MSFT" for stock in stocks))
        self.assertTrue(any(stock.symbol == "F" for stock in stocks))
        self.assertTrue(any(stock.symbol == "RY" for stock in stocks))

        # Test NORTHERN_AMERICA region
        args = FilterStockRegionInput(region_name="NORTH AMERICA", stock_ids=stock_ids)
        stocks = await filter_stocks_by_region(args=args, context=self.context)
        self.assertEqual(3, len(stocks))
        self.assertTrue(any(stock.symbol == "MSFT" for stock in stocks))
        self.assertTrue(any(stock.symbol == "F" for stock in stocks))
        self.assertTrue(any(stock.symbol == "RY" for stock in stocks))

        # Test ASIA region
        args = FilterStockRegionInput(region_name="ASIA", stock_ids=stock_ids)
        stocks = await filter_stocks_by_region(args=args, context=self.context)
        self.assertEqual(3, len(stocks))
        self.assertTrue(any(stock.symbol == "601398" for stock in stocks))
        self.assertTrue(any(stock.symbol == "7203" for stock in stocks))
        self.assertTrue(any(stock.symbol == "TCS" for stock in stocks))

        # Test EUROPE region
        args = FilterStockRegionInput(region_name="EUROPE", stock_ids=stock_ids)
        stocks = await filter_stocks_by_region(args=args, context=self.context)
        self.assertEqual(3, len(stocks))
        self.assertTrue(any(stock.symbol == "SIE" for stock in stocks))
        self.assertTrue(any(stock.symbol == "ES" for stock in stocks))
        self.assertTrue(any(stock.symbol == "RTWP" for stock in stocks))

        # Test UK region
        args = FilterStockRegionInput(region_name="UK", stock_ids=stock_ids)
        stocks = await filter_stocks_by_region(args=args, context=self.context)
        self.assertEqual(1, len(stocks))
        self.assertTrue(any(stock.symbol == "RTWP" for stock in stocks))

    async def test_region_filter_with_subregions(self):
        stock_ids = [MSFT, FORD, RBC, CHN, TOY, TCS, SIE, ES]

        # Test EASTERN_ASIA subregion
        args = FilterStockRegionInput(region_name="EASTERN_ASIA", stock_ids=stock_ids)
        stocks = await filter_stocks_by_region(args=args, context=self.context)
        self.assertEqual(2, len(stocks))
        self.assertTrue(any(stock.symbol == "601398" for stock in stocks))
        self.assertTrue(any(stock.symbol == "7203" for stock in stocks))

        # Test WESTERN_EUROPE subregion
        args = FilterStockRegionInput(region_name="WESTERN_EUROPE", stock_ids=stock_ids)
        stocks = await filter_stocks_by_region(args=args, context=self.context)
        self.assertEqual(2, len(stocks))
        self.assertTrue(any(stock.symbol == "SIE" for stock in stocks))
        self.assertTrue(any(stock.symbol == "ES" for stock in stocks))

    async def test_region_filter_with_country_names(self):
        stock_ids = [MSFT, FORD, RBC, CHN, TOY, TCS, SIE, ES]

        # Test China
        args = FilterStockContryOfDomicileInput(country_name="China", stock_ids=stock_ids)
        stocks = await filter_stocks_by_country_of_domicile(args=args, context=self.context)
        self.assertEqual(1, len(stocks))
        self.assertTrue(any(stock.symbol == "601398" for stock in stocks))

        args = FilterStockContryOfDomicileInput(
            country_name="People's Republic of China", stock_ids=stock_ids
        )
        stocks = await filter_stocks_by_country_of_domicile(args=args, context=self.context)
        self.assertEqual(1, len(stocks))
        self.assertTrue(any(stock.symbol == "601398" for stock in stocks))

        # Test United States
        args = FilterStockContryOfDomicileInput(country_name="United States", stock_ids=stock_ids)
        stocks = await filter_stocks_by_country_of_domicile(args=args, context=self.context)
        self.assertEqual(2, len(stocks))
        self.assertTrue(any(stock.symbol == "MSFT" for stock in stocks))
        self.assertTrue(any(stock.symbol == "F" for stock in stocks))

        args = FilterStockContryOfDomicileInput(country_name="USA", stock_ids=stock_ids)
        stocks = await filter_stocks_by_country_of_domicile(args=args, context=self.context)
        self.assertEqual(2, len(stocks))
        self.assertTrue(any(stock.symbol == "MSFT" for stock in stocks))
        self.assertTrue(any(stock.symbol == "F" for stock in stocks))

        # Test Canada
        args = FilterStockContryOfDomicileInput(country_name="Canada", stock_ids=stock_ids)
        stocks = await filter_stocks_by_country_of_domicile(args=args, context=self.context)
        self.assertEqual(1, len(stocks))
        self.assertTrue(any(stock.symbol == "RY" for stock in stocks))

        args = FilterStockContryOfDomicileInput(country_name="CAN", stock_ids=stock_ids)
        stocks = await filter_stocks_by_country_of_domicile(args=args, context=self.context)
        self.assertEqual(1, len(stocks))
        self.assertTrue(any(stock.symbol == "RY" for stock in stocks))
