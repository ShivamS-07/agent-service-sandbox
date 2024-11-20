import unittest

from agent_service.io_types.stock import StockID
from agent_service.planner.errors import NonRetriableError
from agent_service.tools.sectors import (
    GetStockSectorInput,
    SectorFilterInput,
    SectorID,
    SectorIdentifierLookupInput,
    get_all_gics_classifications,
    get_default_stock_list,
    get_stock_sector,
    gics_sector_industry_filter,
    sector_identifier_lookup,
)
from agent_service.types import PlanRunContext

AAPL = StockID(gbi_id=714, isin="", symbol="AAPL", company_name="")


class SectorIdentifierLookup(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.context = PlanRunContext.get_dummy()

        # uncomment for easier debugging
        from agent_service.utils.logs import init_test_logging

        init_test_logging()

    async def test_geT_stock_sector(self):
        args = GetStockSectorInput(stock_id=AAPL)
        sector_id = await get_stock_sector(args=args, context=self.context)
        print(sector_id)

    async def test_get_all_gics(self):
        gics = get_all_gics_classifications()
        if not gics:
            self.assertTrue(False, gics)

    async def test_no_sector(self):
        q_a = [
            ("wolverine", -1),
            ("None", -1),
            ("null", -1),
            ("", -1),
            ("sector", -1),
            ("sectors", -1),
            ("companies", -1),
            ("No Sector", -1),
            # ("semiconductor sector", -1),
            # ("semiconductor", -1),
        ]
        for q, a in q_a:
            args = SectorIdentifierLookupInput(sector_name=q)
            with self.subTest(q=q, a=a):
                with self.assertRaises(ValueError):
                    result = await sector_identifier_lookup(args, self.context)
                    self.assertEqual(result.sec_id, a, q)

    async def test_sector_level_1_exact_match(self):
        # match multiple because some names are very similar
        q_a = [
            ("Energy", [10, 1010]),
            ("Materials", [15, 1510]),
            ("Industrials", [20]),
            ("Consumer Discretionary", [25]),
            ("Consumer Staples", [30]),
            ("Health Care", [35]),
            ("Financials", [40]),
            ("Information Technology", [45]),
            ("Communication Services", [50, 5010]),
            ("Utilities", [55, 5510]),
            ("Real Estate", [60]),
        ]

        for q, a in q_a:
            args = SectorIdentifierLookupInput(sector_name=q)
            with self.subTest(q=q, a=a):
                result = await sector_identifier_lookup(args, self.context)
                self.assertTrue(result.sec_id in a)

    async def test_sector_level_2_exact_match(self):
        # match multiple because some names are very similar
        q_a = [
            ("Energy", [10, 1010]),
            ("Materials", [15, 1510]),
            ("Capital Goods", [2010]),
            ("Transportation", [2030]),
            ("Consumer Services", [2530]),
            ("Food, Beverage & Tobacco", [3020]),
            ("Pharmaceuticals, Biotechnology & Life Sciences", [3520]),
            ("Banks", [4010, 401010]),
            ("Insurance", [4030, 403010]),
            ("Technology Hardware & Equipment", [4520]),
            ("Media & Entertainment", [5020]),
            ("Utilities", [55, 5510]),
            ("Equity Real Estate Investment Trusts (REITs)", [6010]),
            ("REITs", [6010]),
        ]

        for q, a in q_a:
            args = SectorIdentifierLookupInput(sector_name=q)
            with self.subTest(q=q, a=a):
                result = await sector_identifier_lookup(args, self.context)
                self.assertTrue(result.sec_id in a)

    @unittest.skip("still flaky, needs non-gpt exact match")
    async def test_sector_level_3_exact_match(self):
        # match multiple because some names are very similar
        q_a = [
            ("Oil, Gas & Consumable Fuels", [101020]),
            ("Metals & Mining", [151040]),
            ("Aerospace & Defense", [201010, 20101010]),
            ("Transportation Infrastructure", [203050]),
            ("Household Durables", [252010]),
            ("Hotels, Restaurants & Leisure", [253010]),
            ("Specialty Retail", [255040]),
            ("Beverages", [302010]),
            ("Health Care Providers & Services", [351020]),
            ("Biotechnology", [352010, 35201010]),
            ("Capital Markets", [402030]),
            ("Electronic Equipment, Instruments & Components", [452030, 45203010, 45203015]),
            ("Interactive Media & Services", [502030, 50203010]),
            ("Gas Utilities", [551020, 55102010]),
            ("Specialized REITs", [601080, 60108010]),
        ]

        for q, a in q_a:
            args = SectorIdentifierLookupInput(sector_name=q)
            with self.subTest(q=q, a=a):
                result = await sector_identifier_lookup(args, self.context)
                print(result.sec_id, a)
                self.assertTrue(result.sec_id in a)

    async def test_get_default_stock_list(self):
        stocks = await get_default_stock_list(user_id=self.context.user_id)
        self.assertGreater(len(stocks), 490)
        self.assertLess(len(stocks), 525)

        # cache hit
        stocks = await get_default_stock_list(user_id=self.context.user_id)
        self.assertGreater(len(stocks), 490)
        self.assertLess(len(stocks), 525)

    async def test_gics_sector_industry_filter(self):
        args = SectorFilterInput(
            sector_id=SectorID(sec_id=40, sec_name="Financials"),
            stock_ids=[StockID(gbi_id=1092, symbol="", isin="", company_name="")],
        )  # financials  # BAC

        stocks = await gics_sector_industry_filter(args=args, context=self.context)
        self.assertEqual(1, len(stocks))

        args = SectorFilterInput(
            sector_id=SectorID(sec_id=50, sec_name="Communcation Services"),
            stock_ids=[StockID(gbi_id=1092, symbol="", isin="", company_name="")],
        )  # 'Communication Services'  # BAC

        with self.assertRaises(NonRetriableError):
            stocks = await gics_sector_industry_filter(args=args, context=self.context)

        args = SectorFilterInput(
            sector_id=SectorID(sec_id=40, sec_name="Financials"),  # financials
        )

        stocks = await gics_sector_industry_filter(args=args, context=self.context)
        self.assertGreater(len(stocks), 0)
        self.assertLess(len(stocks), 400)
