import unittest

from agent_service.io_types.stock import StockID
from agent_service.planner.errors import NonRetriableError
from agent_service.tools.sectors import (
    SectorFilterInput,
    SectorID,
    SectorIdentifierLookupInput,
    get_all_gics_classifications,
    get_default_stock_list,
    gics_sector_industry_filter,
    sector_identifier_lookup,
)
from agent_service.types import PlanRunContext


class SectorIdentifierLookup(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.context = PlanRunContext.get_dummy()

        # uncomment for easier debugging
        from agent_service.utils.logs import init_test_logging

        init_test_logging()

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

    async def test_sector_exact_match(self):
        q_a = [
            ("Real Estate", 60),
            ("Energy", 10),
            ("Materials", 15),
            ("Industrials", 20),
            ("Consumer Discretionary", 25),
            ("Consumer Staples", 30),
            ("Health Care", 35),
            ("Financials", 40),
            ("Information Technology", 45),
            ("Communication Services", 50),
            ("Utilities", 55),
            ("Real Estate Management & Development", 6020),
            ("Technology Hardware & Equipment", 4520),
            ("Health Care Equipment & Services", 3510),
            ("Transportation", 2030),
            ("Chemicals", 151010),
            ("Building Products", 201020),
            ("Hotels, Restaurants & Leisure", 253010),
            ("Biotechnology", 352010),
            ("Software", 451030),
            ("Gas Utilities", 551020),
            ("Data Center REITs", 60108050),
            ("Application Software", 45103010),
            ("Food Retail", 30101030),
            ("Heavy Electrical Equipment", 20104020),
            ("Oil & Gas Drilling", 10101010),
        ]

        for q, a in q_a:
            args = SectorIdentifierLookupInput(sector_name=q)
            with self.subTest(q=q, a=a):
                result = await sector_identifier_lookup(args, self.context)
                self.assertEqual(result.sec_id, a, q)

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
