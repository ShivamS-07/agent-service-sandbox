import unittest

from agent_service.GPT.requests import set_use_global_stub
from agent_service.io_types.stock import StockID
from agent_service.tools.sectors import (
    SectorFilterInput,
    SectorID,
    SectorIdentifierLookupInput,
    get_all_sectors,
    get_default_stock_list,
    sector_filter,
    sector_identifier_lookup,
)
from agent_service.types import PlanRunContext


class SectorIdentifierLookup(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        set_use_global_stub(False)
        self.context = PlanRunContext.get_dummy()

        # uncomment for easier debugging
        from agent_service.utils.logs import init_test_logging

        init_test_logging()

    async def test_get_all_sectors(self):
        sectors = get_all_sectors()
        if not sectors:
            self.assertTrue(False, sectors)

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
        ]

        for q, a in q_a:
            args = SectorIdentifierLookupInput(sector_name=q)
            with self.subTest(q=q, a=a):
                result = await sector_identifier_lookup(args, self.context)
                self.assertEqual(result.sec_id, a, q)

    async def test_sector_close_match(self):
        # not intended to be comprehensive
        # but a simple test
        #
        # I spoke with Jon he said it would probably do better
        # if we provided descriptions of each sector
        # but we would need to write them or find some
        q_a = [
            ("property", 60),
            ("realestate", 60),
            ("reit", 60),
            ("oil & gas", 10),
            ("chemicals", 15),
            ("manufacturers", 20),
            ("luxory goods", 25),
            ("everyday products", 30),
            ("hospitals and medical providers", 35),
            ("healthcare", 35),
            ("banks", 40),
            ("IT", 45),
            ("technology", 45),
            ("tech", 45),
            ("phone cable or internet companies", 50),
            ("telecoms", 50),
            ("electric and gas", 55),
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

    async def test_sector_filter(self):
        args = SectorFilterInput(
            sector_id=SectorID(sec_id=40, sec_name="Financials"),
            stock_ids=[StockID(gbi_id=1092, symbol="", isin="")],
        )  # financials  # BAC

        stocks = await sector_filter(args=args, context=self.context)
        self.assertEqual(1, len(stocks))

        args = SectorFilterInput(
            sector_id=SectorID(sec_id=50, sec_name="Communcation Services"),
            stock_ids=[StockID(gbi_id=1092, symbol="", isin="")],
        )  # 'Communication Services'  # BAC

        stocks = await sector_filter(args=args, context=self.context)
        self.assertEqual(0, len(stocks))

        args = SectorFilterInput(
            sector_id=SectorID(sec_id=40, sec_name="Financials"),  # financials
        )

        stocks = await sector_filter(args=args, context=self.context)
        self.assertGreater(len(stocks), 0)
        self.assertLess(len(stocks), 400)
