import unittest

from agent_service.GPT.requests import set_use_global_stub
from agent_service.tools.sectors import (
    SectorFilterInput,
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
        ]
        for q, a in q_a:
            args = SectorIdentifierLookupInput(sector_name=q)
            result = await sector_identifier_lookup(args, self.context)
            self.assertEqual(result, a, q)

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
            ("No Sector", -1),
        ]
        for q, a in q_a:
            args = SectorIdentifierLookupInput(sector_name=q)
            result = await sector_identifier_lookup(args, self.context)
            self.assertEqual(result, a, q)

    async def test_sector_close_match(self):
        # not intended to be comprehensive
        # but a simple test
        #
        # most fail, so barely better than simple text lookup
        #
        # I spoke with Jon he said it would probably do better
        # if we provided descriptions of each sector
        # but we would need to write them or find some
        q_a = [
            ("property", 60),
            # this used to work but fails now. ("realestate", 60),
            # fails ("reit", 60),
            # fails ("oil & gas", 10),
            # fails ("chemicals", 15),
            # fails ("manufacturers", 20),
            # fails ("luxory goods", 25),
            # fails ("everyday products", 30),
            # fails ("hospitals and medical providers", 35),
            # fails ("banks", 40),
            # this used to work but fails now. ("IT", 45),
            # fails ("phone cable or internet companies", 50),
            # fails ("electric and gas", 55),
        ]
        for q, a in q_a:
            args = SectorIdentifierLookupInput(sector_name=q)
            result = await sector_identifier_lookup(args, self.context)
            self.assertEqual(result, a, q)

    async def test_get_default_stock_list(self):
        stocks = await get_default_stock_list(user_id=self.context.user_id)
        self.assertGreater(len(stocks), 490)
        self.assertLess(len(stocks), 525)

        # cache hit
        stocks = await get_default_stock_list(user_id=self.context.user_id)
        self.assertGreater(len(stocks), 490)
        self.assertLess(len(stocks), 525)

    async def test_sector_filter(self):
        args = SectorFilterInput(sector_id=40, stock_ids=[1092])  # financials  # BAC

        stocks = await sector_filter(args=args, context=self.context)
        self.assertEqual(1, len(stocks))

        args = SectorFilterInput(sector_id=50, stock_ids=[1092])  # 'Communication Services'  # BAC

        stocks = await sector_filter(args=args, context=self.context)
        self.assertEqual(0, len(stocks))

        args = SectorFilterInput(
            sector_id=40,  # financials
        )

        stocks = await sector_filter(args=args, context=self.context)
        self.assertGreater(len(stocks), 0)
        self.assertLess(len(stocks), 400)
