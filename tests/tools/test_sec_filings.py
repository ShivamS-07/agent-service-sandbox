import logging
from unittest import IsolatedAsyncioTestCase

from agent_service.io_types.stock import StockID
from agent_service.io_types.text import Text
from agent_service.tools.sec import (
    GetOtherSecFilingsInput,
    GetSecFilingsInput,
    GetSecFilingsWithTypeInput,
    SecFilingsTypeLookupInput,
    SecFilingType,
    get_10k_10q_sec_filings,
    get_non_10k_10q_sec_filings,
    get_sec_filings_with_type,
    sec_filings_type_lookup,
)
from agent_service.types import PlanRunContext
from agent_service.utils.string_utils import get_sections


class TestSecFilings(IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        logging.getLogger("asyncio").setLevel(logging.ERROR)

    async def asyncSetUp(self):
        self.context = PlanRunContext.get_dummy()

    async def test_get_sec_filing(self):
        stock = StockID(gbi_id=714, symbol="", isin="", company_name="")
        sec_filing_texts = await get_10k_10q_sec_filings(
            args=GetSecFilingsInput(stock_ids=[stock]),
            context=self.context,
        )
        self.assertIsNotNone(sec_filing_texts)

    async def test_get_other_sec_filing(self):
        stock = StockID(gbi_id=714, symbol="", isin="", company_name="")
        sec_filing_texts = await get_non_10k_10q_sec_filings(
            args=GetOtherSecFilingsInput(stock_ids=[stock], form_types=["8-K", "S-1"]),
            context=self.context,
        )
        actual_sec_filings = await Text.get_all_strs(sec_filing_texts)
        # in theory it should be 2, but just in case only comparing with 0
        self.assertTrue(len(actual_sec_filings) > 0)

    async def test_sec_filings_type_lookup(self):
        sec_filing_types = await sec_filings_type_lookup(
            args=SecFilingsTypeLookupInput(
                search_term="IPO",
                form_types=[
                    # regular forms
                    "8-K",
                    "S-4",
                    # fuzzy matching
                    "10K",
                    "10Q",
                    "8K",
                    "10K405",
                    "8K15-D5",
                    "APP-ORDR",
                    "NRSRO UPD",
                    "Real Estate Supplement",
                    "property portfolio",
                    "Form DEF 14A",
                    "Debt and financing documents",
                ],
            ),
            context=self.context,
        )
        filing_types = [filing_type.name for filing_type in sec_filing_types]
        self.assertTrue(len(filing_types) > 9)
        self.assertTrue(len(filing_types) < 17)
        self.assertTrue("8-K" in filing_types)
        self.assertTrue("10-K" in filing_types)
        self.assertTrue("10-Q" in filing_types)
        self.assertTrue("S-4" in filing_types)
        self.assertTrue("S-1" in filing_types)  # from search term
        self.assertTrue("10-K405" in filing_types)
        self.assertTrue("8-K15D5" in filing_types)
        self.assertTrue("APP ORDR" in filing_types)
        self.assertTrue("NRSRO-UPD" in filing_types)
        self.assertTrue("DEF 14A" in filing_types)

    async def test_get_sec_filings_with_type(self):
        stock = StockID(gbi_id=714, symbol="", isin="", company_name="")
        sec_filing_texts = await get_sec_filings_with_type(
            args=GetSecFilingsWithTypeInput(
                stock_ids=[stock], form_types=[SecFilingType(name="8-K"), SecFilingType(name="S-1")]
            ),
            context=self.context,
        )
        actual_sec_filings = await Text.get_all_strs(sec_filing_texts)
        # in theory it should be 2, but just in case only comparing with 0
        self.assertTrue(len(actual_sec_filings) > 0)

    async def test_duplicate_sections(self):
        test_string = """First Section

This is the first section of this document, needs to be of a certain length before it counts.

Second Section

This is the second section of this document, needs to be of a certain length before it counts.

Third Section

This is the third section of this document, needs to be of a certain length before it counts.

Second Section

This is a duplicate Second Section, but for consistency it will be the 'first' second section"""

        sections = get_sections(test_string)

        self.assertEqual(len(sections), 4)
        self.assertIn("Second Section", sections)
        self.assertIn("Second Section__2", sections)
        self.assertIn("duplicate", sections["Second Section"])
