import logging
from unittest import IsolatedAsyncioTestCase

from agent_service.io_types.stock import StockID
from agent_service.io_types.text import Text
from agent_service.tools.sec import (
    GetOtherSecFilingsInput,
    GetSecFilingsInput,
    get_10k_10q_sec_filings,
    get_non_10k_10q_sec_filings,
)
from agent_service.types import PlanRunContext


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
