import logging
from unittest import IsolatedAsyncioTestCase

from agent_service.io_types.text import Text
from agent_service.tools.other_text import (
    GetAllTextDataForStocksInput,
    GetSecFilingsInput,
    get_all_text_data_for_stocks,
    get_sec_filings,
    get_stock_aligned_sec_filings,
)
from agent_service.types import PlanRunContext


class TestTextData(IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        logging.getLogger("asyncio").setLevel(logging.ERROR)

    async def asyncSetUp(self):
        self.context = PlanRunContext.get_dummy()

    async def test_get_sec_filing(self):
        sec_filing_texts = await get_sec_filings(
            args=GetSecFilingsInput(stock_ids=[714]),
            context=self.context,
        )
        self.assertIsNotNone(sec_filing_texts)

        sec_filing_mapping = await get_stock_aligned_sec_filings(
            args=GetSecFilingsInput(stock_ids=[714]),
            context=self.context,
        )
        self.assertIsNotNone(sec_filing_mapping)

        actual_sec_filing_1 = Text.get_all_strs(sec_filing_texts)
        actual_sec_filing_2 = Text.get_all_strs(sec_filing_mapping.val)
        self.assertIsNotNone(actual_sec_filing_1)
        self.assertIsNotNone(actual_sec_filing_2)
        # They should have the same content
        self.assertEqual(actual_sec_filing_1[0][:100], actual_sec_filing_2[714][:100])
        # print(actual_sec_filing_1[0])  # useful when debugging

    async def test_get_all_text_data(self):
        all_data = await get_all_text_data_for_stocks(
            args=GetAllTextDataForStocksInput(stock_ids=[18654]),
            context=self.context,
        )
        types = set()
        for text in all_data.val[18654].val:
            types.add(type(text))
        self.assertEqual(len(types), 4)