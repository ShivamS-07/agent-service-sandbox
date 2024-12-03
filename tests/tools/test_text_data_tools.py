import logging
import unittest
from unittest import IsolatedAsyncioTestCase

from agent_service.io_types.stock import StockID
from agent_service.tools.other_text import (
    GetAllStocksFromTextInput,
    GetAllTextDataForStocksInput,
    get_all_stocks_from_text,
    get_default_text_data_for_stocks,
)
from agent_service.types import PlanRunContext
from tests.tools.test_custom_docs import CUSTOM_DOC_DEV_TEST_USER


class TestTextData(IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        logging.getLogger("asyncio").setLevel(logging.ERROR)

    async def asyncSetUp(self):
        # use the custom doc context as this will fetch custom documents as well
        self.context = PlanRunContext.get_dummy(user_id=CUSTOM_DOC_DEV_TEST_USER)

    @unittest.skip("occasionally fails")
    async def test_get_default_text_data(self):

        all_data = await get_default_text_data_for_stocks(
            args=GetAllTextDataForStocksInput(
                stock_ids=[StockID(gbi_id=2781, symbol="", isin="", company_name="")]
            ),
            context=self.context,
        )

        types = set()
        for text in all_data:
            types.add(type(text))
        self.assertEqual(len(types), 4)

    async def test_get_all_stocks_from_text(self):
        stock_text_data = await get_default_text_data_for_stocks(
            args=GetAllTextDataForStocksInput(
                stock_ids=[StockID(gbi_id=2781, symbol="", isin="", company_name="")]
            ),
            context=self.context,
        )
        data = await get_all_stocks_from_text(
            args=GetAllStocksFromTextInput(
                stock_texts=stock_text_data,
            ),
            context=self.context,
        )
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0].gbi_id, 2781)
