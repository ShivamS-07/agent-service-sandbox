import logging
from unittest import IsolatedAsyncioTestCase

from agent_service.io_types.stock import StockID
from agent_service.tools.other_text import (
    GetAllTextDataForStocksInput,
    get_all_text_data_for_stocks,
)
from agent_service.types import PlanRunContext


class TestTextData(IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        logging.getLogger("asyncio").setLevel(logging.ERROR)

    async def asyncSetUp(self):
        self.context = PlanRunContext.get_dummy()

    async def test_get_all_text_data(self):

        all_data = await get_all_text_data_for_stocks(
            args=GetAllTextDataForStocksInput(
                stock_ids=[StockID(gbi_id=18654, symbol="", isin="", company_name="")]
            ),
            context=self.context,
        )

        types = set()
        for text in all_data:
            types.add(type(text))
        self.assertEqual(len(types), 4)
