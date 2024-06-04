import datetime
from unittest.async_case import IsolatedAsyncioTestCase

from agent_service.io_types.stock import StockID
from agent_service.io_types.text import Text
from agent_service.tools.dates import DateFromDateStrInput, get_date_from_date_str
from agent_service.tools.lists import CombineListsInput, add_lists, intersect_lists
from agent_service.types import PlanRunContext


class TestUtilityTools(IsolatedAsyncioTestCase):
    # DATES
    async def test_get_date_from_date_str(self):
        # We don't need to unit test the dateparser library here, just a quick example
        args = DateFromDateStrInput(date_str="3 days ago")
        res = await get_date_from_date_str(args, context=PlanRunContext.get_dummy())
        self.assertEqual(type(res), datetime.date)
        self.assertEqual(res, datetime.date.today() - datetime.timedelta(days=3))

    async def test_get_date_from_date_str_(self):
        # We don't need to unit test the dateparser library here, just a quick example
        args = DateFromDateStrInput(date_str="Last 3 quarters")
        res = await get_date_from_date_str(args, context=PlanRunContext.get_dummy())
        print(res)
        self.assertEqual(type(res), datetime.date)

    async def test_add_lists(self):
        text_list_1 = [Text(id="1"), Text(id="2")]
        text_list_2 = [Text(id="2"), Text(id="3")]
        args = CombineListsInput(list1=text_list_1, list2=text_list_2)
        result = await add_lists(args, context=PlanRunContext.get_dummy())
        self.assertEqual(len(result), 3)
        self.assertIn(Text(id="1"), result)
        self.assertIn(Text(id="2"), result)
        self.assertIn(Text(id="3"), result)

    async def test_intersect_lists(self):
        stock_list_1 = [
            StockID(gbi_id=1, symbol=None, isin="", company_name=""),
            StockID(gbi_id=2, symbol=None, isin="", company_name=""),
            StockID(gbi_id=3, symbol=None, isin="", company_name=""),
        ]
        stock_list_2 = [
            StockID(gbi_id=4, symbol=None, isin="", company_name=""),
            StockID(gbi_id=2, symbol=None, isin="", company_name=""),
            StockID(gbi_id=3, symbol=None, isin="", company_name=""),
        ]
        args = CombineListsInput(list1=stock_list_1, list2=stock_list_2)
        result = await intersect_lists(args, context=PlanRunContext.get_dummy())
        self.assertEqual(len(result), 2)
        self.assertIn(StockID(gbi_id=2, symbol=None, isin="", company_name=""), result)
        self.assertIn(StockID(gbi_id=3, symbol=None, isin="", company_name=""), result)
