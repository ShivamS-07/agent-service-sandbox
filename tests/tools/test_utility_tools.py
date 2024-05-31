import datetime
from unittest.async_case import IsolatedAsyncioTestCase

from agent_service.tools.dates import DateFromDateStrInput, get_date_from_date_str
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
