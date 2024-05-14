import datetime
from unittest.async_case import IsolatedAsyncioTestCase

import pandas as pd

from agent_service.io_types import TimeSeriesTable
from agent_service.tools.dates import DateFromDateStrInput, get_date_from_date_str
from agent_service.tools.lists import (
    FlattenListsInput,
    GetIndexInput,
    collapse_lists,
    get_element_from_list,
)
from agent_service.tools.tables import TimeseriesTableAvgInput, average_table_by_date
from agent_service.types import PlanRunContext


class TestUtilityTools(IsolatedAsyncioTestCase):
    # LISTS
    async def test_collapse_lists(self):
        args = FlattenListsInput(lists_of_lists=[[1, 2, 3], [4, 5, 6], ["a", "b"]])
        res = await collapse_lists(args, context=PlanRunContext.get_dummy())
        self.assertEqual(res, [1, 2, 3, 4, 5, 6, "a", "b"])

    async def test_get_element_from_list(self):
        args = GetIndexInput(list=[1, 2, 3, 4], index=2)
        res = await get_element_from_list(args, context=PlanRunContext.get_dummy())
        self.assertEqual(res, 3)

    # TABLES
    async def test_average_table_by_date(self):
        df = pd.DataFrame(data=[[1, 2, 3], [2, 3, 4]])
        args = TimeseriesTableAvgInput(table=TimeSeriesTable(val=df))
        avg = await average_table_by_date(args, context=PlanRunContext.get_dummy())
        pd.testing.assert_frame_equal(
            avg.val, pd.DataFrame(data=[2.0, 3.0], columns=["Averaged Value"])
        )

    # DATES
    async def test_get_date_from_date_str(self):
        # We don't need to unit test the dateparser library here, just a quick example
        args = DateFromDateStrInput(date_str="3 days ago")
        res = await get_date_from_date_str(args, context=PlanRunContext.get_dummy())
        self.assertEqual(type(res), datetime.date)
        self.assertEqual(res, datetime.date.today() - datetime.timedelta(days=3))
