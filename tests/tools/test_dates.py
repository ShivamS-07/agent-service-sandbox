import datetime
from unittest.async_case import IsolatedAsyncioTestCase

from agent_service.tools.dates import (
    DateRangeInput,
    GetDateRangeInput,
    GetEndOfDateRangeInput,
    GetStartOfDateRangeInput,
    get_date_range,
    get_end_of_date_range,
    get_n_width_date_range_near_date,
    get_start_of_date_range,
)
from agent_service.types import PlanRunContext


class TestDateUtils(IsolatedAsyncioTestCase):
    async def test_get_date_range(self):
        date_range_arg = "Obtain the date range for the past 3 months"
        dummy_context = PlanRunContext.get_dummy()
        date_range = await get_date_range(
            GetDateRangeInput(date_range_str=date_range_arg), dummy_context
        )
        start_date, end_date = date_range.start_date, date_range.end_date
        self.assertGreaterEqual((end_date - start_date).days, 82)
        self.assertLessEqual((end_date - start_date).days, 98)
        self.assertLessEqual(abs(datetime.date.today() - end_date).days, 3)
        self.assertLessEqual(
            abs((datetime.date.today() - datetime.timedelta(days=90)) - start_date).days, 8
        )

    async def test_get_date_range_2000s(self):
        date_range_arg = "Obtain the date range for the 2000's decade"
        dummy_context = PlanRunContext.get_dummy()
        date_range = await get_date_range(
            GetDateRangeInput(date_range_str=date_range_arg), dummy_context
        )
        start_date, end_date = date_range.start_date, date_range.end_date
        self.assertLessEqual(abs(end_date - datetime.date(2010, 1, 1)).days, 10)
        self.assertLessEqual(abs(start_date - datetime.date(2000, 1, 1)).days, 10)

    async def test_get_end_of_date_range(self):
        date_range_arg = "Obtain the date range for the past 3 months"
        dummy_context = PlanRunContext.get_dummy()
        date_range = await get_date_range(
            GetDateRangeInput(date_range_str=date_range_arg), dummy_context
        )
        end_date_range = await get_end_of_date_range(
            GetEndOfDateRangeInput(date_range=date_range), dummy_context
        )
        start_date, end_date = end_date_range.start_date, end_date_range.end_date
        self.assertLessEqual(abs(datetime.date.today() - end_date).days, 3)
        self.assertLessEqual(abs(datetime.date.today() - start_date).days, 3)
        self.assertLessEqual((end_date - start_date).days, 0)

    async def test_get_end_of_date_range_2000s(self):
        date_range_arg = "Obtain the date range for the 2000's decade"
        dummy_context = PlanRunContext.get_dummy()
        date_range = await get_date_range(
            GetDateRangeInput(date_range_str=date_range_arg), dummy_context
        )
        end_date_range = await get_end_of_date_range(
            GetEndOfDateRangeInput(date_range=date_range), dummy_context
        )
        start_date, end_date = end_date_range.start_date, end_date_range.end_date
        self.assertLessEqual(abs(end_date - datetime.date(2010, 1, 1)).days, 10)
        self.assertLessEqual(abs(start_date - datetime.date(2010, 1, 1)).days, 10)
        self.assertEqual(abs(start_date - end_date).days, 0)

    async def test_get_start_of_date_range(self):
        date_range_arg = "Obtain the date range for the past 3 months"
        dummy_context = PlanRunContext.get_dummy()
        date_range = await get_date_range(
            GetDateRangeInput(date_range_str=date_range_arg), dummy_context
        )
        start_date_range = await get_start_of_date_range(
            GetStartOfDateRangeInput(date_range=date_range), dummy_context
        )
        start_date, end_date = start_date_range.start_date, start_date_range.end_date
        self.assertLessEqual(abs(datetime.date.today() - end_date).days, 98)
        self.assertGreaterEqual(abs(datetime.date.today() - end_date).days, 82)
        self.assertLessEqual(abs(datetime.date.today() - start_date).days, 98)
        self.assertGreaterEqual(abs(datetime.date.today() - start_date).days, 82)
        self.assertLessEqual((end_date - start_date).days, 0)

    async def test_get_start_of_date_range_2000s(self):
        date_range_arg = "Obtain the date range for the 2000's decade"
        dummy_context = PlanRunContext.get_dummy()
        date_range = await get_date_range(
            GetDateRangeInput(date_range_str=date_range_arg), dummy_context
        )
        start_date_range = await get_start_of_date_range(
            GetStartOfDateRangeInput(date_range=date_range), dummy_context
        )
        start_date, end_date = start_date_range.start_date, start_date_range.end_date
        self.assertLessEqual(abs(end_date - datetime.date(2000, 1, 1)).days, 10)
        self.assertLessEqual(abs(start_date - datetime.date(2000, 1, 1)).days, 10)
        self.assertEqual(abs(start_date - end_date).days, 0)

    async def test_get_n_width_date_range_near_date_1_year(self):
        near_date = datetime.date(2022, 6, 1)
        width_years = 1
        args = DateRangeInput(near_date=near_date, width_years=width_years)
        dummy_context = PlanRunContext.get_dummy()
        date_range = await get_n_width_date_range_near_date(args, dummy_context)
        start_date, end_date = date_range.start_date, date_range.end_date
        self.assertLessEqual(abs(abs(end_date - start_date).days - 365), 10)
        self.assertLessEqual(abs(start_date - datetime.date(2021, 12, 1)).days, 10)
        self.assertLessEqual(abs(end_date - datetime.date(2022, 12, 1)).days, 10)

    async def test_get_n_width_date_range_near_date_10_days(self):
        near_date = datetime.date(2020, 3, 15)
        width_days = 20
        args = DateRangeInput(near_date=near_date, width_days=width_days)
        dummy_context = PlanRunContext.get_dummy()
        date_range = await get_n_width_date_range_near_date(args, dummy_context)
        start_date, end_date = date_range.start_date, date_range.end_date
        self.assertLessEqual(abs(abs(end_date - start_date).days - 20), 2)
        self.assertLessEqual(abs(start_date - datetime.date(2020, 3, 5)).days, 2)
        self.assertLessEqual(abs(end_date - datetime.date(2020, 3, 25)).days, 2)

    async def test_get_n_width_date_range_near_date_3_quarters(self):
        near_date = datetime.date(2023, 5, 31)
        width_quarters = 3
        args = DateRangeInput(near_date=near_date, width_quarters=width_quarters)
        dummy_context = PlanRunContext.get_dummy()
        date_range = await get_n_width_date_range_near_date(args, dummy_context)
        start_date, end_date = date_range.start_date, date_range.end_date
        self.assertLessEqual(abs(abs(end_date - start_date).days - 270), 10)
        self.assertLessEqual(abs(start_date - datetime.date(2023, 1, 15)).days, 5)
        self.assertLessEqual(abs(end_date - datetime.date(2023, 10, 15)).days, 5)

    async def test_get_n_width_date_range_near_date_10_weeks(self):
        near_date = datetime.date(2024, 6, 20)
        width_weeks = 10
        args = DateRangeInput(near_date=near_date, width_weeks=width_weeks)
        dummy_context = PlanRunContext.get_dummy()
        date_range = await get_n_width_date_range_near_date(args, dummy_context)
        start_date, end_date = date_range.start_date, date_range.end_date
        self.assertLessEqual(abs(abs(end_date - start_date).days - 70), 3)
        self.assertLessEqual(abs(start_date - datetime.date(2024, 5, 16)).days, 2)
        self.assertLessEqual(abs(end_date - datetime.date(2024, 7, 25)).days, 2)

    async def test_get_n_width_date_range_near_date_7_months(self):
        near_date = datetime.date(2016, 9, 20)
        width_months = 7
        args = DateRangeInput(near_date=near_date, width_months=width_months)
        dummy_context = PlanRunContext.get_dummy()
        date_range = await get_n_width_date_range_near_date(args, dummy_context)
        start_date, end_date = date_range.start_date, date_range.end_date
        self.assertLessEqual(abs(abs(end_date - start_date).days - 210), 10)
        self.assertLessEqual(abs(start_date - datetime.date(2016, 6, 5)).days, 5)
        self.assertLessEqual(abs(end_date - datetime.date(2017, 1, 5)).days, 5)
