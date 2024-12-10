import datetime
from unittest import IsolatedAsyncioTestCase

from agent_service.utils.date_utils import (
    convert_horizon_to_date,
    convert_horizon_to_days,
    date_to_pb_timestamp,
    get_next_quarter,
    get_prev_quarter,
    get_year_quarter_for_date,
)


class TestDateUtils(IsolatedAsyncioTestCase):
    async def test_convert_horizon_to_date_1w(self):
        horizon = "1W"
        res = convert_horizon_to_date(horizon)
        self.assertLessEqual(abs(abs(datetime.date.today() - res).days - 7), 1)

    async def test_convert_horizon_to_date_1m(self):
        horizon = "1M"
        res = convert_horizon_to_date(horizon)
        self.assertLessEqual(abs(abs(datetime.date.today() - res).days - 30), 3)

    async def test_convert_horizon_to_date_3m(self):
        horizon = "3M"
        res = convert_horizon_to_date(horizon)
        self.assertLessEqual(abs(abs(datetime.date.today() - res).days - 90), 5)

    async def test_convert_horizon_to_days_1w(self):
        horizon = "1W"
        res = convert_horizon_to_days(horizon)
        self.assertLessEqual(abs(res - 7), 1)

    async def test_convert_horizon_to_days_1m(self):
        horizon = "1M"
        res = convert_horizon_to_days(horizon)
        self.assertLessEqual(abs(res - 30), 3)

    async def test_convert_horizon_to_days_3m(self):
        horizon = "3M"
        res = convert_horizon_to_days(horizon)
        self.assertLessEqual(abs(res - 90), 5)

    async def test_date_to_pb_timestamp_1w(self):
        date = datetime.date(2024, 1, 1)
        ts = date_to_pb_timestamp(date)
        self.assertEqual(ts.ToDatetime().date(), date)

    async def test_get_next_quarter(self):
        q1 = "2020Q1"
        self.assertEqual(get_next_quarter(q1), "2020Q2")
        q2 = "2020Q2"
        self.assertEqual(get_next_quarter(q2), "2020Q3")
        q3 = "2020Q3"
        self.assertEqual(get_next_quarter(q3), "2020Q4")
        q4 = "2020Q4"
        self.assertEqual(get_next_quarter(q4), "2021Q1")

    async def test_get_prev_quarter(self):
        q1 = "2020Q1"
        self.assertEqual(get_prev_quarter(q1), "2019Q4")
        q2 = "2020Q2"
        self.assertEqual(get_prev_quarter(q2), "2020Q1")
        q3 = "2020Q3"
        self.assertEqual(get_prev_quarter(q3), "2020Q2")
        q4 = "2020Q4"
        self.assertEqual(get_prev_quarter(q4), "2020Q3")

    async def test_get_year_quarter_for_date(self):
        res = get_year_quarter_for_date(datetime.date(2023, 2, 8))
        self.assertIsInstance(res, tuple)
        self.assertEqual(res, (2023, 1))
        res = get_year_quarter_for_date(datetime.date(1930, 6, 20))
        self.assertIsInstance(res, tuple)
        self.assertEqual(res, (1930, 2))
        res = get_year_quarter_for_date(datetime.date(2024, 7, 1))
        self.assertIsInstance(res, tuple)
        self.assertEqual(res, (2024, 3))
        res = get_year_quarter_for_date(datetime.date(2024, 12, 31))
        self.assertIsInstance(res, tuple)
        self.assertEqual(res, (2024, 4))
