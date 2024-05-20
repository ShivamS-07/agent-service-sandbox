import datetime
from unittest import IsolatedAsyncioTestCase

from agent_service.tools.feature_data import (
    FeatureDataInput,
    StatisticsIdentifierLookupInput,
    get_latest_date,
    get_statistic_data_for_companies,
    statistic_identifier_lookup,
)
from agent_service.types import PlanRunContext

AAPL = 714
AMZN = 149
MSFT = 6963
VZ = 12250
CLOSE_PRICE = "spiq_close"
SPIQ_DIV_AMOUNT = "spiq_div_amount"
GROSS_PROFIT = "spiq_10"


class TestFeatureDataLookup(IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.context = PlanRunContext.get_dummy()

    async def test_feature_data_3_stock(self):
        args = FeatureDataInput(stock_ids=[AAPL, AMZN, MSFT], field_id=CLOSE_PRICE)
        result = await get_statistic_data_for_companies(args, self.context)
        self.assertEqual(result.val.shape[1], 3)  # num_stocks
        self.assertEqual(result.val.shape[0], 1)  # num_dates

    async def test_feature_data_1_stock_many_dates(self):
        args = FeatureDataInput(
            stock_ids=[AAPL],
            field_id=CLOSE_PRICE,
            start_date=datetime.date(2020, 1, 1),
            end_date=datetime.date(2020, 3, 1),
        )
        result = await get_statistic_data_for_companies(args, self.context)
        self.assertEqual(result.val.shape[1], 1)  # num_stocks
        self.assertGreater(result.val.shape[0], 20)  # num_dates

    async def test_get_latest_date(self):
        # test doesnt throw and returns the same value
        val1 = get_latest_date()
        val2 = get_latest_date()
        self.assertEqual(val1, val2)

    async def test_feature_data_weekend(self):
        args = FeatureDataInput(
            stock_ids=[AAPL],
            field_id=CLOSE_PRICE,
            start_date=datetime.date(2024, 4, 14),  # sunday
            end_date=datetime.date(2024, 4, 14),
        )
        result = await get_statistic_data_for_companies(args, self.context)

        self.assertEqual(result.val.shape[1], 1)  # num_stocks
        self.assertEqual(result.val.shape[0], 1)  # num_dates

    async def test_feature_data_dividend(self):
        args = FeatureDataInput(stock_ids=[VZ], field_id=SPIQ_DIV_AMOUNT)
        result = await get_statistic_data_for_companies(args, self.context)

        self.assertEqual(result.val.shape[1], 1)  # num_stocks
        self.assertEqual(result.val.shape[0], 1)  # num_dates

        args = FeatureDataInput(
            stock_ids=[VZ],
            field_id=SPIQ_DIV_AMOUNT,
            start_date=datetime.date(2023, 4, 14),
            end_date=datetime.date(2024, 4, 14),
        )
        result = await get_statistic_data_for_companies(args, self.context)

        self.assertGreater(result.val.shape[0], 3)  # num_dates

    async def test_feature_data_quarterly(self):

        args = FeatureDataInput(stock_ids=[VZ], field_id=GROSS_PROFIT)
        result = await get_statistic_data_for_companies(args, self.context)

        self.assertEqual(result.val.shape[1], 1)  # num_stocks
        self.assertEqual(result.val.shape[0], 1)  # num_dates


class TestStatisticsIdentifierLookup(IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.context = PlanRunContext.get_dummy()

    async def test_statistic_identifier_lookup_highprice(self):
        self.args = StatisticsIdentifierLookupInput(statistic_name="High Price")
        result = await statistic_identifier_lookup(self.args, self.context)
        self.assertEqual(result, "spiq_high")

    async def test_statistic_identifier_lookup_basiceps(self):
        self.args = StatisticsIdentifierLookupInput(statistic_name="Basic EPS")
        result = await statistic_identifier_lookup(self.args, self.context)
        self.assertEqual(result, "spiq_9")

    async def test_statistic_identifier_lookup_bollinger(self):
        self.args = StatisticsIdentifierLookupInput(statistic_name="Bid Price")
        result = await statistic_identifier_lookup(self.args, self.context)
        self.assertEqual(result, "spiq_bid")
