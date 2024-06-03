import datetime
import unittest

from agent_service.io_types.misc import StockID
from agent_service.io_types.table import STOCK_ID_COL_NAME_DEFAULT
from agent_service.tools.feature_data import (
    FeatureDataInput,
    StatisticId,
    StatisticsIdentifierLookupInput,
    get_latest_date,
    get_statistic_data_for_companies,
    statistic_identifier_lookup,
)
from agent_service.types import PlanRunContext

AAPL = StockID(gbi_id=714, isin="", symbol="AAPL")
AMZN = StockID(gbi_id=149, isin="", symbol="AMZN")
MSFT = StockID(gbi_id=6963, isin="", symbol="AAPL")
VZ = StockID(gbi_id=12250, isin="", symbol="VZ")
CLOSE_PRICE = StatisticId(stat_id="spiq_close", stat_name="Close Price")
SPIQ_DIV_AMOUNT = StatisticId(stat_id="spiq_div_amount", stat_name="Dividend Amount")
GROSS_PROFIT = StatisticId(stat_id="spiq_div_amount", stat_name="Dividend Amount")
GLOBAL_CAN_TO_USD_EXCH_RATE = StatisticId(stat_id="FRED_DEXCAUS", stat_name="CAD to USD")


class TestFeatureDataLookup(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.context = PlanRunContext.get_dummy()

    async def test_feature_data_global(self):
        args = FeatureDataInput(
            stock_ids=[],
            statistic_id=GLOBAL_CAN_TO_USD_EXCH_RATE,
        )
        result = await get_statistic_data_for_companies(args, self.context)
        self.assertEqual(len(result.data["Date"].unique()), 1)  # num_dates

    async def test_feature_data_3_stock(self):
        args = FeatureDataInput(
            stock_ids=[AAPL, AMZN, MSFT],
            statistic_id=CLOSE_PRICE,
        )
        result = await get_statistic_data_for_companies(args, self.context)
        self.assertEqual(len(result.data[STOCK_ID_COL_NAME_DEFAULT].unique()), 3)  # num_stocks
        self.assertEqual(len(result.data["Date"].unique()), 1)  # num_dates

    async def test_feature_data_1_stock_many_dates(self):
        args = FeatureDataInput(
            stock_ids=[AAPL],
            statistic_id=CLOSE_PRICE,
            start_date=datetime.date(2020, 1, 1),
            end_date=datetime.date(2020, 3, 1),
        )
        result = await get_statistic_data_for_companies(args, self.context)
        self.assertEqual(len(result.data[STOCK_ID_COL_NAME_DEFAULT].unique()), 1)  # num_stocks
        self.assertGreater(len(result.data["Date"].unique()), 20)  # num_dates

    async def test_get_latest_date(self):
        # test doesnt throw and returns the same value
        val1 = get_latest_date()
        val2 = get_latest_date()
        self.assertEqual(val1, val2)

    async def test_feature_data_weekend(self):
        args = FeatureDataInput(
            stock_ids=[AAPL],
            statistic_id=CLOSE_PRICE,
            start_date=datetime.date(2024, 4, 14),  # sunday
            end_date=datetime.date(2024, 4, 14),
        )
        result = await get_statistic_data_for_companies(args, self.context)

        self.assertEqual(len(result.data[STOCK_ID_COL_NAME_DEFAULT].unique()), 1)  # num_stocks
        self.assertEqual(len(result.data["Date"].unique()), 1)  # num_dates

    async def test_feature_data_dividend(self):
        args = FeatureDataInput(stock_ids=[VZ], statistic_id=SPIQ_DIV_AMOUNT)
        result = await get_statistic_data_for_companies(args, self.context)

        self.assertEqual(len(result.data[STOCK_ID_COL_NAME_DEFAULT].unique()), 1)  # num_stocks
        self.assertEqual(len(result.data["Date"].unique()), 1)  # num_dates

        args = FeatureDataInput(
            stock_ids=[VZ],
            statistic_id=SPIQ_DIV_AMOUNT,
            start_date=datetime.date(2023, 4, 14),
            end_date=datetime.date(2024, 4, 14),
        )
        result = await get_statistic_data_for_companies(args, self.context)

        self.assertGreater(len(result.data["Date"].unique()), 3)  # num_dates

    async def test_feature_data_quarterly(self):
        args = FeatureDataInput(stock_ids=[VZ], statistic_id=GROSS_PROFIT)
        result = await get_statistic_data_for_companies(args, self.context)

        self.assertEqual(len(result.data[STOCK_ID_COL_NAME_DEFAULT].unique()), 1)  # num_stocks
        self.assertEqual(len(result.data["Date"].unique()), 1)  # num_dates


class TestStatisticsIdentifierLookup(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.context = PlanRunContext.get_dummy()
        # uncomment for easier debugging
        from agent_service.utils.logs import init_test_logging

        init_test_logging()

    async def test_statistic_identifier_lookup_conv(self):
        self.args = StatisticsIdentifierLookupInput(statistic_name="bank prime rate")
        result = await statistic_identifier_lookup(self.args, self.context)
        self.assertEqual(result.stat_id, "FRED_DPRIME")

    async def test_statistic_identifier_lookup_price(self):
        self.args = StatisticsIdentifierLookupInput(statistic_name="Price")
        result = await statistic_identifier_lookup(self.args, self.context)
        self.assertEqual(result.stat_id, "spiq_close")

    async def test_statistic_identifier_lookup_highprice(self):
        self.args = StatisticsIdentifierLookupInput(statistic_name="High Price")
        result = await statistic_identifier_lookup(self.args, self.context)
        self.assertEqual(result.stat_id, "spiq_high")

    async def test_statistic_identifier_lookup_basiceps(self):
        self.args = StatisticsIdentifierLookupInput(statistic_name="Basic EPS")
        result = await statistic_identifier_lookup(self.args, self.context)
        self.assertEqual(result.stat_id, "spiq_9")

    async def test_statistic_identifier_lookup_eps(self):
        self.args = StatisticsIdentifierLookupInput(statistic_name="EPS")
        result = await statistic_identifier_lookup(self.args, self.context)
        self.assertEqual(result.stat_id, "spiq_9")

    async def test_statistic_identifier_lookup_doesnt_exist(self):
        self.args = StatisticsIdentifierLookupInput(statistic_name="Foo bar ratio")
        with self.assertRaises(ValueError):
            await statistic_identifier_lookup(self.args, self.context)

    async def test_statistic_identifier_lookup_doesnt_exist_pe(self):
        # eventually this should be changed to expect it to find something correct
        self.args = StatisticsIdentifierLookupInput(statistic_name="pe ratio")
        with self.assertRaises(ValueError):
            await statistic_identifier_lookup(self.args, self.context)

    async def test_statistic_identifier_lookup_bid_price(self):
        self.args = StatisticsIdentifierLookupInput(statistic_name="Bid Price")
        result = await statistic_identifier_lookup(self.args, self.context)
        self.assertEqual(result.stat_id, "spiq_bid")
