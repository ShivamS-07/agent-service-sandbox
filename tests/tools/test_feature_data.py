import datetime
import time
import unittest

from agent_service.io_type_utils import TableColumnType
from agent_service.io_types.stock import StockID
from agent_service.io_types.table import STOCK_ID_COL_NAME_DEFAULT
from agent_service.planner.errors import NotFoundError
from agent_service.tools.feature_data import (
    FeatureDataInput,
    StatisticId,
    StatisticsIdentifierLookupInput,
    get_latest_date,
    get_statistic_data,
    get_statistic_data_for_companies,
    statistic_identifier_lookup,
)
from agent_service.types import PlanRunContext

AAPL = StockID(gbi_id=714, isin="", symbol="AAPL", company_name="")
AMZN = StockID(gbi_id=149, isin="", symbol="AMZN", company_name="")
MSFT = StockID(gbi_id=6963, isin="", symbol="MSFT", company_name="")
VZ = StockID(gbi_id=12250, isin="", symbol="VZ", company_name="")
TOTAL_ASSETS = StatisticId(stat_id="spiq_1007", stat_name="Total Assets")
CLOSE_PRICE = StatisticId(stat_id="spiq_close", stat_name="Close Price")
PE_RATIO = StatisticId(stat_id="pe_ratio", stat_name="P/E Ratio")
SPIQ_DIV_AMOUNT = StatisticId(stat_id="spiq_div_amount", stat_name="Dividend Amount")
GROSS_PROFIT = StatisticId(stat_id="spiq_div_amount", stat_name="Dividend Amount")
GLOBAL_CAN_TO_USD_EXCH_RATE = StatisticId(stat_id="FRED_DEXCAUS", stat_name="CAD to USD")
GLOBAL_FRED_WTI1 = StatisticId(
    stat_id="FRED_DCOILWTICO",
    stat_name="Crude Oil Prices: West Texas Intermediate (WTI) - Cushing, Oklahoma",
)
GLOBAL_FRED_WTI2 = StatisticId(
    stat_id="FRED_WTISPLC", stat_name="Spot Crude Oil Price: West Texas Intermediate (WTI)"
)


class TestFeatureDataLookup1(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.context = PlanRunContext.get_dummy()

    async def test_feature_data_global(self):
        args = FeatureDataInput(
            stock_ids=[],
            statistic_id=GLOBAL_CAN_TO_USD_EXCH_RATE,
        )
        result = await get_statistic_data_for_companies(args, self.context)
        df = result.to_df()
        self.assertEqual(len(df["Date"].unique()), 1)  # num_dates

    async def test_feature_data_global2(self):
        tic = time.perf_counter()
        end_date = datetime.datetime.now(datetime.UTC).date()
        start_date = end_date - datetime.timedelta(days=365.25 * 5)
        # This is daily.
        result = await get_statistic_data(self.context, GLOBAL_FRED_WTI1, start_date, end_date)
        df = result.to_df().dropna()
        toc = time.perf_counter()
        time_total = toc - tic
        print(f"{time_total=}.")
        self.assertTrue(time_total < 30.0)
        self.assertTrue(len(df) > 240)

    async def test_feature_data_global3(self):
        tic = time.perf_counter()
        end_date = datetime.datetime.now(datetime.UTC).date()
        start_date = end_date - datetime.timedelta(days=365.25 * 5)
        # This is monthly.
        result = await get_statistic_data(self.context, GLOBAL_FRED_WTI2, start_date, end_date)
        df = result.to_df().dropna()
        toc = time.perf_counter()
        time_total = toc - tic
        print(f"{time_total=}.")
        self.assertTrue(time_total < 30.0)
        self.assertTrue(len(df) > 50)

    async def test_preset_feature_data_3_stock(self):
        args = FeatureDataInput(
            stock_ids=[AAPL, AMZN, MSFT],
            statistic_id=PE_RATIO,
        )
        result = await get_statistic_data_for_companies(args, self.context)
        df = result.to_df()
        self.assertEqual(len(df[STOCK_ID_COL_NAME_DEFAULT].unique()), 3)  # num_stocks
        self.assertEqual(len(df["Date"].unique()), 1)  # num_dates
        self.assertEqual(result.columns[0].metadata.col_type, TableColumnType.DATE)

    @unittest.skip("Come back to this later")
    async def test_feature_data_3_quarter_axes(self):
        args = FeatureDataInput(
            stock_ids=[AAPL, AMZN, MSFT],
            statistic_id=TOTAL_ASSETS,
        )
        result = await get_statistic_data_for_companies(args, self.context)
        df = result.to_df()
        self.assertEqual(len(df[STOCK_ID_COL_NAME_DEFAULT].unique()), 3)  # num_stocks
        self.assertEqual(len(df["Period"].unique()), 1)  # num_dates
        self.assertEqual(result.columns[0].metadata.col_type, TableColumnType.STRING)

    async def test_feature_data_3_stock(self):
        args = FeatureDataInput(
            stock_ids=[AAPL, AMZN, MSFT],
            statistic_id=CLOSE_PRICE,
        )
        result = await get_statistic_data_for_companies(args, self.context)
        df = result.to_df()
        self.assertEqual(len(df[STOCK_ID_COL_NAME_DEFAULT].unique()), 3)  # num_stocks
        self.assertEqual(len(df["Date"].unique()), 1)  # num_dates

    async def test_feature_data_1_stock_many_dates(self):
        args = FeatureDataInput(
            stock_ids=[AAPL],
            statistic_id=CLOSE_PRICE,
            start_date=datetime.date(2020, 1, 1),
            end_date=datetime.date(2020, 3, 1),
        )
        result = await get_statistic_data_for_companies(args, self.context)
        df = result.to_df()
        self.assertEqual(len(df[STOCK_ID_COL_NAME_DEFAULT].unique()), 1)  # num_stocks
        self.assertGreater(len(df["Date"].unique()), 20)  # num_dates

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

        df = result.to_df()
        self.assertEqual(len(df[STOCK_ID_COL_NAME_DEFAULT].unique()), 1)  # num_stocks
        self.assertEqual(len(df["Date"].unique()), 1)  # num_dates


class TestFeatureDataLookup2(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.context = PlanRunContext.get_dummy()

    async def test_feature_data_dividend(self):
        args = FeatureDataInput(stock_ids=[VZ], statistic_id=SPIQ_DIV_AMOUNT)
        result = await get_statistic_data_for_companies(args, self.context)

        df = result.to_df()
        self.assertEqual(len(df[STOCK_ID_COL_NAME_DEFAULT].unique()), 1)  # num_stocks
        self.assertEqual(len(df["Date"].unique()), 1)  # num_dates

        args = FeatureDataInput(
            stock_ids=[VZ],
            statistic_id=SPIQ_DIV_AMOUNT,
            start_date=datetime.date(2023, 4, 14),
            end_date=datetime.date(2024, 4, 14),
        )
        result = await get_statistic_data_for_companies(args, self.context)
        df = result.to_df()
        self.assertGreater(len(df["Date"].unique()), 3)  # num_dates

    async def test_feature_data_quarterly(self):
        args = FeatureDataInput(stock_ids=[VZ], statistic_id=GROSS_PROFIT)
        result = await get_statistic_data_for_companies(args, self.context)

        df = result.to_df()
        self.assertEqual(len(df[STOCK_ID_COL_NAME_DEFAULT].unique()), 1)  # num_stocks
        self.assertEqual(len(df["Date"].unique()), 1)  # num_dates


class TestStatisticsIdentifierLookup1(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.context = PlanRunContext.get_dummy()
        # uncomment for easier debugging
        from agent_service.utils.logs import init_test_logging

        init_test_logging()

    async def test_statistic_identifier_lookup_global(self):
        self.args = StatisticsIdentifierLookupInput(statistic_name="bank prime rate")
        result = await statistic_identifier_lookup(self.args, self.context)
        self.assertEqual(result.stat_id, "FRED_DPRIME")

        self.args = StatisticsIdentifierLookupInput(statistic_name="CPI")
        result = await statistic_identifier_lookup(self.args, self.context)
        self.assertEqual(result.stat_id, "FRED_CPIAUCSL")

        self.args = StatisticsIdentifierLookupInput(statistic_name="federal interest rate")
        result = await statistic_identifier_lookup(self.args, self.context)
        self.assertEqual(result.stat_id, "FRED_DFF")

    async def test_statistic_identifier_lookup_price(self):
        self.args = StatisticsIdentifierLookupInput(statistic_name="Price")
        result = await statistic_identifier_lookup(self.args, self.context)
        self.assertEqual(result.stat_id, "spiq_close")

    async def test_statistic_identifier_lookup_highprice(self):
        self.args = StatisticsIdentifierLookupInput(statistic_name="High Price")
        result = await statistic_identifier_lookup(self.args, self.context)
        self.assertEqual(result.stat_id, "spiq_high")


class TestStatisticsIdentifierLookup2(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.context = PlanRunContext.get_dummy()
        # uncomment for easier debugging
        from agent_service.utils.logs import init_test_logging

        init_test_logging()

    async def test_statistic_identifier_lookup_basiceps(self):
        self.args = StatisticsIdentifierLookupInput(statistic_name="Basic EPS")
        result = await statistic_identifier_lookup(self.args, self.context)
        self.assertEqual(result.stat_id, "spiq_9")

    @unittest.skip("test is failing, needs to be fixed")
    async def test_statistic_identifier_lookup_eps(self):
        self.args = StatisticsIdentifierLookupInput(statistic_name="EPS")
        result = await statistic_identifier_lookup(self.args, self.context)
        self.assertEqual(result.stat_id, "spiq_9")

    async def test_statistic_identifier_lookup_doesnt_exist(self):
        self.args = StatisticsIdentifierLookupInput(statistic_name="Foo bar ratio")
        with self.assertRaises(NotFoundError):
            await statistic_identifier_lookup(self.args, self.context)

    async def test_statistic_identifier_lookup_pe(self):
        # eventually this should be changed to expect it to find something correct
        self.args = StatisticsIdentifierLookupInput(statistic_name="pe ratio")
        result = await statistic_identifier_lookup(self.args, self.context)
        self.assertEqual(result.stat_id, "pe_ratio")

    async def test_statistic_identifier_lookup_bid_price(self):
        self.args = StatisticsIdentifierLookupInput(statistic_name="Bid Price")
        result = await statistic_identifier_lookup(self.args, self.context)
        self.assertEqual(result.stat_id, "spiq_bid")
