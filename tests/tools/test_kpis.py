import logging
from typing import List
from unittest import IsolatedAsyncioTestCase

from agent_service.io_types.stock import StockID
from agent_service.io_types.table import Table
from agent_service.io_types.text import KPIText
from agent_service.tools.dates import DateRangeInput, get_n_width_date_range_near_date
from agent_service.tools.kpis import (
    CompanyKPIsRequest,
    EquivalentKPITexts,
    GetImportantKPIsForStock,
    GetRelevantKPIsForStockGivenTopic,
    GetRelevantKPIsForStocksGivenTopic,
    KPIsRequest,
    get_important_kpis_for_stock,
    get_kpis_table_for_stock,
    get_overlapping_kpis_table_for_stock,
    get_relevant_kpis_for_multiple_stocks_given_topic,
    get_relevant_kpis_for_stock_given_topic,
)
from agent_service.types import PlanRunContext
from agent_service.utils.date_utils import get_now_utc


class TestTextData(IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        logging.getLogger("asyncio").setLevel(logging.ERROR)

    async def asyncSetUp(self):
        self.context = PlanRunContext.get_dummy()

    async def test_kpi_defaults(self):
        stock_id = StockID(gbi_id=714, symbol="APPL", isin="", company_name="")

        gen_kpi_list: List[KPIText] = await get_important_kpis_for_stock(  # type: ignore
            args=GetImportantKPIsForStock(stock_id=stock_id), context=self.context
        )
        topic_kpi_list: List[KPIText] = await get_relevant_kpis_for_stock_given_topic(  # type: ignore
            GetRelevantKPIsForStockGivenTopic(stock_id=stock_id, topic="Apple TV"),
            context=self.context,
        )

        gen_kpis_table: Table = await get_kpis_table_for_stock(  # type: ignore
            CompanyKPIsRequest(
                stock_id=stock_id, table_name="General KPI Table", kpis=gen_kpi_list
            ),
            context=self.context,
        )
        self.assertGreater(len(gen_kpis_table.columns), 3)

        topic_kpis_table: Table = await get_kpis_table_for_stock(  # type: ignore
            CompanyKPIsRequest(
                stock_id=stock_id, table_name="Topic KPI Table (Apple TV)", kpis=topic_kpi_list
            ),
            context=self.context,
        )

        # default is current day + 7 more quarters
        self.assertEqual(8, topic_kpis_table.get_num_rows())

    async def test_kpi_daterange(self):

        stock_id = StockID(gbi_id=714, symbol="APPL", isin="", company_name="")

        num_quarters = 4
        date_range = await get_n_width_date_range_near_date(
            args=DateRangeInput(range_ending_on=get_now_utc(), width_quarters=num_quarters),
            context=self.context,
        )

        topic_kpi_list: List[KPIText] = await get_relevant_kpis_for_stock_given_topic(  # type: ignore
            GetRelevantKPIsForStockGivenTopic(stock_id=stock_id, topic="Net sales - iPhone Actual"),
            context=self.context,
        )

        topic_kpis_table1: Table = await get_kpis_table_for_stock(  # type: ignore
            CompanyKPIsRequest(
                stock_id=stock_id,
                table_name="Topic KPI Table (iPhones)",
                kpis=topic_kpi_list[:1],
                num_prev_quarters=num_quarters - 1,
            ),
            context=self.context,
        )

        # current quarter + num_quarters prev to that
        self.assertEqual(num_quarters, topic_kpis_table1.get_num_rows())
        self.assertGreater(len(topic_kpis_table1.columns), 2)

        topic_kpis_table2: Table = await get_kpis_table_for_stock(  # type: ignore
            CompanyKPIsRequest(
                stock_id=stock_id,
                table_name="Topic KPI Table (iPhones)",
                kpis=topic_kpi_list[:1],
                date_range=date_range,
            ),
            context=self.context,
        )

        # 4 quarters of total with should be 1 quarter for current date + 3 prev quarters = 4 total
        self.assertEqual(num_quarters, topic_kpis_table2.get_num_rows())
        self.assertEqual(topic_kpis_table1.get_num_rows(), topic_kpis_table2.get_num_rows())
        self.assertEqual(topic_kpis_table1, topic_kpis_table2)

        # 8 quarter wide date range should be equivalent to the default behavior with no args
        num_quarters = 8
        date_range = await get_n_width_date_range_near_date(
            args=DateRangeInput(range_ending_on=get_now_utc(), width_quarters=num_quarters),
            context=self.context,
        )

        topic_kpis_table1: Table = await get_kpis_table_for_stock(  # type: ignore
            CompanyKPIsRequest(
                stock_id=stock_id,
                table_name="Topic KPI Table (iPhones)",
                kpis=topic_kpi_list[:1],
            ),
            context=self.context,
        )

        # current quarter + num_quarters prev to that
        self.assertEqual(num_quarters, topic_kpis_table1.get_num_rows())

        topic_kpis_table2: Table = await get_kpis_table_for_stock(  # type: ignore
            CompanyKPIsRequest(
                stock_id=stock_id,
                table_name="Topic KPI Table (iPhones)",
                kpis=topic_kpi_list[:1],
                date_range=date_range,
            ),
            context=self.context,
        )

        self.assertEqual(num_quarters, topic_kpis_table2.get_num_rows())
        self.assertEqual(topic_kpis_table1.get_num_rows(), topic_kpis_table2.get_num_rows())
        self.assertEqual(topic_kpis_table1, topic_kpis_table2)

    async def test_kpi_multi_stock(self):

        microsoft = StockID(gbi_id=6963, symbol="MSFT", isin="")
        amazon = StockID(gbi_id=149, symbol="AMZN", isin="")
        alphabet = StockID(gbi_id=10096, symbol="GOOG", isin="")

        stocks = [microsoft, amazon, alphabet]
        equivalent_kpis: EquivalentKPITexts = await get_relevant_kpis_for_multiple_stocks_given_topic(  # type: ignore
            GetRelevantKPIsForStocksGivenTopic(stock_ids=stocks, shared_metric="Cloud"),
            context=self.context,
        )

        num_prev_quarters = 4
        equivalent_kpis_table: Table = await get_overlapping_kpis_table_for_stock(  # type: ignore
            args=KPIsRequest(
                equivalent_kpis=equivalent_kpis,
                table_name="Cloud Computing",
                num_future_quarters=0,
                num_prev_quarters=num_prev_quarters,
            ),
            context=self.context,
        )

        self.assertEqual(
            len(stocks) * (num_prev_quarters + 1), equivalent_kpis_table.get_num_rows()
        )