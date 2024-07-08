import datetime
import logging
import unittest
from typing import List
from unittest import IsolatedAsyncioTestCase

from agent_service.io_types.dates import DateRange
from agent_service.io_types.stock import StockID
from agent_service.io_types.table import Table
from agent_service.io_types.text import KPIText
from agent_service.tools.kpis.tools import (
    CompanyKPIsRequest,
    EquivalentKPITexts,
    GetImportantKPIsForStock,
    GetKPIForStockGivenTopic,
    GetRelevantKPIsForStocksGivenTopic,
    KPIsRequest,
    get_important_kpis_for_stock,
    get_kpis_for_stock_given_topics,
    get_kpis_table_for_stock,
    get_overlapping_kpis_table_for_stocks,
    get_relevant_kpis_for_multiple_stocks_given_topic,
)
from agent_service.types import PlanRunContext


@unittest.skip("flaky")
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
        topic_kpi_list: List[KPIText] = await get_kpis_for_stock_given_topics(  # type: ignore
            GetKPIForStockGivenTopic(stock_id=stock_id, topics=["Apple TV"]),
            context=self.context,
        )

        gen_kpis_table: Table = await get_kpis_table_for_stock(  # type: ignore
            CompanyKPIsRequest(
                stock_id=stock_id,
                table_name="General KPI Table",
                kpis=gen_kpi_list,
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

        # Default behavour expects one quarter
        self.assertEqual(1, topic_kpis_table.get_num_rows())

    async def test_kpi_daterange(self):

        stock_id = StockID(gbi_id=714, symbol="APPL", isin="", company_name="")

        num_quarters = 4
        date_range = DateRange(
            start_date=datetime.date.fromisoformat("2023-07-02"),
            end_date=datetime.date.fromisoformat("2024-07-02"),
        )

        topic_kpi_list: List[KPIText] = await get_kpis_for_stock_given_topics(  # type: ignore
            GetKPIForStockGivenTopic(stock_id=stock_id, topics=["Net sales - iPhone Actual"]),
            context=self.context,
        )

        topic_kpis_table: Table = await get_kpis_table_for_stock(  # type: ignore
            CompanyKPIsRequest(
                stock_id=stock_id,
                table_name="Topic KPI Table (iPhones)",
                kpis=topic_kpi_list[:1],
                date_range=date_range,
            ),
            context=self.context,
        )

        # current quarter + num_quarters prev to that
        self.assertEqual(num_quarters, topic_kpis_table.get_num_rows())
        self.assertGreater(len(topic_kpis_table.columns), 2)

        # 8 quarter wide date range
        num_quarters = 8
        date_range = DateRange(
            start_date=datetime.date.fromisoformat("2022-07-02"),
            end_date=datetime.date.fromisoformat("2024-07-02"),
        )

        topic_kpis_table: Table = await get_kpis_table_for_stock(  # type: ignore
            CompanyKPIsRequest(
                stock_id=stock_id,
                table_name="Topic KPI Table (iPhones)",
                kpis=topic_kpi_list[:1],
                date_range=date_range,
            ),
            context=self.context,
        )
        self.assertEqual(num_quarters, topic_kpis_table.get_num_rows())
        self.assertGreater(len(topic_kpis_table.columns), 2)

    async def test_kpi_multi_stock(self):

        microsoft = StockID(gbi_id=6963, symbol="MSFT", isin="")
        amazon = StockID(gbi_id=149, symbol="AMZN", isin="")
        alphabet = StockID(gbi_id=10096, symbol="GOOG", isin="")

        stocks = [microsoft, amazon, alphabet]
        equivalent_kpis: EquivalentKPITexts = await get_relevant_kpis_for_multiple_stocks_given_topic(  # type: ignore
            GetRelevantKPIsForStocksGivenTopic(stock_ids=stocks, shared_metric="Cloud"),
            context=self.context,
        )

        num_prev_quarters = 3
        equivalent_kpis_table: Table = await get_overlapping_kpis_table_for_stocks(  # type: ignore
            args=KPIsRequest(
                equivalent_kpis=equivalent_kpis,
                table_name="Cloud Computing",
                date_range=DateRange(
                    start_date=datetime.date.fromisoformat("2023-07-02"),
                    end_date=datetime.date.fromisoformat("2024-07-02"),
                ),
            ),
            context=self.context,
        )

        self.assertEqual(
            len(stocks) * (num_prev_quarters + 1), equivalent_kpis_table.get_num_rows()
        )
