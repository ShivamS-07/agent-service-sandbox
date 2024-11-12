import datetime
import unittest

from agent_service.io_type_utils import HistoryEntry, TableColumnType
from agent_service.io_types.stock import StockID
from agent_service.io_types.table import StockTable, Table, TableColumnMetadata
from agent_service.tools.tables import (
    CreateTableStockListArgs,
    GetStockListFromTableArgs,
    JoinStockListTableArgs,
    JoinTableArgs,
    create_table_from_stock_list,
    get_stock_identifier_list_from_table,
    join_stock_list_to_table,
    join_tables,
)
from agent_service.types import PlanRunContext
from tests.tools.table_data import (
    STOCK4,
    STOCK5,
    STOCK6,
    TEST_STOCK_DATE_TABLE1,
    TEST_STOCK_DATE_TABLE2,
    TEST_STOCK_MONTH_TABLE,
    TEST_STOCK_QTR_TABLE,
    TEST_STOCK_TABLE1,
    TEST_STOCK_TABLE2,
    TEST_STOCK_YEAR_TABLE,
    TEST_STRING_DATE_TABLE1,
    TEST_STRING_DATE_TABLE2,
)


class TestTableTools(unittest.IsolatedAsyncioTestCase):
    async def test_join_stock_date_tables(self):
        args = JoinTableArgs(input_tables=[TEST_STOCK_DATE_TABLE1, TEST_STOCK_DATE_TABLE2])
        result: Table = await join_tables(args=args, context=PlanRunContext.get_dummy())

        self.assertEqual(
            [col.metadata for col in result.columns],
            [
                TableColumnMetadata(label="Date", col_type=TableColumnType.DATE, unit=None),
                TableColumnMetadata(label="Security", col_type=TableColumnType.STOCK, unit=None),
                TableColumnMetadata(label="Close Price", col_type=TableColumnType.FLOAT, unit=None),
                TableColumnMetadata(label="Open Price", col_type=TableColumnType.FLOAT, unit=None),
            ],
        )
        self.assertEqual(result.get_num_rows(), 24)
        self.assertEqual(len(result.columns), 4)
        df = result.to_df()
        self.assertEqual(set((sec.gbi_id for sec in df["Security"])), {72, 76, 78, 112, 124, 149})

    async def test_join_string_date_tables(self):
        args = JoinTableArgs(input_tables=[TEST_STRING_DATE_TABLE1, TEST_STRING_DATE_TABLE2])
        result: Table = await join_tables(args=args, context=PlanRunContext.get_dummy())

        self.assertEqual(
            [col.metadata for col in result.columns],
            [
                TableColumnMetadata(label="Date", col_type=TableColumnType.DATE, unit=None),
                TableColumnMetadata(label="Security", col_type=TableColumnType.STRING, unit=None),
                TableColumnMetadata(label="Close Price", col_type=TableColumnType.FLOAT, unit=None),
                TableColumnMetadata(label="Open Price", col_type=TableColumnType.FLOAT, unit=None),
            ],
        )
        self.assertEqual(result.get_num_rows(), 24)
        self.assertEqual(len(result.columns), 4)

    async def test_join_tables_vertically(self):
        args = JoinTableArgs(
            input_tables=[TEST_STRING_DATE_TABLE1, TEST_STRING_DATE_TABLE2], row_join=True
        )
        result: Table = await join_tables(args=args, context=PlanRunContext.get_dummy())

        self.assertEqual(
            [col.metadata.label for col in result.columns],
            ["Date", "Security", "Close Price"],
        )
        self.assertEqual(result.get_num_rows(), 27)
        self.assertEqual(len(result.columns), 3)

    async def test_join_stock_tables(self):
        args = JoinTableArgs(input_tables=[TEST_STOCK_TABLE1, TEST_STOCK_TABLE2])
        result: Table = await join_tables(args=args, context=PlanRunContext.get_dummy())
        self.assertEqual(
            [col.metadata for col in result.columns],
            [
                TableColumnMetadata(label="Security", col_type=TableColumnType.STOCK, unit=None),
                TableColumnMetadata(
                    label="News Summary", col_type=TableColumnType.STRING, unit=None
                ),
                TableColumnMetadata(
                    label="Earnings Summary", col_type=TableColumnType.STRING, unit=None
                ),
            ],
        )
        self.assertEqual(result.get_num_rows(), 3)
        self.assertEqual(len(result.columns), 3)
        df = result.to_df()
        # Make sure the history merged correctly
        self.assertEqual(
            df["Security"][0],
            StockID(
                history=[
                    HistoryEntry(
                        explanation="Test 1",
                        title="",
                        entry_type=TableColumnType.STRING,
                        unit=None,
                        citations=[],
                    ),
                    HistoryEntry(
                        explanation="Test 2",
                        title="",
                        entry_type=TableColumnType.STRING,
                        unit=None,
                        citations=[],
                    ),
                ],
                gbi_id=112,
                symbol="GOOG",
                isin="",
                company_name="",
            ),
        )
        self.assertEqual(set((sec.gbi_id for sec in df["Security"])), {112, 124, 149})

    async def test_get_stock_identifier_list_from_table(self):
        args = GetStockListFromTableArgs(input_table=TEST_STOCK_TABLE1)
        result = await get_stock_identifier_list_from_table(
            args=args, context=PlanRunContext.get_dummy()
        )
        self.assertEqual([stock.gbi_id for stock in result], [112, 124, 149])
        first_stock = result[0]
        self.assertEqual(
            first_stock.history,
            [
                HistoryEntry(explanation="Test 1", title="Test"),
                HistoryEntry(title="News Summary", explanation="blah1"),
            ],
        )

    async def test_create_table_from_stock_list(self):
        stocks = [STOCK4, STOCK5, STOCK6]
        result = await create_table_from_stock_list(
            args=CreateTableStockListArgs(stock_list=stocks), context=PlanRunContext.get_dummy()
        )
        self.assertEqual(len(result.columns), 2)
        self.assertEqual(result.columns[1].metadata.label, "Test")

    async def test_join_stock_list_to_table(self):
        result = await join_stock_list_to_table(
            args=JoinStockListTableArgs(
                input_table=StockTable(columns=TEST_STOCK_TABLE1.columns),
                stock_list=[STOCK4, STOCK5, STOCK6],
            ),
            context=PlanRunContext.get_dummy(),
        )
        self.assertEqual(len(result.columns), 3)
        self.assertEqual(result.columns[2].metadata.label, "Test")

    async def test_join_different_date_types(self):
        with self.subTest("Date and Quarter"):
            args = JoinTableArgs(input_tables=[TEST_STOCK_DATE_TABLE1, TEST_STOCK_QTR_TABLE])
            result: Table = await join_tables(args=args, context=PlanRunContext.get_dummy())
            self.assertEqual(len(result.columns), 3)
            self.assertIsInstance(result.to_df()["Date"][0], datetime.date)

        with self.subTest("Date and Year"):
            args = JoinTableArgs(input_tables=[TEST_STOCK_DATE_TABLE1, TEST_STOCK_YEAR_TABLE])
            result: Table = await join_tables(args=args, context=PlanRunContext.get_dummy())
            self.assertEqual(len(result.columns), 4)
            self.assertIsInstance(result.to_df()["Date"][0], datetime.date)

        with self.subTest("Year and Month"):
            args = JoinTableArgs(input_tables=[TEST_STOCK_YEAR_TABLE, TEST_STOCK_MONTH_TABLE])
            result: Table = await join_tables(args=args, context=PlanRunContext.get_dummy())
            self.assertEqual(len(result.columns), 3)
            self.assertIsInstance(result.to_df()["Month"][0], str)
