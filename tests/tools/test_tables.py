import unittest

from agent_service.io_type_utils import HistoryEntry, TableColumnType
from agent_service.io_types.stock import StockID
from agent_service.io_types.table import Table, TableColumnMetadata
from agent_service.tools.tables import (
    GetStockListFromTableArgs,
    JoinTableArgs,
    get_stock_identifier_list_from_table,
    join_tables,
)
from agent_service.types import PlanRunContext
from tests.tools.table_data import (
    TEST_STOCK_DATE_TABLE1,
    TEST_STOCK_DATE_TABLE2,
    TEST_STOCK_TABLE1,
    TEST_STOCK_TABLE2,
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
        self.assertEqual(result.get_num_rows(), 27)
        self.assertEqual(len(result.columns), 4)
        df = result.to_df()
        self.assertEqual(set((sec.gbi_id for sec in df["Security"])), {112, 124, 149, 72, 76, 78})

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
                symbol="",
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
                HistoryEntry(explanation="Test 1"),
                HistoryEntry(title="News Summary", explanation="blah1"),
            ],
        )
