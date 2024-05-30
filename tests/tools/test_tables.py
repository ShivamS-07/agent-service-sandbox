import unittest

from agent_service.io_types.table import Table, TableColumn, TableColumnType
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
            result.columns,
            [
                TableColumn(label="Date", col_type=TableColumnType.DATE, unit=None),
                TableColumn(label="Security", col_type=TableColumnType.STOCK, unit=None),
                TableColumn(label="Close Price", col_type=TableColumnType.FLOAT, unit=None),
                TableColumn(label="Open Price", col_type=TableColumnType.FLOAT, unit=None),
            ],
        )
        self.assertEqual(len(result.data), 27)
        self.assertEqual(len(result.data.columns), 4)
        self.assertEqual(set(result.data["Security"]), {112, 124, 149, 72, 76, 78})

    async def test_join_stock_tables(self):
        args = JoinTableArgs(input_tables=[TEST_STOCK_TABLE1, TEST_STOCK_TABLE2])
        result: Table = await join_tables(args=args, context=PlanRunContext.get_dummy())
        self.assertEqual(
            result.columns,
            [
                TableColumn(label="Security", col_type=TableColumnType.STOCK, unit=None),
                TableColumn(label="News Summary", col_type=TableColumnType.STRING, unit=None),
                TableColumn(label="Earnings Summary", col_type=TableColumnType.STRING, unit=None),
            ],
        )
        self.assertEqual(len(result.data), 3)
        self.assertEqual(len(result.data.columns), 3)
        self.assertEqual(set(result.data["Security"]), {112, 124, 149})

    async def test_get_stock_identifier_list_from_table(self):
        args = GetStockListFromTableArgs(input_table=TEST_STOCK_TABLE1)
        result = await get_stock_identifier_list_from_table(
            args=args, context=PlanRunContext.get_dummy()
        )
        self.assertEqual(result, [112, 124, 149])
