import datetime
import unittest

import pandas as pd

from agent_service.io_types.graph import (
    DataPoint,
    GraphDataset,
    LineGraph,
    PieGraph,
    PieSection,
)
from agent_service.io_types.stock import StockID
from agent_service.io_types.table import Table, TableColumnMetadata, TableColumnType
from agent_service.tools.graphs import (
    MakeLineGraphArgs,
    MakePieGraphArgs,
    make_line_graph,
    make_pie_graph,
)
from agent_service.types import PlanRunContext

STOCK1 = StockID(gbi_id=1, symbol="", isin="", company_name="")
STOCK2 = StockID(gbi_id=2, symbol="", isin="", company_name="")
STOCK3 = StockID(gbi_id=3, symbol="", isin="", company_name="")

TIMESERIES_TABLE = Table.from_df_and_cols(
    columns=[
        TableColumnMetadata(label="Date", col_type=TableColumnType.DATE),
        TableColumnMetadata(
            label="Stock ID",
            col_type=TableColumnType.STOCK,
        ),
        TableColumnMetadata(
            label="Value",
            col_type=TableColumnType.FLOAT,
        ),
    ],
    data=pd.DataFrame(
        data=[
            [datetime.date(2024, 1, 1), STOCK1, 1],
            [datetime.date(2024, 1, 2), STOCK1, 3],
            [datetime.date(2024, 1, 1), STOCK2, 2],
            [datetime.date(2024, 1, 2), STOCK2, 4],
            [datetime.date(2024, 1, 1), STOCK3, 3],
            [datetime.date(2024, 1, 2), STOCK3, 5],
        ],
        columns=["Date", "Stock ID", "Value"],
    ),
)


class TestGraphTools(unittest.IsolatedAsyncioTestCase):
    async def test_pie_graph(self):
        cases = [
            (
                MakePieGraphArgs(
                    input_table=TIMESERIES_TABLE,
                ),
                PieGraph(
                    label_type=TableColumnType.STOCK,
                    data_type=TableColumnType.FLOAT,
                    data=[
                        PieSection(label=STOCK1, value=3),
                        PieSection(label=STOCK2, value=4),
                        PieSection(label=STOCK3, value=5),
                    ],
                ),
            )
        ]

        for args, expected in cases:
            actual = await make_pie_graph(args, PlanRunContext.get_dummy())
            self.assertEqual(actual, expected)

    async def test_line_graph(self):
        cases = [
            (
                MakeLineGraphArgs(input_table=TIMESERIES_TABLE),
                LineGraph(
                    x_axis_type=TableColumnType.DATE,
                    y_axis_type=TableColumnType.FLOAT,
                    data=[
                        GraphDataset(
                            dataset_id_type=TableColumnType.STOCK,
                            dataset_id=STOCK1,
                            points=[
                                DataPoint(x_val=datetime.date(2024, 1, 1), y_val=1),
                                DataPoint(x_val=datetime.date(2024, 1, 2), y_val=3),
                            ],
                        ),
                        GraphDataset(
                            dataset_id_type=TableColumnType.STOCK,
                            dataset_id=STOCK2,
                            points=[
                                DataPoint(x_val=datetime.date(2024, 1, 1), y_val=2),
                                DataPoint(x_val=datetime.date(2024, 1, 2), y_val=4),
                            ],
                        ),
                        GraphDataset(
                            dataset_id_type=TableColumnType.STOCK,
                            dataset_id=STOCK3,
                            points=[
                                DataPoint(x_val=datetime.date(2024, 1, 1), y_val=3),
                                DataPoint(x_val=datetime.date(2024, 1, 2), y_val=5),
                            ],
                        ),
                    ],
                ),
            )
        ]

        for args, expected in cases:
            actual = await make_line_graph(args, PlanRunContext.get_dummy())
            self.assertEqual(actual, expected)
