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
from agent_service.io_types.table import Table, TableColumn, TableColumnType
from agent_service.tools.graphs import (
    MakeLineGraphArgs,
    MakePieGraphArgs,
    make_line_graph,
    make_pie_graph,
)
from agent_service.types import PlanRunContext

TIMESERIES_TABLE = Table(
    columns=[
        TableColumn(label="Date", col_type=TableColumnType.DATE, is_indexed=True),
        TableColumn(
            label="AAPL",
            col_type=TableColumnType.FLOAT,
            is_indexed=True,
            label_stock_id=714,
            col_label_is_stock_id=True,
        ),
        TableColumn(
            label="MSFT",
            col_type=TableColumnType.FLOAT,
            is_indexed=True,
            label_stock_id=6963,
            col_label_is_stock_id=True,
        ),
        TableColumn(
            label="TSLA",
            col_type=TableColumnType.FLOAT,
            is_indexed=True,
            label_stock_id=1,
            col_label_is_stock_id=True,
        ),
    ],
    data=pd.DataFrame(
        data=[[1, 2, 3], [3, 4, 5]],
        index=[datetime.date(2024, 1, 1), datetime.date(2024, 1, 2)],
        columns=["AAPL", "MSFT", "TSLA"],
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
                        PieSection(label="AAPL", value=3),
                        PieSection(label="MSFT", value=4),
                        PieSection(label="TSLA", value=5),
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
                            dataset_id="AAPL",
                            dataset_stock_id=714,
                            points=[
                                DataPoint(x_val=datetime.date(2024, 1, 1), y_val=1),
                                DataPoint(x_val=datetime.date(2024, 1, 2), y_val=3),
                            ],
                        ),
                        GraphDataset(
                            dataset_id="MSFT",
                            dataset_stock_id=6963,
                            points=[
                                DataPoint(x_val=datetime.date(2024, 1, 1), y_val=2),
                                DataPoint(x_val=datetime.date(2024, 1, 2), y_val=4),
                            ],
                        ),
                        GraphDataset(
                            dataset_id="TSLA",
                            dataset_stock_id=1,
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
