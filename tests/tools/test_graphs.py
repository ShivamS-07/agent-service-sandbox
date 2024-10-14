import datetime
import unittest

import pandas as pd

from agent_service.io_type_utils import TableColumnType
from agent_service.io_types.graph import (
    BarData,
    BarDataPoint,
    BarGraph,
    DataPoint,
    GraphDataset,
    GraphType,
    LineGraph,
    PieGraph,
    PieSection,
)
from agent_service.io_types.stock import StockID
from agent_service.io_types.table import Table, TableColumnMetadata
from agent_service.tools.graphs import (
    MakeBarGraphArgs,
    MakeGenericGraphArgs,
    MakeLineGraphArgs,
    MakePieGraphArgs,
    make_bar_graph,
    make_generic_graph,
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

TIMESERIES_TABLE_NO_DATASET = Table.from_df_and_cols(
    columns=[
        TableColumnMetadata(label="Quarter", col_type=TableColumnType.QUARTER),
        TableColumnMetadata(
            label="Value",
            col_type=TableColumnType.FLOAT,
        ),
    ],
    data=pd.DataFrame(
        data=[
            ["2023Q1", 1],
            ["2023Q2", 2],
            ["2023Q3", 3],
        ],
        columns=["Quarter", "Value"],
    ),
)

CATEGORICAL_TABLE_NO_DATASET = Table.from_df_and_cols(
    columns=[
        TableColumnMetadata(label="Stock ID", col_type=TableColumnType.STOCK),
        TableColumnMetadata(
            label="Value",
            col_type=TableColumnType.STRING,
        ),
    ],
    data=pd.DataFrame(
        data=[
            [STOCK1, "X"],
            [STOCK1, "Y"],
            [STOCK1, "Z"],
            [STOCK2, "X"],
            [STOCK2, "Z"],
            [STOCK3, "Z"],
            [STOCK3, "Y"],
        ],
        columns=["Stock ID", "Value"],
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
            ),
            (
                MakePieGraphArgs(
                    input_table=TIMESERIES_TABLE_NO_DATASET,
                ),
                PieGraph(
                    label_type=TableColumnType.QUARTER,
                    data_type=TableColumnType.FLOAT,
                    data=[
                        PieSection(label="2023Q1", value=1),
                        PieSection(label="2023Q2", value=2),
                        PieSection(label="2023Q3", value=3),
                    ],
                ),
            ),
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
            ),
            (
                MakeLineGraphArgs(input_table=TIMESERIES_TABLE_NO_DATASET),
                LineGraph(
                    x_axis_type=TableColumnType.QUARTER,
                    y_axis_type=TableColumnType.FLOAT,
                    data=[
                        GraphDataset(
                            dataset_id_type=TableColumnType.STRING,
                            dataset_id="Value",
                            points=[
                                DataPoint(x_val="2023Q1", y_val=1),
                                DataPoint(x_val="2023Q2", y_val=2),
                                DataPoint(x_val="2023Q3", y_val=3),
                            ],
                        ),
                    ],
                ),
            ),
        ]

        for args, expected in cases:
            actual = await make_line_graph(args, PlanRunContext.get_dummy())
            self.assertEqual(actual, expected)

    async def test_bar_graph(self):
        cases = [
            (
                MakeBarGraphArgs(input_table=TIMESERIES_TABLE),
                BarGraph(
                    data_type=TableColumnType.FLOAT,
                    index_type=TableColumnType.DATE,
                    label_type=TableColumnType.STOCK,
                    data=[
                        BarData(
                            index=datetime.date(2024, 1, 1),
                            values=[
                                BarDataPoint(label=STOCK1, value=1),
                                BarDataPoint(label=STOCK2, value=2),
                                BarDataPoint(label=STOCK3, value=3),
                            ],
                        ),
                        BarData(
                            index=datetime.date(2024, 1, 2),
                            values=[
                                BarDataPoint(label=STOCK1, value=3),
                                BarDataPoint(label=STOCK2, value=4),
                                BarDataPoint(label=STOCK3, value=5),
                            ],
                        ),
                    ],
                ),
            ),
            (
                MakeBarGraphArgs(input_table=TIMESERIES_TABLE_NO_DATASET),
                BarGraph(
                    data_type=TableColumnType.FLOAT,
                    index_type=TableColumnType.QUARTER,
                    data=[
                        BarData(
                            index="2023Q1",
                            values=[
                                BarDataPoint(label="Value", value=1),
                            ],
                        ),
                        BarData(
                            index="2023Q2",
                            values=[
                                BarDataPoint(label="Value", value=2),
                            ],
                        ),
                        BarData(
                            index="2023Q3",
                            values=[
                                BarDataPoint(label="Value", value=3),
                            ],
                        ),
                    ],
                ),
            ),
            (
                MakeBarGraphArgs(input_table=CATEGORICAL_TABLE_NO_DATASET),
                BarGraph(
                    data_type=TableColumnType.INTEGER,
                    index_type=TableColumnType.STRING,
                    data=[
                        BarData(index="X", values=[BarDataPoint(label="Count", value=2)]),
                        BarData(index="Y", values=[BarDataPoint(label="Count", value=2)]),
                        BarData(index="Z", values=[BarDataPoint(label="Count", value=3)]),
                    ],
                ),
            ),
        ]
        for i, (args, expected) in enumerate(cases):
            actual = await make_bar_graph(args, PlanRunContext.get_dummy())
            with self.subTest(f"Case {i + 1}"):
                self.assertEqual(actual, expected)

    async def test_generic_graph(self):
        prefer_pie = Table.from_df_and_cols(
            columns=[
                TableColumnMetadata(label="Quarter", col_type=TableColumnType.QUARTER),
                TableColumnMetadata(
                    label="Value",
                    col_type=TableColumnType.FLOAT,
                ),
            ],
            data=pd.DataFrame(
                data=[
                    ["2023Q1", 1],
                    ["2023Q2", 2],
                    ["2023Q3", 3],
                ],
                columns=["Quarter", "Value"],
            ),
        )
        prefer_pie.prefer_graph_type = GraphType.PIE

        cases = [
            (TIMESERIES_TABLE_NO_DATASET, BarGraph),
            (TIMESERIES_TABLE, LineGraph),
            (prefer_pie, PieGraph),
        ]
        for table, expect_type in cases:
            args = MakeGenericGraphArgs(input_table=table)
            actual = await make_generic_graph(args, PlanRunContext.get_dummy())
            self.assertIsInstance(actual, expect_type)
