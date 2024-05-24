import pandas as pd

from agent_service.io_types.graph import (
    DataPoint,
    GraphDataset,
    LineGraph,
    PieGraph,
    PieSection,
)
from agent_service.io_types.table import Table, TableColumn, TableColumnType
from agent_service.tool import ToolArgs, tool
from agent_service.types import PlanRunContext


class MakeLineGraphArgs(ToolArgs):
    input_table: Table


@tool(
    description="""This function creates a line graph from an input Table.
Line graphs should be created when the user specifically asks for them. If the
user is not specific, line graphs are best created with data that changes over
time, for example things like stock prices or other numerical data. In cases
where you need to make a graph from a Table with data like that, this use this
function. (Ideally the table index should be dates or some other type
representing time.) It is extremely important that you provide enough data to
graph. E.g. at least 7-14 datapoints to make a nice line. Note that the input
must be a Table!
"""
)
async def make_line_graph(args: MakeLineGraphArgs, context: PlanRunContext) -> LineGraph:
    cols = args.input_table.columns
    if len(cols) < 2:
        raise RuntimeError("Table must have at least an index and one column to make a line graph!")
    if len(args.input_table.data) < 2:
        raise RuntimeError("Need at least two points to make a line graph!")
    index_col = cols[0]
    if not index_col.is_indexed:
        raise RuntimeError("Table without index cannot be converted to a line graph!")

    first_y_axis_col = None
    cols_to_graph = []
    for col, df_col in zip(cols[1:], args.input_table.data.columns):
        # Keep adding columns while they're the same types as the first graphable column
        if first_y_axis_col is None and col.col_type in (
            TableColumnType.CURRENCY,
            TableColumnType.FLOAT,
            TableColumnType.INTEGER,
            TableColumnType.DELTA,
            TableColumnType.PCT_DELTA,
            TableColumnType.PERCENT,
        ):
            first_y_axis_col = col
        if first_y_axis_col and col.col_type != first_y_axis_col.col_type:
            continue
        cols_to_graph.append((col, df_col))

    if not cols_to_graph or not first_y_axis_col:
        raise RuntimeError(f"No numerical columns found, cannot create line graph: {cols}")

    data = []
    for col, df_col in cols_to_graph:
        points = [
            DataPoint(x_val=idx, y_val=val)  # type: ignore
            for idx, val in args.input_table.data[df_col].items()
        ]
        data.append(
            GraphDataset(
                dataset_id=(str(col.label)),
                points=points,
                dataset_stock_id=col.label_stock_id,
            )
        )

    return LineGraph(
        x_axis_type=index_col.col_type,
        x_unit=index_col.unit,
        y_axis_type=first_y_axis_col.col_type,
        y_unit=first_y_axis_col.unit,
        data=data,
    )


class MakePieGraphArgs(ToolArgs):
    input_table: Table


@tool(
    description="""This function creates a pie graph from an input Table.
Pie chart graphs should be created when the user specifically asks for them. If
the user is not specific, pie graphs are best created from simple one
dimensional data, or data that has categories. For example, a table containing
the country of origin for a group of stocks, where the stocks could be grouped
by country and presented in the chart. You should MAKE SURE the table ONLY has
the necessary data before calling this, so we don't graph the wrong thing!
(Ideally a single index and a single column!)
Note that the input must be a Table!
"""
)
async def make_pie_graph(args: MakePieGraphArgs, context: PlanRunContext) -> PieGraph:
    cols = args.input_table.columns
    if len(cols) < 2:
        raise RuntimeError("Must have at least index and one column to create pie chart!")
    # Do some analysis on the table to convert it to the best-fit graph.
    col_labels_are_stocks = True
    all_col_types_are_same = True
    index_col = None
    prev_col_type = None
    for col in cols:
        if not index_col and col.is_indexed:
            index_col = col
            continue
        if not prev_col_type:
            prev_col_type = col.col_type
        else:
            if col.col_type != prev_col_type:
                all_col_types_are_same = False

        if not col.col_label_is_stock_id:
            # As soon as we hit a column that's not an index or a stock, set to False
            col_labels_are_stocks = False

    df = args.input_table.data
    if (
        col_labels_are_stocks
        and all_col_types_are_same
        and (
            not index_col or index_col.col_type in (TableColumnType.DATE, TableColumnType.DATETIME)
        )
    ):
        # Essentially this handles creating pie charts out of raw feature
        # tables, which we should support.

        # Create a new dataframe to represent the data we want to chart. New
        # index are the stocks, one data column is the last row. Using the last
        # row since it's often the "latest" or "most recent".
        df = pd.DataFrame(index=df.columns, data=df.iloc[-1])
        index_col = TableColumn(label="Stocks", col_type=TableColumnType.STOCK, is_indexed=True)
        data_col = TableColumn(label="Data", col_type=cols[1].col_type, unit=cols[1].unit)
        cols = [index_col, data_col]

    if not index_col:
        index_col = cols[0]
    for col, df_col in zip(cols[1:], df.columns):
        # Now, just get the next column that will work and use that. For now
        # just ignore if there are multiple columns.

        if col.col_type in (
            TableColumnType.PERCENT,
            TableColumnType.FLOAT,
            TableColumnType.INTEGER,
            TableColumnType.CURRENCY,
        ):
            # We have numerical data. Categories are simply the index, no
            # counting required.
            return PieGraph(
                label_type=index_col.col_type,
                data_type=col.col_type,
                unit=col.unit,
                data=[PieSection(label=str(idx), value=val) for idx, val in df[df_col].items()],
            )
        if col.col_type in (TableColumnType.STRING, TableColumnType.STOCK, TableColumnType.BOOLEAN):
            # We have categorical data, countin is required
            return PieGraph(
                label_type=index_col.col_type,
                data_type=col.col_type,
                unit=col.unit,
                data=[
                    PieSection(label=str(idx), value=val)
                    for idx, val in df[df_col].value_counts().items()
                ],
            )

    raise RuntimeError("Input table cannot be converted to a pie chart!")
