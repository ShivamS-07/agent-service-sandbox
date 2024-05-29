from agent_service.io_types.graph import (
    DataPoint,
    GraphDataset,
    LineGraph,
    PieGraph,
    PieSection,
)
from agent_service.io_types.table import Table, TableColumnType
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
        raise RuntimeError("Table must have at least two columns to make a line graph!")
    if len(args.input_table.data) < 2:
        raise RuntimeError("Need at least two points to make a line graph!")

    x_axis_col = None
    dataset_col = None
    data_col = None
    for col, df_col in zip(cols, args.input_table.data.columns):
        if not x_axis_col and col.col_type in (
            TableColumnType.DATE,
            TableColumnType.DATETIME,
            TableColumnType.INTEGER,
        ):
            x_axis_col = (col, df_col)
            continue

        if not dataset_col and len(cols) > 2:
            # We only have a dataset column if we have > 2 columns. Otherwise we
            # just have the X axis and the value.
            dataset_col = (col, df_col)
            continue

        # Keep adding columns while they're the same types as the first graphable column
        if not data_col and col.col_type in (
            TableColumnType.CURRENCY,
            TableColumnType.FLOAT,
            TableColumnType.INTEGER,
            TableColumnType.DELTA,
            TableColumnType.PCT_DELTA,
            TableColumnType.PERCENT,
        ):
            data_col = (col, df_col)

    if not data_col:
        raise RuntimeError(f"No numerical columns found, cannot create line graph: {cols}")

    if not x_axis_col:
        raise RuntimeError(
            f"Unable to create line graph for table with no possible x axis column: {cols}"
        )

    data = []
    df = args.input_table.data
    y_col, y_df_col = data_col
    x_col, x_df_col = x_axis_col
    if dataset_col is None:
        # In this case, we only have a single dataset so we don't need to do any
        # fancy grouping. The dataset name is simply the name of the graphed
        # column. This only happens when there are exactly two columns, so we
        # can just use the first col to graph.
        dataset = GraphDataset(
            dataset_id=str(y_col.label),
            dataset_id_type=TableColumnType.STRING,
            points=[
                DataPoint(x_val=x_val, y_val=y_val)
                for x_val, y_val in zip(df[x_df_col], df[y_df_col])
                if x_val is not None and y_val is not None
            ],
        )
        data = [dataset]

    else:
        ds_col, dataset_df_col = dataset_col
        grouped_df = df.set_index(dataset_df_col)
        for dataset_val in grouped_df.index.unique():
            # For each unique value in the dataset column, extract data for the
            # graph. For example, if the dataset column contains stock ID's,
            # this will create a dataset for each stock.
            dataset_data = grouped_df.loc[dataset_val]
            dataset = GraphDataset(
                dataset_id=dataset_val,
                dataset_id_type=ds_col.col_type,
                points=[
                    DataPoint(x_val=x_val, y_val=y_val)
                    for x_val, y_val in zip(dataset_data[x_df_col], dataset_data[y_df_col])
                    if x_val is not None and y_val is not None
                ],
            )
            data.append(dataset)

    return LineGraph(
        x_axis_type=x_axis_col[0].col_type,
        x_unit=x_col.unit,
        y_axis_type=y_col.col_type,
        y_unit=y_col.unit,
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
    df = args.input_table.data
    if len(cols) < 2:
        raise RuntimeError("Must have at least index and one column to create pie chart!")
    # Do some analysis on the table to convert it to the best-fit graph.
    label_col = None
    label_df_col = None
    data_col = None
    data_df_col = None
    for col, df_col in zip(cols, df.columns):
        if not label_col and col.col_type in (
            TableColumnType.STOCK,
            TableColumnType.STRING,
        ):
            label_col = col
            label_df_col = df_col
            continue

        if label_col:
            # Once we found a label column, the next column is the data column.
            data_col = col
            data_df_col = df_col
            break

    if not label_col:
        raise RuntimeError("Must have valid label column to make a pie chart!")
    if not data_col or not data_df_col:
        raise RuntimeError("Must have valid data column to make a pie chart!")

    # Get the "most recent" if there are multiple groups
    grouped_df = df.groupby(label_df_col).last()
    if data_col.col_type in (
        TableColumnType.PERCENT,
        TableColumnType.FLOAT,
        TableColumnType.INTEGER,
        TableColumnType.CURRENCY,
    ):
        # We have numerical data. Categories are simply the index, no
        # counting required.
        return PieGraph(
            label_type=label_col.col_type,
            data_type=data_col.col_type,
            unit=data_col.unit,
            data=[
                PieSection(label=idx, value=val)  # type: ignore
                for idx, val in grouped_df[data_df_col].items()
            ],
        )
    if data_col.col_type in (
        TableColumnType.STRING,
        TableColumnType.STOCK,
        TableColumnType.BOOLEAN,
    ):
        # We have categorical data, counting is required
        return PieGraph(
            label_type=data_col.col_type,
            data_type=TableColumnType.INTEGER,
            unit=data_col.unit,
            data=[
                PieSection(label=category, value=count)  # type: ignore
                for category, count in grouped_df[data_df_col].value_counts().items()
            ],
        )

    raise RuntimeError("Input table cannot be converted to a pie chart!")
