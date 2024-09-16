from copy import deepcopy
from typing import Any, Dict, List, cast

import pandas as pd

from agent_service.io_type_utils import HistoryEntry, TableColumnType
from agent_service.io_types.graph import (
    BarData,
    BarDataPoint,
    BarGraph,
    DataPoint,
    Graph,
    GraphDataset,
    GraphType,
    LineGraph,
    PieGraph,
    PieSection,
)
from agent_service.io_types.stock import StockID
from agent_service.io_types.table import Table, TableColumn, TableColumnMetadata
from agent_service.io_types.text import KPIText, TextCitation
from agent_service.tool import ToolArgs, tool
from agent_service.types import PlanRunContext
from agent_service.utils.prefect import get_prefect_logger


def consolidate_table_columns(table: Table) -> Table:
    """
    Takes in a table and consolidates Security and Metric columns into 2 Label and Amount columns
    This is used for creating graphs from tables that have multiple Metric columns.
    This results in a table with 3 columns: Date, Label, Amount
    if there are Date and Security columns in the original input table.
    """

    cols = table.columns

    if len(cols) <= 3:
        return table

    date_col = None
    security_col = None
    metric_col = None
    quarter_col = []
    label_col: List[Any] = []
    amount_col = []
    for col in cols:
        if not date_col and col.metadata.col_type.is_date_type():
            date_col = col
            continue
        if not security_col and col.metadata.col_type == TableColumnType.STOCK:
            security_col = col
            continue
        if col.metadata.col_type in (
            TableColumnType.CURRENCY,
            TableColumnType.FLOAT,
            TableColumnType.INTEGER,
            TableColumnType.DELTA,
            TableColumnType.PCT_DELTA,
            TableColumnType.PERCENT,
        ):
            metric_col = col
            for i, c in enumerate(col.data):
                if date_col is not None:
                    quarter_col.append(date_col.data[i])
                if security_col is not None:
                    security = cast(StockID, security_col.data[i])
                    label_col.append(
                        f"{security.symbol + ' - ' if security.symbol else ''}{col.metadata.label}"
                    )
                else:
                    label_col.append(col.metadata.label)
                amount_col.append(c)

    if security_col is not None and date_col is not None and metric_col is not None:
        cols = [
            TableColumn(
                metadata=TableColumnMetadata(
                    label=date_col.metadata.col_type.capitalize(),
                    col_type=date_col.metadata.col_type,
                ),
                data=quarter_col,
            ),
            TableColumn(
                metadata=TableColumnMetadata(label="Label", col_type=TableColumnType.STRING),
                data=label_col,
            ),
            TableColumn(
                metadata=TableColumnMetadata(label="Amount", col_type=metric_col.metadata.col_type),
                data=amount_col,
            ),
        ]

    return Table(columns=cols)


class MakeLineGraphArgs(ToolArgs):
    input_table: Table


@tool(
    description="""This function creates a line graph from an input Table.
Line graphs should be created when the user specifically asks for them, or uses the terms
"plot" when referring to the graphing action. If the
user is not specific, line graphs are best created with data that changes on a short
time interval, for example things like stock prices, but not quarterly sales. In cases
where you need to make a graph from a Table with data like that, this use this
function. (Ideally the table index should be dates or some other type
representing time.). You should never pass a Table to this function when the table
has also been passed to another tool for filtering/ranking, such tables are NOT
compatible with a line graphs.
It is extremely important that you provide enough data to
graph. E.g. at least 7-14 datapoints to make a nice line. Note that the input
must be a Table! If the source data is stock pricing data, financial data, or
economic time series data, then a date range should be used to acquire the data
rather than a single date or no date.
Note that if the client mentions wanting to overlay one kind of data with another,
you should first construct a table with all the required data and then create a single graph.
"""
)
async def make_line_graph(args: MakeLineGraphArgs, context: PlanRunContext) -> LineGraph:
    cols = args.input_table.columns
    if len(cols) < 2:
        raise RuntimeError("Table must have at least two columns to make a line graph!")
    if args.input_table.get_num_rows() < 2:
        # this check assumes there is only 1 dataset to be graphed
        # it fails to block a multidateset table (multi stock) where each stock
        # has only 1 datapoint
        raise RuntimeError("Need at least two points to make a line graph!")

    input_table = consolidate_table_columns(args.input_table)
    cols = input_table.columns
    x_axis_col = None
    dataset_col = None
    data_col = None
    # TODO can probably clean this up to not use the df at all
    df = input_table.to_df()
    for col, df_col in zip(cols, df.columns):
        if not x_axis_col and col.metadata.col_type.is_date_type():
            x_axis_col = (col, df_col)
            continue

        if (
            not dataset_col
            and len(cols) > 2
            and col.metadata.col_type
            not in (
                TableColumnType.CURRENCY,
                TableColumnType.FLOAT,
                TableColumnType.INTEGER,
                TableColumnType.DELTA,
                TableColumnType.PCT_DELTA,
                TableColumnType.PERCENT,
                TableColumnType.INTEGER_WITH_UNIT,
                TableColumnType.FLOAT_WITH_UNIT,
            )
        ):
            # We only have a dataset column if we have > 2 columns. Otherwise we
            # just have the X axis and the value.
            dataset_col = (col, df_col)
            continue

        # Keep adding columns while they're the same types as the first graphable column
        if not data_col and col.metadata.col_type in (
            TableColumnType.CURRENCY,
            TableColumnType.FLOAT,
            TableColumnType.INTEGER,
            TableColumnType.DELTA,
            TableColumnType.PCT_DELTA,
            TableColumnType.PERCENT,
            TableColumnType.INTEGER_WITH_UNIT,
            TableColumnType.FLOAT_WITH_UNIT,
        ):
            data_col = (col, df_col)

    if not data_col:
        raise RuntimeError(f"No numerical columns found, cannot create line graph: {cols}")

    if not x_axis_col:
        raise RuntimeError(
            f"Unable to create line graph for table with no possible x axis column: {cols}"
        )

    data = []
    y_col, y_df_col = data_col
    x_col, x_df_col = x_axis_col
    if dataset_col is None and len(cols) <= 2:
        # In this case, we only have a single dataset so we don't need to do any
        # fancy grouping. The dataset name is simply the name of the graphed
        # column. This only happens when there are exactly two columns, so we
        # can just use the first col to graph.
        dataset = GraphDataset(
            dataset_id=str(y_col.metadata.label),
            dataset_id_type=TableColumnType.STRING,
            points=[
                DataPoint(x_val=x_val, y_val=y_val)
                for x_val, y_val in zip(df[x_df_col], df[y_df_col])
                if not pd.isna(x_val) and not pd.isna(y_val)
            ],
        )
        data = [dataset]
    elif dataset_col is None:
        # In this case, we have no "dataset" column (e.g. no stocks or whatever)
        # and we just have an X axis with multiple data columns. In this case,
        # we want to construct datasets simply based on the titles of the
        # columns.
        for col in cols:
            if col.metadata == x_col.metadata:
                # Ignore the x-axis
                continue
            data.append(
                GraphDataset(
                    dataset_id=str(col.metadata.label),
                    dataset_id_type=TableColumnType.STRING,
                    points=[
                        DataPoint(x_val=x_val, y_val=y_val)  # type: ignore
                        for x_val, y_val in zip(x_col.data, col.data)
                        if x_val is not None and y_val is not None
                    ],
                )
            )
    else:
        ds_col, dataset_df_col = dataset_col
        # get the unique dataset keys
        dataset_vals = pd.unique(df[dataset_df_col])
        for dataset_val in dataset_vals:
            # For each unique value in the dataset column, extract data for the
            # graph. For example, if the dataset column contains stock ID's,
            # this will create a dataset for each stock.
            dataset_data = df.loc[df[dataset_df_col] == dataset_val]
            try:
                # if all values are NaN, skip this dataset
                if dataset_data[y_df_col].isnull().all():
                    continue
                dataset = GraphDataset(
                    dataset_id=dataset_val,
                    dataset_id_type=ds_col.metadata.col_type,
                    points=[
                        DataPoint(x_val=x_val, y_val=y_val)
                        for x_val, y_val in zip(dataset_data[x_df_col], dataset_data[y_df_col])
                        if not pd.isna(x_val) and not pd.isna(y_val)
                    ],
                )
            except TypeError:
                raise RuntimeError("Cannot graph table with a single row!")
            data.append(dataset)

    return LineGraph(
        x_axis_type=x_axis_col[0].metadata.col_type,
        x_unit=x_col.metadata.unit,
        y_axis_type=y_col.metadata.col_type,
        y_unit=y_col.metadata.unit,
        data=data,
    )


class MakePieGraphArgs(ToolArgs):
    input_table: Table


@tool(
    description="""This function creates a pie graph from an input Table.
Pie chart graphs should be created when the user specifically asks for them. If
the user is not specific, pie graphs are best created from simple one
dimensional data, or data that has categories representing parts of a whole.
In specific, pie graphs should be used for only one quarter of data, not for multiple.
For example, a table containing
the country of origin for a group of stocks, where the stocks could be grouped
by country and presented in the chart. You should MAKE SURE the table ONLY has
the necessary data before calling this, so we don't graph the wrong thing!
(Ideally a single index and a single column!)
Note that the input must be a Table!
If you are creating a pie graph involving a broad statistic (e.g. sales) and some more
specific statistic that makes up a part of that statistic (e.g car sales), then, to make a valid
pie chart, make sure you do a table transform to replace "revenue" with "other revenue"
by removing the part of the sales that is car sales, before you call this function, this
function will NOT do that transformation for you, it just directly graphs the number that
are provided to it in the input table.
"""
)
async def make_pie_graph(args: MakePieGraphArgs, context: PlanRunContext) -> PieGraph:
    if len(args.input_table.columns) < 2:
        raise RuntimeError("Must have at least two columns to create pie chart!")

    modified_table = deepcopy(args.input_table)
    cols: List[TableColumn] = []
    for col in args.input_table.columns:
        if col.is_data_identical() is False:
            cols.append(col)
    # Removes any cols with duplicate data
    modified_table.columns = cols
    df = modified_table.to_df()

    # Bit of a hack: if there is only a single row, the first column is a stock,
    # and the rest of the columns are the same types, transpose the table.
    column_types = [col.metadata.col_type for col in cols]
    unique_col_types = set(column_types[1:])
    if (
        len(df) == 1  # one row
        and cols[0].metadata.col_type == TableColumnType.STOCK  # first col stock
        and len(unique_col_types) == 1  # other cols have same types
    ):
        new_cols = [
            TableColumn(
                metadata=TableColumnMetadata(label="Name", col_type=TableColumnType.STRING),
                data=[col.metadata.label for col in cols[1:]],
            ),  # labels of columns = first column values
            TableColumn(
                metadata=TableColumnMetadata(label="Value", col_type=unique_col_types.pop()),
                data=[col.data[0] for col in cols[1:]],
            ),  # data = second column values
        ]
        cols = new_cols
        modified_table = Table(columns=new_cols)
        df = modified_table.to_df()

    # Do some analysis on the table to convert it to the best-fit graph.
    label_col = None
    label_df_col = None
    data_col = None
    data_df_col = None
    for col, df_col in zip(cols, df.columns):
        if not label_col and col.metadata.col_type in (
            TableColumnType.STOCK,
            TableColumnType.STRING,
            TableColumnType.QUARTER,
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

    if data_col.metadata.col_type in (
        TableColumnType.PERCENT,
        TableColumnType.FLOAT,
        TableColumnType.INTEGER,
        TableColumnType.CURRENCY,
    ):
        # We have numerical data. Categories are simply the index, no
        # counting required.
        if data_col.metadata.row_descs:
            # Since we have row descriptions we assume each row should be preserved
            # along with their ordering, do not groupby or drop rows
            kpi_history = HistoryEntry(
                citations=[
                    TextCitation(
                        source_text=KPIText(
                            val=kpi_desc[0].name, explanation=kpi_desc[0].explanation
                        )
                    )
                    for kpi_desc in data_col.metadata.row_descs.values()
                ]
            )
            kpi_name_to_id_mapping = {
                citation.source_text.val: citation.source_text.id  # type: ignore
                for citation in kpi_history.citations
            }
            pie_graph_data = []
            for i, (idx, val) in enumerate(df.itertuples(index=False)):
                pie_graph_data.append(
                    PieSection(
                        label=idx,  # type: ignore
                        value=val,
                        citation_refs=[
                            kpi_name_to_id_mapping[kpi_desc.name]
                            for kpi_desc in data_col.metadata.row_descs[i]
                        ],
                    )
                )
            return PieGraph(
                label_type=label_col.metadata.col_type,
                data_type=data_col.metadata.col_type,
                unit=data_col.metadata.unit,
                history=[kpi_history],
                data=pie_graph_data,
            )
        else:
            # Get the "most recent" if there are multiple groups
            grouped_df = df.groupby(label_df_col).last()
            return PieGraph(
                label_type=label_col.metadata.col_type,
                data_type=data_col.metadata.col_type,
                unit=data_col.metadata.unit,
                data=[
                    PieSection(label=idx, value=val)  # type: ignore
                    for idx, val in grouped_df[data_df_col].items()
                ],
            )
    if data_col.metadata.col_type in (
        TableColumnType.STRING,
        TableColumnType.STOCK,
        TableColumnType.BOOLEAN,
        TableColumnType.QUARTER,
    ):
        # Get the "most recent" if there are multiple groups
        grouped_df = df.groupby(label_df_col).last()
        # We have categorical data, counting is required
        return PieGraph(
            label_type=data_col.metadata.col_type,
            data_type=TableColumnType.INTEGER,
            unit=data_col.metadata.unit,
            data=[
                PieSection(label=category, value=count)  # type: ignore
                for category, count in grouped_df[data_df_col].value_counts().items()
            ],
        )

    raise RuntimeError("Input table cannot be converted to a pie chart!")


class MakeBarGraphArgs(ToolArgs):
    input_table: Table


@tool(
    description="""This function creates a bar graph from an input Table.
Bar graphs should be created when the user specifically asks for them. If the
user is not specific, bar graphs are best created with multi-dimensional data
that falls into a small number of discrete and possibly ordered buckets.
For example, a bar graph containing values for
a revenue statistic for each fiscal quarter for many stocks is a good use of a bar graph,
while market share within an industry or daily stock prices are not good uses.

In case where you need to make a graph from a Table with data like that, this use this
function. (The table can be indexed by arbitrary, low numbers of labels)

Note that the input must be a Table! If the source data is stock pricing data, financial data, or
economic time series data, then a date range should be used to acquire the data rather than a
single date or no date.
"""
)
async def make_bar_graph(args: MakeBarGraphArgs, context: PlanRunContext) -> BarGraph:
    cols = args.input_table.columns
    if len(cols) < 2:
        raise RuntimeError("Table must have at least two columns to make a bar graph!")

    input_table = consolidate_table_columns(args.input_table)
    cols = input_table.columns
    x_axis_col = None
    dataset_col = None
    data_col = None
    # TODO can probably clean this up to not use the df at all
    df = input_table.to_df()
    for col, df_col in zip(cols, df.columns):
        # allow flexible types for bar axes
        if not x_axis_col and col.metadata.col_type in (
            TableColumnType.STOCK,
            TableColumnType.INTEGER,
            TableColumnType.DATE,
            TableColumnType.DATETIME,
            TableColumnType.STRING,
            TableColumnType.QUARTER,
        ):
            x_axis_col = (col, df_col)
            continue

        if not dataset_col and len(cols) > 2:
            # We only have a dataset column if we have > 2 columns. Otherwise we
            # just have the X axis and the value.
            dataset_col = (col, df_col)
            continue

        if not data_col and col.metadata.col_type in (
            TableColumnType.CURRENCY,
            TableColumnType.FLOAT,
            TableColumnType.INTEGER,
            TableColumnType.DELTA,
            TableColumnType.PCT_DELTA,
            TableColumnType.PERCENT,
        ):
            data_col = (col, df_col)

    if not data_col:
        raise RuntimeError(f"No numerical columns found, cannot create bar graph: {cols}")

    if not x_axis_col:
        raise RuntimeError(
            f"Unable to create bar graph for table with no possible x axis column: {cols}"
        )

    data: List[BarData] = []
    data_col, data_df_col = data_col
    x_axis_col, x_axis_df_col = x_axis_col
    label_type = TableColumnType.STRING
    index_type = x_axis_col.metadata.col_type
    if dataset_col is None:
        # In this case, we only have a single dataset so we don't need to do any
        # fancy grouping. The dataset name is simply the name of the graphed
        # column. This only happens when there are exactly two columns, so we
        # can just use the first col to graph.
        data = [
            BarData(index=x_val, values=[BarDataPoint(label=data_col.metadata.label, value=y_val)])
            for x_val, y_val in zip(df[x_axis_df_col], df[data_df_col])
            if not pd.isna(x_val) and not pd.isna(y_val)
        ]
    else:
        ds_col, dataset_df_col = dataset_col
        # get the unique dataset keys
        pivoted_df = df.pivot_table(
            index=x_axis_df_col,
            columns=dataset_df_col,
            values=data_df_col,
            aggfunc="last",
        )
        records: List[Dict[Any, Any]] = pivoted_df.to_dict(orient="records")
        data = [
            BarData(
                index=index_val,
                values=[BarDataPoint(label=k, value=v) for k, v in record.items()],
            )
            for index_val, record in zip(pivoted_df.index, records)
            if not pd.isna(index_val)
        ]
        label_type = ds_col.metadata.col_type

    return BarGraph(
        data_type=data_col.metadata.col_type,
        data_unit=data_col.metadata.unit,
        data=data,
        index_type=index_type,
        label_type=label_type,
    )


class MakeGenericGraphArgs(ToolArgs):
    input_table: Table


@tool(
    description="""This function creates a generic graph from an input Table.
Use this tool if the user is not specific about the graph type, for example if they
ask to make a graph without specifying the type of graph ("give me a graph for x")

In this case, this tool will decide the type of graph to use depending on the table structure.

Note that the input must be a Table! If the source data is stock pricing data, financial data, or
economic time series data, then a date range should be used to acquire the data rather than a
single date or no date.
"""
)
async def make_generic_graph(args: MakeGenericGraphArgs, context: PlanRunContext) -> Graph:
    cols = args.input_table.columns
    # use some very simple heuristics if the LLM could not make this determination
    # 1. Line graphs are great for timeseries data (date axes + numerical values)
    # 2. Bar graphs are great for sparse timeseries data or categorical data (
    #       string axes + has "dataset" + numerical values)
    # 3. Pie graphs are great for showing parts of a whole (string axes + numerical values)
    first_col_type = cols[0].metadata.col_type
    num_rows = args.input_table.get_num_rows()
    has_line_graph_compatible_cols = first_col_type in [
        TableColumnType.INTEGER,
        TableColumnType.FLOAT,
        TableColumnType.DATE,
        TableColumnType.DATETIME,
    ]

    # if we have a graph preference that the table itself had set, use that
    logger = get_prefect_logger(__name__)
    if args.input_table.prefer_graph_type == GraphType.LINE:
        logger.info("Making line graph based on table preference.")
        graph = await make_line_graph(args, context)
    elif args.input_table.prefer_graph_type == GraphType.BAR:
        logger.info("Making bar graph based on table preference.")
        graph = await make_bar_graph(args, context)
    elif args.input_table.prefer_graph_type == GraphType.PIE:
        logger.info("Making pie graph based on table preference.")
        graph = await make_pie_graph(args, context)
    elif has_line_graph_compatible_cols and num_rows >= 2:
        # has a numeric/date axis and enough datapoints? make a line graph
        logger.info("Inferred best option as line graph")
        graph = await make_line_graph(args, context)
    else:
        # else make a bar graph. we never infer a Pie chart by default
        logger.info("Inferred best option as bar graph")
        graph = await make_bar_graph(args, context)

    assert isinstance(graph, Graph)
    return graph
