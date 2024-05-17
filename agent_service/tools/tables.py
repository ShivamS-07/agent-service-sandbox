import pandas as pd

from agent_service.io_types import TimeSeriesTable
from agent_service.tool import ToolArgs, ToolCategory, ToolRegistry, tool
from agent_service.types import PlanRunContext


class TimeseriesTableAvgInput(ToolArgs):
    table: TimeSeriesTable
    new_column: str = "Averaged Value"


@tool(
    description=(
        "This function collapses a time series table to a single column by"
        " taking the average (mean) score across all columns for each date,"
        " the resulting table has a single column with the provided new_column header."
    ),
    category=ToolCategory.TABLE,
    tool_registry=ToolRegistry,
)
async def average_table_by_date(
    args: TimeseriesTableAvgInput, context: PlanRunContext
) -> TimeSeriesTable:
    df = args.table.val
    output_df = df.copy(deep=True)
    output_df = output_df.mean(axis=1).rename(args.new_column).to_frame()
    return TimeSeriesTable(val=output_df)


class ConcatTimeseriesInput(ToolArgs):
    table1: TimeSeriesTable
    table2: TimeSeriesTable


@tool(
    description=(
        "This function concatenates two compatible time series tables together, "
        "the resulting table has the same dates (the rows) and all the columns in both tables."
    ),
    category=ToolCategory.TABLE,
    tool_registry=ToolRegistry,
    is_visible=False,
)
async def concat_time_series_tables(
    args: ConcatTimeseriesInput, context: PlanRunContext
) -> TimeSeriesTable:
    df1 = args.table1.val
    df2 = args.table2.val
    output_df = pd.concat([df1, df2], axis=1)

    return TimeSeriesTable(val=output_df)
