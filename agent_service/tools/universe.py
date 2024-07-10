import datetime
from typing import Optional

import pandas as pd

from agent_service.external.pa_backtest_svc_client import (
    get_universe_sector_performance_for_date_range,
)
from agent_service.io_type_utils import TableColumnType
from agent_service.io_types.dates import DateRange
from agent_service.io_types.table import Table, TableColumnMetadata
from agent_service.tool import ToolArgs, ToolCategory, ToolRegistry, tool
from agent_service.types import PlanRunContext

UniverseID = str


class GetPortfolioWorkspaceHoldingsInput(ToolArgs):
    universe_id: UniverseID
    start_date: Optional[datetime.date] = None
    end_date: Optional[datetime.date] = None
    date_range: Optional[DateRange] = None


@tool(
    description=(
        "This function returns the performance for a stock universe given a date range. "
        "Use this function if you want to get the performance of a universe."
    ),
    category=ToolCategory.STATISTICS,
    tool_registry=ToolRegistry,
    is_visible=False,
)
async def get_benchmark_performance(
    args: GetPortfolioWorkspaceHoldingsInput, context: PlanRunContext
) -> Table:

    # if a date range obj was provided, fill in any missing dates
    if args.date_range:
        if not args.start_date:
            args.start_date = args.date_range.start_date
        if not args.end_date:
            args.end_date = args.date_range.end_date

    universe_performance = await get_universe_sector_performance_for_date_range(
        start_date=args.start_date, end_date=args.end_date, stock_universe_id=args.universe_id
    )
    data = {
        "Sector Name": [datapoint.sector_name for datapoint in universe_performance],
        "Performance": [datapoint.performance for datapoint in universe_performance],
    }
    df = pd.DataFrame(data)
    table = Table.from_df_and_cols(
        data=df,
        columns=[
            TableColumnMetadata(label="Sector Name", col_type=TableColumnType.STRING),
            TableColumnMetadata(label="Performance", col_type=TableColumnType.FLOAT),
        ],
    )

    return table
