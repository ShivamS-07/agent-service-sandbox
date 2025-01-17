import datetime
from typing import List

from agent_service.external.feature_svc_client import get_intraday_prices
from agent_service.io_types.stock import StockID
from agent_service.io_types.table import StockTable
from agent_service.io_types.text import Text
from agent_service.tool import ToolArgs, ToolCategory, default_tool_registry, tool
from agent_service.tools.tool_log import tool_log
from agent_service.types import PlanRunContext


class GetStockIntradayPriceInput(ToolArgs):
    stock_ids: List[StockID]


@tool(
    description=(
        "Given a list of stock ID's, return a Table of stock prices"
        " the price will be the most up to date current price available"
        " also known as the intra-day price, real-time price, current price."
        " Only use this tool when the client asks to see current price, today's price,"
        " intra day, and realtime prices. Historical prices, prices on a certain date,"
        " close prices should use get_statistic_data_for_companies function instead.\n"
        " Since it only contains one price per stock, this tool cannot be used directly to calculate"
        " stock percentage gain, returns, movement of today's prices, deltas, etc. If the client"
        " asks for something like that, DO NOT call this tool, use the get_statistic_data_for_companies"
        " tool, which will call this tool internally to get the latest prices for its calculation."
        " You must only show this data directly to the user, i.e. directly before you call prepare_output"
        " you must never, ever pass it to a table transform, If you pass the output of this tool to the"
        " table transform tool you will be fired."
        " This tool does NOT provide market cap, use get_statistic_data_for_companies for that."
    ),
    category=ToolCategory.STATISTICS,
    tool_registry=default_tool_registry(),
)
async def get_stock_intraday_prices(
    args: GetStockIntradayPriceInput, context: PlanRunContext
) -> StockTable:
    table, min_time, max_time = await get_intraday_prices(args.stock_ids)

    # if the difference between the oldest and newest price is greater than 1 hour
    # log the full date range
    if max_time - min_time > datetime.timedelta(minutes=5):
        await tool_log(
            log=Text(val=f"The stock prices are from times between {min_time} and {max_time}"),
            context=context,
        )
    else:
        # else, log just the earliest time.
        await tool_log(
            log=Text(val=f"The stock prices are from {min_time}"),
            context=context,
        )

    return table
