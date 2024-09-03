from typing import List

import pandas as pd

from agent_service.external.feature_svc_client import get_intraday_prices
from agent_service.io_types.stock import StockID
from agent_service.io_types.table import (
    StockTable,
    TableColumnMetadata,
    TableColumnType,
)
from agent_service.tool import ToolArgs, ToolCategory, ToolRegistry, tool
from agent_service.types import PlanRunContext


class GetStockIntradayPriceInput(ToolArgs):
    stock_ids: List[StockID]


@tool(
    description=(
        "Given a list of stock ID's, return a Table of one price for each stock"
        " the price will be the most up to date current price available"
        " also known as the intra-day price, real-time price, current price."
        " Only use this tool when the client asks to see current price, today's price,"
        " intra day, and realtime prices.\n"
        "Historical prices, prices on a certain date, close prices should use"
        " get_statistic_data_for_companies function instead"
    ),
    category=ToolCategory.STOCK,
    tool_registry=ToolRegistry,
)
async def get_stock_intraday_prices(
    args: GetStockIntradayPriceInput, context: PlanRunContext
) -> StockTable:
    prices = await get_intraday_prices([stock_id.gbi_id for stock_id in args.stock_ids])

    id2stock = {stock_id.gbi_id: stock_id for stock_id in args.stock_ids}
    new_gbi_ids = list(prices.keys())
    price_list = list(prices.values())

    # preserve the history of the old stock_ids
    new_stock_ids = [
        id2stock.get(id, (await StockID.from_gbi_id_list([id]))[0]) for id in new_gbi_ids
    ]

    df = pd.DataFrame()
    df["Security"] = new_stock_ids
    df["Current Price"] = price_list

    table = StockTable.from_df_and_cols(
        data=df,
        columns=[
            TableColumnMetadata(label="Security", col_type=TableColumnType.STOCK),
            # TODO
            # this should be col_type=<TableColumnType.CURRENCY: 'currency'>,
            # but I dont know where to get unit='USD', from
            # other than asking for close price also
            TableColumnMetadata(label="Current Price", col_type=TableColumnType.FLOAT),
        ],
    )

    return table
