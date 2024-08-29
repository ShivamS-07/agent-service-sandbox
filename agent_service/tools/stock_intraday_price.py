from typing import Dict, List

from agent_service.external.feature_svc_client import get_intraday_prices
from agent_service.io_types.stock import StockID
from agent_service.tool import ToolArgs, ToolCategory, ToolRegistry, tool
from agent_service.types import PlanRunContext


class GetStockIntradayPriceInput(ToolArgs):
    stock_ids: List[StockID]


@tool(
    description=("Given a list of stock ID's, return a list of intra-day, real-time prices."),
    category=ToolCategory.STOCK,
    tool_registry=ToolRegistry,
)
async def get_stock_intraday_prices(
    args: GetStockIntradayPriceInput, context: PlanRunContext
) -> Dict[int, float]:
    prices = await get_intraday_prices([stock_id.gbi_id for stock_id in args.stock_ids])
    return prices
