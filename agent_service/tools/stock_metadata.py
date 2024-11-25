from typing import List

from agent_service.io_types.stock import StockID
from agent_service.io_types.text import StockDescriptionText
from agent_service.tool import ToolArgs, ToolCategory, ToolRegistry, tool
from agent_service.tools.tool_log import tool_log
from agent_service.types import PlanRunContext


class GetStockDescriptionInput(ToolArgs):
    stock_ids: List[StockID]


@tool(
    description=(
        "Given a list of stock ID's, return a list of descriptions for the stocks. "
        "A stock description generally contains basic, general information about the company's "
        "operations, including major products, services, and holdings, the regions they operate in, etc. "
    ),
    category=ToolCategory.TEXT_RETRIEVAL,
    tool_registry=ToolRegistry,
)
async def get_company_descriptions(
    args: GetStockDescriptionInput, context: PlanRunContext
) -> List[StockDescriptionText]:
    await tool_log(log=f"Found {len(args.stock_ids)} company descriptions", context=context)
    return [
        StockDescriptionText(id=stock_id.gbi_id, stock_id=stock_id) for stock_id in args.stock_ids
    ]
