from typing import List

from agent_service.io_types import CompanyDescriptionText
from agent_service.tool import ToolArgs, ToolCategory, ToolRegistry, tool
from agent_service.types import PlanRunContext


class GetStockDescriptionInput(ToolArgs):
    stock_ids: List[int]


@tool(
    description="Given a list of stock ID's, return a list of descriptions for the stocks.",
    category=ToolCategory.STOCK,
    tool_registry=ToolRegistry,
)
async def get_company_descriptions(
    args: GetStockDescriptionInput, context: PlanRunContext
) -> List[CompanyDescriptionText]:
    return [CompanyDescriptionText(id=stock_id) for stock_id in args.stock_ids]
