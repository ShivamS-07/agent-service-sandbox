from typing import List

from agent_service.io_types.stock import StockAlignedTextGroups, StockID
from agent_service.io_types.text import CompanyDescriptionText, TextGroup
from agent_service.tool import ToolArgs, ToolCategory, ToolRegistry, tool
from agent_service.types import PlanRunContext


class GetStockDescriptionInput(ToolArgs):
    stock_ids: List[StockID]


@tool(
    description=(
        "Given a list of stock ID's, return a list of descriptions for the stocks. "
        "A stock description generally contains basic, general information about the company's "
        "operations, including major products, services, and holdings, the regions they operate in, etc. "
        "This function should be used if you intend to summarize one or handful of descriptions, "
        " or show these descriptions to the user directly"
        "For example, if a user asked `Please give me a quick rundown on what "
        "the company Snowflake does`, you would use this function to get the information"
        " You should also use it for answering a question about a specific stock related to"
        " information likely to be discussed in company description."
    ),
    category=ToolCategory.STOCK,
    tool_registry=ToolRegistry,
)
async def get_company_descriptions(
    args: GetStockDescriptionInput, context: PlanRunContext
) -> List[CompanyDescriptionText]:
    return [CompanyDescriptionText(id=stock_id.gbi_id) for stock_id in args.stock_ids]


@tool(
    description=(
        "Given a list of stock ID's, returns a StockAlignedTextGroups objects with "
        "mappings from stock ids to descriptions for those ids. "
        "A stock description generally contains basic, general information about the company's "
        "operations, including major products, services, and holdings, the regions they operate in, etc. "
        "This function should be used if you intend to use the descriptions in ways "
        "that require you to preserve the mapping from stocks, for example filtering stocks "
        "based on their description. "
        "For example, if the client asked: `I want a list of stocks which sell electronics`"
        " you would use this function to get information about the companies so you can filter "
        " to the ones asked for by the client"
        " You should NOT use this function for getting data for questions about specific stocks."
    ),
    category=ToolCategory.STOCK,
    tool_registry=ToolRegistry,
)
async def get_company_descriptions_stock_aligned(
    args: GetStockDescriptionInput, context: PlanRunContext
) -> StockAlignedTextGroups:
    return StockAlignedTextGroups(
        val={
            stock_id: TextGroup(val=[CompanyDescriptionText(id=stock_id.gbi_id)])
            for stock_id in args.stock_ids
        }
    )
