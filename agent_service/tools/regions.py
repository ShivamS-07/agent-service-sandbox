from typing import List

from agent_service.tool import ToolArgs, ToolCategory, ToolRegistry, tool
from agent_service.types import PlanRunContext
from agent_service.utils.postgres import get_psql

ONE_HOUR = 60 * 60


class FilterStockRegionInput(ToolArgs):
    stock_ids: List[int]
    region_name: str


@tool(
    description=(
        "This function takes a list of stock ID's an ISO3 country code string"
        " like 'USA' or 'CAN', and it filters the list of stocks by the given region."
        " It returns the filtered list of stock IDs."
    ),
    category=ToolCategory.STOCK,
    tool_registry=ToolRegistry,
)
async def filter_stocks_by_region(
    args: FilterStockRegionInput, context: PlanRunContext
) -> List[int]:
    sql = """
    SELECT gbi_security_id FROM master_security
    WHERE security_region = %(region)s
    AND gbi_security_id = ANY(%(stocks)s)
    """
    db = get_psql()
    rows = db.generic_read(sql, {"stocks": args.stock_ids, "region": args.region_name})
    return [row["gbi_security_id"] for row in rows]
