from typing import List

from agent_service.io_type_utils import HistoryEntry
from agent_service.io_types.stock import StockID
from agent_service.tool import ToolArgs, ToolCategory, ToolRegistry, tool
from agent_service.types import PlanRunContext
from agent_service.utils.postgres import get_psql

ONE_HOUR = 60 * 60


class FilterStockRegionInput(ToolArgs):
    stock_ids: List[StockID]
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
) -> List[StockID]:
    sql = """
    SELECT gbi_security_id
    WHERE security_region = %(region)s
    AND gbi_security_id = ANY(%(stocks)s)
    """
    db = get_psql()
    rows = db.generic_read(
        sql, {"stocks": [stock.gbi_id for stock in args.stock_ids], "region": args.region_name}
    )
    stocks_to_include = {row["gbi_security_id"] for row in rows}
    return [
        stock.with_history_entry(HistoryEntry(explanation=f"In region '{args.region_name}'"))
        for stock in args.stock_ids
        if stock.gbi_id in stocks_to_include
    ]
