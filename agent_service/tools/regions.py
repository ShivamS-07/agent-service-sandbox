from typing import List

from agent_service.io_types.misc import StockID
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
    SELECT gbi_security_id, symbol, isin FROM master_security
    WHERE security_region = %(region)s
    AND gbi_security_id = ANY(%(stocks)s)
    """
    db = get_psql()
    rows = db.generic_read(
        sql, {"stocks": [stock.gbi_id for stock in args.stock_ids], "region": args.region_name}
    )
    return [
        StockID(gbi_id=row["gbi_security_id"], symbol=row["symbol"], isin=row["isin"])
        for row in rows
    ]
