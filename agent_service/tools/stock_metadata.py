from typing import List

from agent_service.tool import ToolArgs, ToolCategory, ToolRegistry, tool
from agent_service.types import PlanRunContext
from agent_service.utils.postgres import get_psql


class GetStockDescriptionInput(ToolArgs):
    stock_ids: List[int]


@tool(
    description="Given a list of stock ID's, return a list of descriptions for the stocks.",
    category=ToolCategory.STOCK,
    tool_registry=ToolRegistry,
)
async def get_company_descriptions(
    args: GetStockDescriptionInput, context: PlanRunContext
) -> List[str]:
    sql = """
    SELECT ssm.gbi_id, cds.company_description_short
    FROM spiq_security_mapping ssm
    JOIN nlp_service.company_descriptions_short cds
    ON cds.spiq_company_id = ssm.spiq_company_id
    WHERE ssm.gbi_id = ANY(%(stocks)s)
    """
    db = get_psql()
    rows = db.generic_read(sql, {"stocks": args.stock_ids})
    stock_desc_map = {row["gbi_id"]: row["company_description_short"] for row in rows}
    return [stock_desc_map[stock] for stock in args.stock_ids]
