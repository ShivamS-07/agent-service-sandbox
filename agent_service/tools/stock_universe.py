from typing import List

from agent_service.tool import ToolArgs, ToolCategory, ToolRegistry, tool
from agent_service.types import PlanRunContext
from agent_service.utils.postgres import get_psql


class GetStockUniverseInput(ToolArgs):
    # name of the universe to lookup
    universe_name: str


@tool(
    description=(
        "This function takes a string (S&P500 e.g.)"
        "which refers to a stock universe, converts it to a string identifier "
        " and return the list of stock identifiers in the universe."
    ),
    category=ToolCategory.STOCK,
    tool_registry=ToolRegistry,
)
async def get_stock_universe(args: GetStockUniverseInput, context: PlanRunContext) -> List[int]:
    """Returns the list of stock identifiers given a stock universe name.

    Args:
        args (GetStockUniverseInput): The input arguments for the stock universe lookup.
        context (PlanRunContext): The context of the plan run.

    Returns:
        list[int]: The list of stock identifiers in the universe.
    """
    db = get_psql()
    # TODO :
    # add a cache for the stock universe
    # switch to using GetEtfHoldingsForDate not db

    # Find the universe id/name
    sql = """
    SELECT id, name
    FROM "data".etf_universes
    WHERE gbi_id IN (
        SELECT (ingestion_configuration->'benchmark')::INT
        FROM gbi_stock_universe
    )
    ORDER BY word_similarity(lower(name), lower(%s)) DESC
    LIMIT 1
    """
    rows = db.generic_read(sql, [args.universe_name])
    universe_id = rows[0]["id"]
    # universe_name = rows[0]["name"]

    # Find the stocks in the universe
    sql = """
    SELECT DISTINCT gbi_id
    FROM "data".etf_universe_holdings
    WHERE etf_id = %s
    AND to_z > NOW()
    """
    rows = db.generic_read(sql, [universe_id])

    return [row["gbi_id"] for row in rows]