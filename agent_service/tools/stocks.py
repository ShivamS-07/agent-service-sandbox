from typing import List

from agent_service.tool import ToolArgs, ToolCategory, ToolRegistry, tool
from agent_service.types import PlanRunContext
from agent_service.utils.postgres import get_psql


class StockIdentifierLookupInput(ToolArgs):
    # name or symbole of the stock to lookup
    stock_name: str


@tool(
    description=(
        "This function takes a string (microsoft, apple, AAPL, TESLA, META, e.g.) "
        "which refers to a stock, and converts it to an integer identifier."
    ),
    category=ToolCategory.STOCK,
    tool_registry=ToolRegistry,
    is_visible=False,
)
async def stock_identifier_lookup(args: StockIdentifierLookupInput, context: PlanRunContext) -> int:
    """Returns the integer identifier of a stock given its name or symbol (microsoft, apple, AAPL, TESLA, META, e.g.).

    This function performs a series of queries to find the stock's identifier. It starts with an exact symbol match,
    followed by a word similarity name match, and finally a word similarity symbol match. It only
    proceeds to the next query if the previous one returns no results.


    Args:
        args (StockIdentifierLookupInput): The input arguments for the stock lookup.
        context (PlanRunContext): The context of the plan run.

    Returns:
        int: The integer identifier of the stock.
    """
    db = get_psql()
    # TODO:
    # Handling ISIN's
    # Handling non US stocks
    # Handling multiple possible matches
    # Using chat context to help decide
    # Use embedding to find the closest match (e.g. "google" vs "alphabet")
    # Use trading volume to help decide

    # Exact symbol match
    sql = """
    SELECT gbi_security_id FROM master_security ms
    WHERE lower(ms.symbol) = lower(%s)
    AND ms.is_public
    AND ms.asset_type = 'Common Stock'
    AND ms.is_primary_trading_item = true
    AND ms.region = 'United States'
    AND ms.to_z is null
    """
    rows = db.generic_read(sql, [args.stock_name])
    if rows:
        return rows[0]["gbi_security_id"]

    # Word similarity name match
    sql = """
    SELECT gbi_security_id FROM master_security ms
    WHERE ms.asset_type = 'Common Stock'
    AND ms.is_public
    AND ms.is_primary_trading_item = true
    AND ms.region = 'United States'
    AND ms.to_z is null
    ORDER BY word_similarity(lower(ms.name), lower(%s)) DESC
    LIMIT 1
    """
    rows = db.generic_read(sql, [args.stock_name])
    if rows:
        return rows[0]["gbi_security_id"]

    # Word similarity symbol match
    sql = """
    SELECT gbi_security_id FROM master_security ms
    WHERE word_similarity(lower(ms.symbol), lower(%s)) > 0.2
    AND ms.is_public
    AND ms.asset_type = 'Common Stock'
    AND ms.is_primary_trading_item = true
    AND ms.region = 'United States'
    AND ms.to_z is null
    ORDER BY word_similarity(lower(ms.symbol), lower(%s)) DESC
    LIMIT 1
    """
    rows = db.generic_read(sql, [args.stock_name, args.stock_name])
    if rows:
        return rows[0]["gbi_security_id"]

    raise ValueError(f"Could not find the stock {args.stock_name}")


class StockIDsToTickerInput(ToolArgs):
    stock_ids: List[int]


@tool(
    description=(
        "This converts a list of uninterpretable stock identifiers to a list of human readable tickers"
    ),
    category=ToolCategory.STOCK,
    tool_registry=ToolRegistry,
)
async def convert_stock_identifiers_to_tickers(
    args: StockIDsToTickerInput, context: PlanRunContext
) -> List[str]:
    db = get_psql()
    sql = """
    SELECT gbi_security_id AS gbi_id, symbol FROM master_security
    WHERE gbi_security_id = ANY(%(gbi_ids)s)
    """
    rows = db.generic_read(sql, {"gbi_ids": args.stock_ids})
    # Map to make sure they're in the same order
    mapping = {row["gbi_id"]: row["symbol"] for row in rows}
    return [mapping[stock_id] for stock_id in args.stock_ids]


class StatisticsIdentifierLookupInput(ToolArgs):
    # name of the statistic to lookup
    statistic_name: str


class GetStockUniverseInput(ToolArgs):
    # name of the universe to lookup
    universe_name: str


@tool(
    description=(
        "This function takes a string"
        "which refers to a stock universe, and converts it to a string identifier "
        " and returns the list of stock identifiers in the universe."
        "Stock universes are generally major market indexes like the S&P 500 or the"
        "Stoxx 600"
    ),
    category=ToolCategory.STOCK,
    tool_registry=ToolRegistry,
    is_visible=False,
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
    SELECT spiq_company_id, name
    FROM "data".etf_universes
    WHERE gbi_id IN (
        SELECT (ingestion_configuration->'benchmark')::INT
        FROM gbi_stock_universe
    )
    ORDER BY word_similarity(lower(name), lower(%s)) DESC
    LIMIT 1
    """
    rows = db.generic_read(sql, [args.universe_name])
    universe_spiq_company_id = rows[0]["spiq_company_id"]
    # universe_name = rows[0]["name"]

    # Find the stocks in the universe
    sql = """
    SELECT DISTINCT gbi_id
    FROM "data".etf_universe_holdings
    WHERE spiq_company_id = %s
    AND to_z > NOW()
    """
    rows = db.generic_read(sql, [universe_spiq_company_id])

    return [row["gbi_id"] for row in rows]
