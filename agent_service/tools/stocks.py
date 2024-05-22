# Author(s): Mohammad Zarei, David Grohmann

from typing import Any, Dict, List

from agent_service.external.stock_search_dao import async_sort_stocks_by_volume
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
    rows = await raw_stock_identifier_lookup(args, context)
    if len(rows) == 1:
        return rows[0]["gbi_security_id"]

    # we have multiple matches, lets use dollar trading volume to choose the most likely match
    gbi_ids = [r["gbi_security_id"] for r in rows]
    stock_by_volume = await async_sort_stocks_by_volume(gbi_ids)  # fixme asyncify

    if stock_by_volume:
        return stock_by_volume[0][0]

    # if nothing returned from stock search then just pick the first match
    return rows[0]["gbi_security_id"]


async def raw_stock_identifier_lookup(
    args: StockIdentifierLookupInput, context: PlanRunContext
) -> List[Dict[str, Any]]:
    """Returns the the stocks with the closest text match to the input string
    such as name, isin or symbol (JP3633400001, microsoft, apple, AAPL, TESLA, META, e.g.).

    This function performs a series of queries to find the stock's identifier.
    It starts with an exact symbol match,
    then an exact ISIN match,
    followed by a word similarity name match, and finally a word similarity symbol match.
    It only proceeds to the next query if the previous one returns no results.


    Args:
        args (StockIdentifierLookupInput): The input arguments for the stock lookup.
        context (PlanRunContext): The context of the plan run.

    Returns:
        int: The integer identifier of the stock.
    """
    db = get_psql()
    # TODO:
    # Using chat context to help decide
    # Use embedding to find the closest match (e.g. "google" vs "alphabet")
    # ignore common company suffixes like Corp and Inc.
    # get alternative name db up and query it
    # make use of custom doc company tagging machinery

    # Exact symbol match
    sql = """
    SELECT gbi_security_id, symbol, isin, security_region, currency, name
    FROM master_security ms
    WHERE ms.symbol = upper(%s)
    AND ms.is_public
    AND ms.asset_type in ('Common Stock', 'Depositary Receipt (Common Stock)')
    AND ms.is_primary_trading_item = true
    AND ms.to_z is null
    """
    rows = db.generic_read(sql, [args.stock_name])
    if rows:
        # useful for debugging
        # print("symbol match: ", rows)
        return rows

    # Exact ISIN match
    sql = """
    SELECT gbi_security_id, symbol, isin, security_region, currency, name
    FROM master_security ms
    WHERE ms.isin = upper(%s)
    AND ms.is_public
    AND ms.asset_type  in ('Common Stock', 'Depositary Receipt (Common Stock)')
    AND ms.is_primary_trading_item = true
    AND ms.to_z is null
    """
    rows = db.generic_read(sql, [args.stock_name])
    if rows:
        # useful for debugging
        # print("isin match: ", rows)
        return rows

    # Word similarity name match

    # should we also allow 'Depositary Receipt (Common Stock)') ?
    sql = """
    select * from (SELECT gbi_security_id, symbol, isin, security_region, currency, name
    , strict_word_similarity(lower(ms.name), lower(%s)) as ws
    FROM master_security ms
    WHERE ms.asset_type = 'Common Stock'
    AND ms.is_public
    AND ms.is_primary_trading_item = true
    AND ms.to_z is null
    ORDER BY ws DESC
    LIMIT 50) as tmp_ms
    WHERE
    tmp_ms.ws >= 0.2
    """
    rows = db.generic_read(sql, [args.stock_name])
    if rows:
        # the weaker the match the more results to be
        # considered for trading volume tie breaker

        # exact text  match
        matches = [r for r in rows if r["ws"] >= 1.0]
        if matches:
            # if there is more than 1 exact match we have to break the tie
            return matches

        # strong text  match
        matches = [r for r in rows if r["ws"] >= 0.9]
        if matches:
            return matches[:5]

        matches = [r for r in rows if r["ws"] >= 0.7]
        if matches:
            return matches[:5]

        matches = [r for r in rows if r["ws"] >= 0.4]
        if matches:
            return matches[:10]

        matches = [r for r in rows if r["ws"] >= 0.3]
        if matches:
            return matches[:20]

        # very weak text match
        matches = [r for r in rows if r["ws"] > 0.2]
        if matches:
            return matches[:50]

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
