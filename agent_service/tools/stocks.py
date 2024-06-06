# Author(s): Mohammad Zarei, David Grohmann


from typing import Any, Dict, List

from agent_service.external.stock_search_dao import async_sort_stocks_by_volume
from agent_service.io_types.stock import StockID
from agent_service.tool import ToolArgs, ToolCategory, ToolRegistry, tool
from agent_service.tools.tool_log import tool_log
from agent_service.types import PlanRunContext
from agent_service.utils.postgres import get_psql
from agent_service.utils.prefect import get_prefect_logger


class StockIdentifierLookupInput(ToolArgs):
    # name or symbol of the stock to lookup
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
async def stock_identifier_lookup(
    args: StockIdentifierLookupInput, context: PlanRunContext
) -> StockID:
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
    logger = get_prefect_logger(__name__)
    logger.info(f"Attempting to map '{args.stock_name}' to a stock")
    rows = await raw_stock_identifier_lookup(args, context)
    if not rows:
        raise ValueError(f"Could not find any stocks related to: '{args.stock_name}'")

    if len(rows) == 1:
        only_stock = rows[0]
        logger.info(f"found only 1 stock {only_stock}")
        return StockID(
            gbi_id=only_stock["gbi_security_id"],
            symbol=only_stock["symbol"],
            isin=only_stock["isin"],
            company_name=only_stock["name"],
        )

    logger.info(f"found {len(rows)} best potential matching stocks")
    # we have multiple matches, lets use dollar trading volume to choose the most likely match
    gbiid2stocks = {r["gbi_security_id"]: r for r in rows}
    gbi_ids = list(gbiid2stocks.keys())
    stocks_sorted_by_volume = await async_sort_stocks_by_volume(gbi_ids)  # fixme asyncify

    if stocks_sorted_by_volume:
        gbi_id = stocks_sorted_by_volume[0][0]
        stock = gbiid2stocks.get(gbi_id)
        if not stock:
            stock = rows[0]
            logger.warning("Logic error!")
            # should not be possible
        else:
            logger.info(f"Top stock volumes: {stocks_sorted_by_volume[:10]}")
    else:
        # if nothing returned from stock search then just pick the first match
        stock = rows[0]
        logger.info("No stock volume info available!")

    logger.info(f"found stock: {stock} from '{args.stock_name}'")
    await tool_log(
        log=f"Interpreting '{args.stock_name}' as {stock['symbol']}: {stock['name']}",
        context=context,
    )

    return StockID(
        gbi_id=stock["gbi_security_id"],
        symbol=stock["symbol"],
        isin=stock["isin"],
        company_name=stock["name"],
    )


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
    logger = get_prefect_logger(__name__)
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
        logger.info("found by symbol")
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
        logger.info("found by ISIN")
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
            logger.info(f"found {len(matches)} perfect matches")
            return matches

        # strong text  match
        matches = [r for r in rows if r["ws"] >= 0.9]
        if matches:
            logger.info(f"found {len(matches)} nearly perfect matches")
            return matches[:5]

        matches = [r for r in rows if r["ws"] >= 0.7]
        if matches:
            logger.info(f"found {len(matches)} strong matches")
            return matches[:5]

        matches = [r for r in rows if r["ws"] >= 0.4]
        if matches:
            logger.info(f"found {len(matches)} medium matches")
            return matches[:10]

        matches = [r for r in rows if r["ws"] >= 0.3]
        if matches:
            logger.info(f"found {len(matches)} weak matches")
            return matches[:20]

        # very weak text match
        matches = [r for r in rows if r["ws"] > 0.2]
        if matches:
            logger.info(f"found {len(matches)} very weak matches")
            return matches[:50]

        if rows:
            logger.info(
                f"found {len(rows)} potential matches but they were all likely unrelated to the user intent"
            )

    raise ValueError(f"Could not find any stocks related to: '{args.stock_name}'")


class MultiStockIdentifierLookupInput(ToolArgs):
    # name or symbol of the stock to lookup
    stock_names: List[str]


@tool(
    description=(
        "This function takes a list of strings e.g. ['microsoft', 'apple', 'TESLA', 'META'] "
        "which refer to stocks, and converts them to a list of integer identifiers."
    ),
    category=ToolCategory.STOCK,
    tool_registry=ToolRegistry,
    is_visible=False,
)
async def multi_stock_identifier_lookup(
    args: MultiStockIdentifierLookupInput, context: PlanRunContext
) -> List[StockID]:
    # Just runs stock identifier look up below for each stock in the list
    # Probably can be done more efficiently

    output: List[StockID] = []
    for stock_name in args.stock_names:
        output.append(
            await stock_identifier_lookup(  # type: ignore
                StockIdentifierLookupInput(stock_name=stock_name), context
            )
        )
    return output


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
        " If the client wants to filter over stocks but does not specify an initial set"
        " of stocks, you should call this tool with 'S&P 500'"
    ),
    category=ToolCategory.STOCK,
    tool_registry=ToolRegistry,
    is_visible=False,
)
async def get_stock_universe(args: GetStockUniverseInput, context: PlanRunContext) -> List[StockID]:
    """Returns the list of stock identifiers given a stock universe name.

    Args:
        args (GetStockUniverseInput): The input arguments for the stock universe lookup.
        context (PlanRunContext): The context of the plan run.

    Returns:
        list[StockID]: The list of stock identifiers in the universe.
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
    SELECT DISTINCT ON (gbi_id)
    gbi_id, symbol, isin, name
    FROM "data".etf_universe_holdings euh
    JOIN master_security ms ON ms.gbi_security_id = euh.gbi_id
    WHERE spiq_company_id = %s
    AND euh.to_z > NOW()
    """
    rows = db.generic_read(sql, [universe_spiq_company_id])

    return [
        StockID(
            gbi_id=row["gbi_id"], symbol=row["symbol"], isin=row["isin"], company_name=row["name"]
        )
        for row in rows
    ]
