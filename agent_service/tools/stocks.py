# Author(s): Mohammad Zarei, David Grohmann


from typing import Any, Dict, List, Optional

import pandas as pd
from gbi_common_py_utils.numpy_common import NumpySheet
from gbi_common_py_utils.utils.environment import get_environment_tag

from agent_service.external.pa_backtest_svc_client import (
    universe_stock_factor_exposures,
)
from agent_service.external.stock_search_dao import async_sort_stocks_by_volume
from agent_service.io_type_utils import TableColumnType
from agent_service.io_types.stock import StockID
from agent_service.io_types.table import Table, TableColumnMetadata
from agent_service.tool import ToolArgs, ToolCategory, ToolRegistry, tool
from agent_service.tools.tool_log import tool_log
from agent_service.types import PlanRunContext
from agent_service.utils.postgres import get_psql
from agent_service.utils.prefect import get_prefect_logger

STOCK_ID_COL_NAME_DEFAULT = "Security"
GROWTH_LABEL = "Growth"
VALUE_LABEL = "Value"


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
    stocks_sorted_by_volume = await async_sort_stocks_by_volume(gbi_ids)

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
    It starts with a bloomberg parsekey match,
    then an exact ISIN match,
    followed by a text similarity match to the official name and alternate names,
    and an exact ticker symbol matches are also allowed.
    It only proceeds to the next query if the previous one returns no results.


    Args:
        args (StockIdentifierLookupInput): The input arguments for the stock lookup.
        context (PlanRunContext): The context of the plan run.

    Returns:
        List[Dict[str, Any]]: DB rows representing the potentially matching stocks.
    """
    logger = get_prefect_logger(__name__)
    db = get_psql()
    # TODO:
    # Using chat context to help decide
    # Use embedding to find the closest match (e.g. "google" vs "alphabet")
    # ignore common company suffixes like Corp and Inc.
    # make use of custom doc company tagging machinery

    bloomberg_rows = get_stocks_if_bloomberg_parsekey(args, context)
    if bloomberg_rows:
        logger.info("found bloomberg parsekey")
        return bloomberg_rows

    # Exact gbi alt name match
    # these are hand crafted strings to be used only when needed
    sql = """
    SELECT gbi_security_id, ms.symbol, ms.isin, ms.security_region, ms.currency,
    ms.name, gan.alt_name as gan_alt_name
    FROM master_security ms
    JOIN "data".gbi_id_alt_names gan ON gan.gbi_id = ms.gbi_security_id
    WHERE
    upper(gan.alt_name) = upper(%(search_term)s)
    AND gan.enabled
    AND ms.is_public
    AND ms.asset_type in ('Common Stock', 'Depositary Receipt (Common Stock)')
    AND ms.is_primary_trading_item = true
    AND ms.to_z is null
    """
    rows = db.generic_read(sql, {"search_term": args.stock_name})
    if rows:
        logger.info("found exact gbi alt name")
        return rows

    # ISINs are 12 chars long, 2 chars, 10 digits
    if (
        12 == len(args.stock_name)
        and args.stock_name[0:2].isalpha()
        and args.stock_name[2:].isalnum()
    ):
        # Exact ISIN match
        sql = """
        SELECT gbi_security_id, ms.symbol, ms.isin, ms.security_region, ms.currency, name,
        'ms.isin' as match_col, ms.isin as match_text
        FROM master_security ms
        WHERE ms.isin = upper(%(search_term)s)
        AND ms.is_public
        AND ms.asset_type  in ('Common Stock', 'Depositary Receipt (Common Stock)')
        AND ms.is_primary_trading_item = true
        AND ms.to_z is null

        UNION

        -- legacy ISINs
        SELECT gbi_security_id, ms.symbol, ms.isin, ms.security_region, ms.currency, name,
        'ssm.isin' as match_col, ssm.isin as match_text
        FROM master_security ms
        JOIN spiq_security_mapping ssm ON ssm.gbi_id = ms.gbi_security_id
        WHERE ssm.isin = upper(%(search_term)s)
        AND ms.is_public
        AND ms.asset_type  in ('Common Stock', 'Depositary Receipt (Common Stock)')
        AND ms.is_primary_trading_item = true
        AND ms.to_z is null
        """
        rows = db.generic_read(sql, {"search_term": args.stock_name})
        if rows:
            # useful for debugging
            # print("isin match: ", rows)
            logger.info("found by ISIN")
            return rows

    # Word similarity name match

    # should we also allow 'Depositary Receipt (Common Stock)') ?
    sql = """
    select * from (

    -- ticker symbol (exact match only)
    SELECT gbi_security_id, ms.symbol, ms.isin, ms.security_region, ms.currency,
    ms.name, 'ticker symbol' as match_col, ms.symbol as match_text,
    1.0 AS ws
    FROM master_security ms
    WHERE
    ms.asset_type  in ('Common Stock', 'Depositary Receipt (Common Stock)')
    AND ms.is_public
    AND ms.is_primary_trading_item = true
    AND ms.to_z is null
    AND ms.symbol = upper(%(search_term)s)

    UNION

    -- company name
    SELECT gbi_security_id, symbol, ms.isin, ms.security_region, ms.currency,
    name, 'name' as match_col, ms.name as match_text,
    (strict_word_similarity(lower(ms.name), lower(%(search_term)s)) +
    strict_word_similarity(lower(%(search_term)s), lower(ms.name))) / 2
    AS ws
    FROM master_security ms
    WHERE
    ms.asset_type  in ('Common Stock', 'Depositary Receipt (Common Stock)')
    AND ms.is_public
    AND ms.is_primary_trading_item = true
    AND ms.to_z is null

    UNION

    -- company alt name
    SELECT gbi_security_id, symbol, ms.isin, ms.security_region, ms.currency,
    name, 'comp alt name' as match_col, alt_name as match_text,
    (strict_word_similarity(lower(alt_name), lower(%(search_term)s)) +
    strict_word_similarity(lower(%(search_term)s), lower(alt_name))) / 2
    AS ws
    FROM master_security ms
    JOIN spiq_security_mapping ssm ON ssm.gbi_id = ms.gbi_security_id
    JOIN "data".company_alt_names can ON ssm.spiq_company_id = can.spiq_company_id
    WHERE
    ms.asset_type  in ('Common Stock', 'Depositary Receipt (Common Stock)')
    AND can.enabled
    AND ms.is_public
    AND ms.is_primary_trading_item = true
    AND ms.to_z is null

    UNION

    -- gbi alt name
    SELECT gbi_security_id, symbol, ms.isin, ms.security_region, ms.currency,
    name, 'gbi alt name' as match_col, alt_name as match_text,
    (strict_word_similarity(lower(alt_name), lower(%(search_term)s)) +
    strict_word_similarity(lower(%(search_term)s), lower(alt_name))) / 2
    AS ws
    FROM master_security ms
    JOIN "data".gbi_id_alt_names gan ON gan.gbi_id = ms.gbi_security_id
    WHERE
    ms.asset_type  in ('Common Stock', 'Depositary Receipt (Common Stock)')
    AND gan.enabled
    AND ms.is_public
    AND ms.is_primary_trading_item = true
    AND ms.to_z is null

    ORDER BY ws DESC
    LIMIT 100) as tmp_ms
    WHERE
    tmp_ms.ws >= 0.2
    """
    rows = db.generic_read(sql, {"search_term": args.stock_name})
    if rows:
        # the weaker the match the more results to be
        # considered for trading volume tie breaker
        matches = [r for r in rows if r["ws"] >= 0.6]
        if matches:
            matches = [r for r in rows if r["ws"] >= 0.50]
            logger.info(f"found {len(matches)} strong matches")
            return matches[:20]

        matches = [r for r in rows if r["ws"] >= 0.4]
        if matches:
            matches = [r for r in rows if r["ws"] >= 0.30]
            logger.info(f"found {len(matches)} medium matches")
            return matches[:30]

        matches = [r for r in rows if r["ws"] >= 0.3]
        if matches:
            matches = [r for r in rows if r["ws"] >= 0.20]
            logger.info(f"found {len(matches)} weak matches")
            return matches[:40]

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
        "which refer to stocks, and converts them to a list of integer identifiers. "
        " Since most other tools take lists of stocks, you should generally use this function "
        " to look up stocks mentioned by the client (instead of stock_identifier_lookup), "
        " even when there is only one stock."
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
        " which refers to a stock universe, and converts it to a string identifier "
        " and then returns the list of stock identifiers in the universe."
        " Stock universes are generally major market indexes like the S&P 500 or the"
        " Stoxx 600 or the similar names of ETFs or the 3-6 letter ticker symbols for ETFs"
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
    logger = get_prefect_logger(__name__)
    # TODO :
    # add a cache for the stock universe
    # switch to using GetEtfHoldingsForDate not db

    etf_stock = await get_stock_info_for_universe(args, context)
    universe_spiq_company_id = etf_stock["spiq_company_id"]
    stock_universe_list = await get_stock_universe_list_from_universe_company_id(
        universe_spiq_company_id, context
    )

    logger.info(
        f"found {len(stock_universe_list)} holdings in ETF: {etf_stock} from '{args.universe_name}'"
    )
    await tool_log(
        log=f"Found {len(stock_universe_list)} holdings in {etf_stock['symbol']}: {etf_stock['name']}",
        context=context,
    )

    return stock_universe_list


async def get_stock_info_for_universe(args: GetStockUniverseInput, context: PlanRunContext) -> Dict:
    """Returns the company id of the best match universe.

    Args:
        args (GetStockUniverseInput): The input arguments for the stock universe lookup.
        context (PlanRunContext): The context of the plan run.

    Returns:
        int: company id
    """
    logger = get_prefect_logger(__name__)
    db = get_psql()

    # Find the universe id/name by reusing the stock lookup, and then filter by ETF
    etf_stock_match = await get_stock_universe_from_etf_stock_match(args, context)

    if etf_stock_match:
        stock = etf_stock_match
        sql = """
        SELECT gbi_id, spiq_company_id, name
        FROM "data".etf_universes
        WHERE gbi_id = ANY ( %s )
        """
        potential_etf_gbi_ids = [etf_stock_match["gbi_security_id"]]
        etf_rows = db.generic_read(sql, [potential_etf_gbi_ids])
        gbiid2companyid = {r["gbi_id"]: r["spiq_company_id"] for r in etf_rows}
        universe_spiq_company_id = gbiid2companyid[stock["gbi_security_id"]]
        stock["spiq_company_id"] = universe_spiq_company_id
    else:
        logger.info(f"Could not find ETF directly for '{args.universe_name}'")
        gbi_uni_row = await get_stock_universe_gbi_stock_universe(args, context)
        if gbi_uni_row:
            universe_spiq_company_id = gbi_uni_row["spiq_company_id"]
            stock = gbi_uni_row
        else:
            raise ValueError(
                f"Could not find any stock universe related to: '{args.universe_name}'"
            )

    return stock


async def get_stock_universe_list_from_universe_company_id(
    universe_spiq_company_id: int, context: PlanRunContext
) -> List[StockID]:
    """Returns the list of stock identifiers given a stock universe's company id.

    Args:
        universe_spiq_company_id: int
        context (PlanRunContext): The context of the plan run.

    Returns:
        list[StockID]: The list of stock identifiers in the universe.
    """
    db = get_psql()

    # Find the stocks in the universe
    sql = """
    SELECT DISTINCT ON (gbi_id)
    gbi_id, symbol, ms.isin, name
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


class GetRiskExposureForStocksInput(ToolArgs):
    stock_list: List[StockID]


@tool(
    description=(
        "This function takes a list of stock ids"
        " and returns a table of named factor exposure values for each stock"
    ),
    category=ToolCategory.STOCK,
    tool_registry=ToolRegistry,
    is_visible=True,
)
async def get_risk_exposure_for_stocks(
    args: GetRiskExposureForStocksInput, context: PlanRunContext
) -> Table:
    env = get_environment_tag()
    # TODO when risk model ism integration is complete
    # accept a risk model id as input and default to NA model
    # Default to SP 500
    DEV_SP500_UNIVERSE_ID = "249a293b-d9e2-4905-94c1-c53034f877c9"
    PROD_SP500_UNIVERSE_ID = "4e5f2fd3-394e-4db9-aad3-5c20abf9bf3c"

    universe_id = DEV_SP500_UNIVERSE_ID
    if env == "PROD":
        # If we are in Prod use SPY
        universe_id = PROD_SP500_UNIVERSE_ID

    exposures = await universe_stock_factor_exposures(
        user_id=context.user_id,
        universe_id=universe_id,
        # TODO default to None for now
        risk_model_id=None,
        gbi_ids=[stock.gbi_id for stock in args.stock_list],
    )

    # numpysheet/cube use str gbi_ids
    gbi_id_map = {str(stock.gbi_id): stock for stock in args.stock_list}
    factors = NumpySheet.initialize_from_proto_bytes(
        data=exposures.SerializeToString(), cols_are_dates=False
    )
    df = pd.DataFrame(factors.np_data, index=factors.rows, columns=factors.columns)
    # build the table

    df[STOCK_ID_COL_NAME_DEFAULT] = df.index

    cols = list(df)
    # move the column to head of list using index, pop and insert
    cols.insert(0, cols.pop(cols.index(STOCK_ID_COL_NAME_DEFAULT)))
    df = df.loc[:, cols]
    df[STOCK_ID_COL_NAME_DEFAULT] = df[STOCK_ID_COL_NAME_DEFAULT].map(gbi_id_map)

    table = Table.from_df_and_cols(
        data=df,
        columns=[
            TableColumnMetadata(
                label=STOCK_ID_COL_NAME_DEFAULT,
                col_type=TableColumnType.STOCK,
            ),
            TableColumnMetadata(
                label="Trading Activity",
                col_type=TableColumnType.FLOAT,
            ),
            TableColumnMetadata(
                label="Idiosyncratic",
                col_type=TableColumnType.FLOAT,
            ),
            TableColumnMetadata(
                label="Market",
                col_type=TableColumnType.FLOAT,
            ),
            TableColumnMetadata(
                label="Size",
                col_type=TableColumnType.FLOAT,
            ),
            TableColumnMetadata(
                label="Leverage",
                col_type=TableColumnType.FLOAT,
            ),
            TableColumnMetadata(
                label=VALUE_LABEL,
                col_type=TableColumnType.FLOAT,
            ),
            TableColumnMetadata(
                label=GROWTH_LABEL,
                col_type=TableColumnType.FLOAT,
            ),
            TableColumnMetadata(
                label="Momentum",
                col_type=TableColumnType.FLOAT,
            ),
            TableColumnMetadata(
                label="Volatility",
                col_type=TableColumnType.FLOAT,
            ),
        ],
    )
    return table


class GrowthFilterInput(ToolArgs):
    stock_ids: Optional[List[StockID]] = None
    min_value: float = 1


@tool(
    description=(
        "This function takes a list of stock ids"
        " and filters them acccording to how growth-y they are"
        " if no stock_list is provided, a default list will be used"
        " min_value will default to 1 standard deviation,"
        " the larger the value then the filterd stocks will be even more growthy"
    ),
    category=ToolCategory.STOCK,
    tool_registry=ToolRegistry,
    is_visible=True,
)
async def growth_filter(args: GrowthFilterInput, context: PlanRunContext) -> List[StockID]:
    stock_ids = args.stock_ids
    if stock_ids == []:
        # degenerate case should i log or throw?
        await tool_log(log="No stocks left to filter by 'growth'", context=context)
        return []

    if stock_ids is None:
        stock_uni_args = GetStockUniverseInput(universe_name="S&P 500")
        stock_ids = await get_stock_universe(stock_uni_args, context)  # type: ignore
        if not stock_ids:
            raise Exception("could not retrieve default stock list")

    if stock_ids is None:
        logger = get_prefect_logger(__name__)
        logger.info("we need universe stocks to proceed")
        return []

    risk_args = GetRiskExposureForStocksInput(stock_list=stock_ids)

    risk_table = await get_risk_exposure_for_stocks(risk_args, context)
    # mypy thinks this is not a table but a generic ComplexIO Base
    df = risk_table.to_df()  # type: ignore
    filtered_df = df.loc[df[GROWTH_LABEL] >= args.min_value]
    stocks = filtered_df[STOCK_ID_COL_NAME_DEFAULT].squeeze().to_list()
    await tool_log(log=f"Filtered {len(stock_ids)} stocks down to {len(stocks)}", context=context)
    return stocks


class ValueFilterInput(ToolArgs):
    stock_ids: Optional[List[StockID]] = None
    min_value: float = 1


@tool(
    description=(
        "This function takes a list of stock ids"
        " and filters them acccording to how value-y they are"
        " if no stock_list is provided, a default list will be used"
        " min_value will default to 1 standard deviation,"
        " the larger the value then the filtered stocks will be even more valuey"
    ),
    category=ToolCategory.STOCK,
    tool_registry=ToolRegistry,
    is_visible=True,
)
async def value_filter(args: ValueFilterInput, context: PlanRunContext) -> List[StockID]:
    stock_ids = args.stock_ids
    if stock_ids == []:
        # degenerate case should i log or throw?
        await tool_log(log="No stocks left to filter by 'value'", context=context)
        return []

    if stock_ids is None:
        stock_uni_args = GetStockUniverseInput(universe_name="S&P 500")
        stock_ids = await get_stock_universe(stock_uni_args, context)  # type: ignore
        if not stock_ids:
            raise Exception("could not retrieve default stock list")

    if stock_ids is None:
        logger = get_prefect_logger(__name__)
        logger.info("we need universe stocks to proceed")
        return []

    risk_args = GetRiskExposureForStocksInput(stock_list=stock_ids)

    risk_table = await get_risk_exposure_for_stocks(risk_args, context)

    # mypy thinks this is not a table but a generic ComplexIO Base
    df = risk_table.to_df()  # type: ignore
    filtered_df = df.loc[df[VALUE_LABEL] >= args.min_value]
    stocks = filtered_df[STOCK_ID_COL_NAME_DEFAULT].squeeze().to_list()

    await tool_log(log=f"Filtered {len(stock_ids)} stocks down to {len(stocks)}", context=context)
    return stocks


async def get_stock_universe_gbi_stock_universe(
    args: GetStockUniverseInput, context: PlanRunContext
) -> Optional[Dict]:
    """
    Returns an optional dict representing the best gbi universe match
    """
    logger = get_prefect_logger(__name__)
    db = get_psql()

    logger.info(f"looking in gbi_stock_universe for: '{args.universe_name}'")
    sql = """
    SELECT * FROM (
    SELECT etfs.spiq_company_id, etfs.name, ms.gbi_security_id, ms.symbol,
    strict_word_similarity(lower(gsu.name), lower(%s)) AS ws,
    gsu.name as gsu_name
    FROM "data".etf_universes etfs
    JOIN gbi_stock_universe gsu
    ON
    etfs.gbi_id = (gsu.ingestion_configuration->'benchmark')::INT
    JOIN master_security ms ON ms.gbi_security_id = etfs.gbi_id
    ) as tmp
    ORDER BY ws DESC
    LIMIT 20
    """
    rows = db.generic_read(sql, [args.universe_name])
    if rows:
        logger.info(f"Found {len(rows)} potential gbi universe matches for: '{args.universe_name}'")
        logger.info(f"Found gbi universe {rows[0]} for: '{args.universe_name}'")
        return rows[0]
    return None


async def get_stock_universe_from_etf_stock_match(
    args: GetStockUniverseInput, context: PlanRunContext
) -> Optional[Dict]:
    """
    Returns an optional dict representing the best ETF match
    """
    logger = get_prefect_logger(__name__)
    db = get_psql()

    # Find the universe id/name by reusing the stock lookup, and then filter by ETF
    logger.info(f"Attempting to map '{args.universe_name}' to a stock universe")
    stock_args = StockIdentifierLookupInput(stock_name=args.universe_name)

    stock_rows = await raw_stock_identifier_lookup(stock_args, context)
    if not stock_rows:
        return None

    # TODO we could extend the stock search tool to have an etf_only flag
    # filter the stock matches down to supported ETF list
    sql = """
    SELECT gbi_id, spiq_company_id, name
    FROM "data".etf_universes
    WHERE gbi_id = ANY ( %s )
    """
    potential_etf_gbi_ids = [r["gbi_security_id"] for r in stock_rows]
    etf_rows = db.generic_read(sql, [potential_etf_gbi_ids])
    gbiid2companyid = {r["gbi_id"]: r["spiq_company_id"] for r in etf_rows}

    rows = [r for r in stock_rows if r["gbi_security_id"] in gbiid2companyid]

    if not rows:
        return None

    logger.info(f"found {len(rows)} best potential matching ETFs")
    if 1 == len(rows):
        return rows[0]

    # we have multiple matches, lets use dollar trading volume to choose the most likely match
    gbiid2stocks = {r["gbi_security_id"]: r for r in rows}
    gbi_ids = list(gbiid2stocks.keys())
    stocks_sorted_by_volume = await async_sort_stocks_by_volume(gbi_ids)

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

    return stock


def get_stocks_if_bloomberg_parsekey(
    args: StockIdentifierLookupInput, context: PlanRunContext
) -> List[Dict[str, Any]]:
    """Returns the stocks with matching ticker and bloomberg exchange code
    by mapping the exchange code to a country isoc code and matching against our DB

    Examples: "IBM US", "AZN LN", "6758 JP Equity"

    Args:
        args (StockIdentifierLookupInput): The input arguments for the stock lookup.
        context (PlanRunContext): The context of the plan run.

    Returns:
        List of potential db matches
    """
    logger = get_prefect_logger(__name__)
    db = get_psql()

    search_term = args.stock_name
    search_terms = search_term.split()

    MAX_TOKENS = 2
    EXCH_CODE_LEN = 2
    MAX_SYMBOL_LEN = 8

    if len(search_terms) == MAX_TOKENS + 1 and search_terms[-1].upper() == "EQUITY":
        # if we did get the Equity 'yellow key', remove that token
        # we dont need it for searching as we only support stocks currently
        search_terms = search_terms[:2]

    symbol = search_terms[0]
    exch_code = search_terms[-1]
    if (
        len(search_terms) != MAX_TOKENS
        or len(exch_code) != EXCH_CODE_LEN
        or len(symbol) > MAX_SYMBOL_LEN
    ):
        # not a parsekey
        return []

    iso3 = bloomberg_exchange_to_country_iso3.get(exch_code)

    if not iso3:
        logger.info(
            f"either '{args.stock_name}' just looked similar to a parsekey"
            f" or we are missing an exchange code mapping for: '{exch_code}'"
        )
        return []

    sql = """
    -- ticker symbol + country (exact match only)
    SELECT gbi_security_id, ms.symbol, ms.isin, ms.security_region, ms.currency,
    ms.name, 'ticker symbol' as match_col, ms.symbol || ' ' || ms.security_region as match_text,
    1.0 AS ws
    FROM master_security ms
    WHERE
    ms.asset_type  in ('Common Stock', 'Depositary Receipt (Common Stock)')
    AND ms.is_public
    AND ms.to_z is null
    AND ms.symbol = upper(%(symbol)s)
    AND ms.security_region = upper(%(iso3)s)
    """

    rows = db.generic_read(sql, {"symbol": symbol, "iso3": iso3})
    if rows:
        logger.info("found bloomberg parsekey match")
        return rows

    logger.info(f"Looks like a bloomberg parsekey but couldn't find a match: '{args.stock_name}'")
    return []


# this was built by merging a list of bloomberg exchange codes
# and the wikipedia page for iso2/3char country codes
# this includes both specific exchanges like 'UN' = Nasdaq
# and also 'composite' exchanges like 'US' that aggregates all USA based exchanges
# sources:
# https://insights.ultumus.com/bloomberg-exchange-code-to-mic-mapping
# https://www.inforeachinc.com/bloomberg-exchange-code-mapping
# https://en.wikipedia.org/wiki/List_of_ISO_3166_country_codes
# see github for details
# https://github.com/GBI-Core/agent-service/pull/274#issuecomment-2179379326
bloomberg_exchange_to_country_iso3 = {
    # EQUITY EXCH CODE : iso3
    #           # BBG composite, iso2, name/desc
    "AL": "ALB",  # AL AL ALBANIA
    "AL": "ALB",  # AL AL ALB
    "DU": "ARE",  # DU AE NASDAQ
    "DH": "ARE",  # UH AE ABU
    "DB": "ARE",  # DU AE DFM
    "DU": "ARE",  # DU AE ARE
    "UH": "ARE",  # UH AE ARE
    "AM": "ARG",  # AR AR MENDOZA
    "AF": "ARG",  # AR AR BUENOS
    "AC": "ARG",  # AR AR BUENOS
    "AS": "ARG",  # AR AR BUENOS
    "AR": "ARG",  # AR AR ARG
    "AY": "ARM",  # AY AM NASDAQ
    "AY": "ARM",  # AY AM ARM
    "PF": "AUS",  # AU AU ASIA
    "AQ": "AUS",  # AU AU ASX
    "AH": "AUS",  # AU AU CHIX
    "SI": "AUS",  # AU AU SIM
    "AT": "AUS",  # AU AU ASE
    "AO": "AUS",  # AU AU NSX
    "AU": "AUS",  # AU AU AUS
    "AV": "AUT",  # AV AT VIENNA
    "XA": "AUT",  # EO AT CEESEG
    "AV": "AUT",  # AV AT AUT
    "AZ": "AZE",  # AZ AZ BAKU
    "AZ": "AZE",  # AZ AZ AZE
    "BB": "BEL",  # BB BE EN
    "BB": "BEL",  # BB BE BEL
    "BD": "BGD",  # BD BD DHAKA
    "BD": "BGD",  # BD BD BGD
    "BU": "BGR",  # BU BG BULGARIA
    "BU": "BGR",  # BU BG BGR
    "BI": "BHR",  # BI BH BAHRAIN
    "BI": "BHR",  # BI BH BHR
    "BM": "BHS",  # BM BS BAHAMAS
    "BM": "BHS",  # BM BS BHS
    "BK": "BIH",  # BK BA BANJA
    "BT": "BIH",  # BT BA SARAJEVO
    "BK": "BIH",  # BK BA BIH
    "BT": "BIH",  # BT BA BIH
    "RB": "BLR",  # RB BY BELARUS
    "RB": "BLR",  # RB BY BLR
    "BH": "BMU",  # BH BM BERMUDA
    "BH": "BMU",  # BH BM BMU
    "VB": "BOL",  # VB BO BOLIVIA
    "VB": "BOL",  # VB BO BOL
    "BN": "BRA",  # BZ BR SAO
    "BS": "BRA",  # BZ BR BM&FBOVESPA
    "BV": "BRA",  # BZ BR BOVESPA
    "BR": "BRA",  # BZ BR RIO
    "BO": "BRA",  # BZ BR SOMA
    "BZ": "BRA",  # BZ BR BRA
    "BA": "BRB",  # BA BB BRIDGETOWN
    "BA": "BRB",  # BA BB BRB
    "BG": "BWA",  # BG BW GABORONE
    "BG": "BWA",  # BG BW BWA
    "TX": "CAN",  # CN CA CHIX
    "DV": "CAN",  # CN CA CHIX
    "TK": "CAN",  # CN CA LIQUIDNET
    "DG": "CAN",  # CN CA LYNX
    "TR": "CAN",  # CN CA TRIACT
    "TV": "CAN",  # CN CA TRIACT
    "QF": "CAN",  # CN CA AQTS
    "QH": "CAN",  # CN CA AQRS
    "TG": "CAN",  # CN CA OMEGA
    "CJ": "CAN",  # CN CA PURE
    "TY": "CAN",  # CN CA SIGMA
    "TJ": "CAN",  # CN CA TMX
    "TN": "CAN",  # CN CA ALPHAVENTURE
    "TA": "CAN",  # CN CA ALPHATORONTO
    "CF": "CAN",  # CN CA CANADA
    "DS": "CAN",  # CN CA CX2
    "DT": "CAN",  # CN CA CX2
    "DK": "CAN",  # CN CA NASDAQ
    "DJ": "CAN",  # CN CA NASDAQ
    "TW": "CAN",  # CN CA INSTINET
    "CT": "CAN",  # CN CA TORONTO
    "CV": "CAN",  # CN CA VENTURE
    "CN": "CAN",  # CN CA CAN
    "BW": "CHE",  # SW CH BX
    "SR": "CHE",  # SW CH BERNE
    "SX": "CHE",  # SW CH SIX
    "SE": "CHE",  # SW CH SIX
    "VX": "CHE",  # VX CH SIX
    "SW": "CHE",  # SW CH CHE
    "VX": "CHE",  # VX CH CHE
    "CE": "CHL",  # CI CL SAINT
    "CC": "CHL",  # CI CL SANT.
    "CI": "CHL",  # CI CL CHL
    "C2": "CHN",  # CH CN Nrth
    "CS": "CHN",  # CH CN SHENZHEN
    "CG": "CHN",  # CH CN SHANGHAI
    "C1": "CHN",  # C1 CN Nth
    "CH": "CHN",  # CH CN CHN
    "C1": "CHN",  # C1 CN CHN
    "IA": "CIV",  # IA CI ABIDJAN
    "BC": "CIV",  # BC CI BRVM
    "ZS": "CIV",  # ZS CI SENEGAL
    "IA": "CIV",  # IA CI CIV
    "BC": "CIV",  # BC CI CIV
    "ZS": "CIV",  # ZS CI CIV
    "DE": "CMR",  # DE CM DOULASTKEXCH
    "DE": "CMR",  # DE CM CMR
    "CX": "COL",  # CB CO BOLSA
    "CB": "COL",  # CB CO COL
    "VR": "CPV",  # VR CV CAPE
    "VR": "CPV",  # VR CV CPV
    "CR": "CRI",  # CR CR COSTA
    "CR": "CRI",  # CR CR CRI
    "KY": "CYM",  # KY KY CAYMAN
    "KY": "CYM",  # KY KY CYM
    "CY": "CYP",  # CY CY NICOSIA
    "YC": "CYP",  # CY CY CYPRUS
    "CY": "CYP",  # CY CY CYP
    "CK": "CZE",  # CP CZ PRAGUE
    "CD": "CZE",  # CP CZ PRAGUE-SPAD
    "KL": "CZE",  # CP CZ PRAGUE-BLOCK
    "RC": "CZE",  # CP CZ CZECH
    "CP": "CZE",  # CP CZ CZE
    "GW": "DEU",  # GR DE STUTGT
    "PG": "DEU",  # PG DE PLUS
    "GB": "DEU",  # GR DE BERLIN
    "GC": "DEU",  # GR DE BREMEN
    "GD": "DEU",  # GR DE DUSSELDORF
    "BQ": "DEU",  # BQ DE EQUIDUCT
    "GY": "DEU",  # GR DE XETRA
    "GQ": "DEU",  # GR DE XETRA
    "GE": "DEU",  # GR DE XTRA
    "GT": "DEU",  # GR DE XETRA
    "GF": "DEU",  # GR DE FRANKFURT
    "XD": "DEU",  # EO DE DEUTSCHE
    "TH": "DEU",  # TH DE TRADEGATE
    "GH": "DEU",  # GR DE HAMBURG
    "GI": "DEU",  # GR DE HANNOVER
    "GM": "DEU",  # GR DE MUNICH
    "EX": "DEU",  # GR DE NEWEX
    "QT": "DEU",  # QT DE Quotrix
    "GS": "DEU",  # GR DE STUTTGART
    "XS": "DEU",  # EO DE STUTTGRT
    "GR": "DEU",  # GR DE DEU
    "PG": "DEU",  # PG DE DEU
    "BQ": "DEU",  # BQ DE DEU
    "TH": "DEU",  # TH DE DEU
    "QT": "DEU",  # QT DE DEU
    "DD": "DNK",  # DC DK DANSK
    "DC": "DNK",  # DC DK COPENHAGEN
    "DF": "DNK",  # DC DK FN
    "DC": "DNK",  # DC DK DNK
    "AG": "DZA",  # AG DZ ALGERIASTEXC
    "AG": "DZA",  # AG DZ DZA
    "EG": "ECU",  # ED EC GUAYAQUIL
    "EQ": "ECU",  # ED EC QUITO
    "ED": "ECU",  # ED EC ECU
    "EI": "EGY",  # EY EG NILEX
    "EC": "EGY",  # EY EG EGX
    "EY": "EGY",  # EY EG EGY
    "SB": "ESP",  # SM ES BARCELONA
    "SO": "ESP",  # SM ES BILBAO
    "SN": "ESP",  # SM ES MADRID
    "SQ": "ESP",  # SM ES CONTINUOUS
    "SA": "ESP",  # SM ES VALENCIA
    "SM": "ESP",  # SM ES ESP
    "ET": "EST",  # ET EE TALLINN
    "ET": "EST",  # ET EE EST
    "FF": "FIN",  # FH FI FN
    "FH": "FIN",  # FH FI HELSINKI
    "FH": "FIN",  # FH FI FIN
    "FS": "FJI",  # FS FJ SPSE
    "FS": "FJI",  # FS FJ FJI
    "FP": "FRA",  # FP FR PARIS
    "FP": "FRA",  # FP FR FRA
    "QX": "GBR",  # QX GB AQUIS
    "EB": "GBR",  # EB GB BATS
    "XB": "GBR",  # EO GB BOAT
    "K3": "GBR",  # K3 GB BLINK
    "B3": "GBR",  # B3 GB BLOCKMATCH
    "XV": "GBR",  # EO GB BATSChiX
    "IX": "GBR",  # IX GB CHI-X
    "XC": "GBR",  # EO GB CHI-X
    "L3": "GBR",  # L3 GB LIQUIDNET
    "ES": "GBR",  # ES GB NASDAQ
    "NQ": "GBR",  # NQ GB NASDAQ
    "S1": "GBR",  # S1 GB SIGMA
    "A0": "GBR",  # A0 GB ASSETMATCH
    "DX": "GBR",  # DX GB TURQUOISE
    "TQ": "GBR",  # TQ GB TURQUOISE
    "LD": "GBR",  # LD GB NYSE
    "LN": "GBR",  # LN GB LONDON
    "LI": "GBR",  # LI GB LONDON
    "EU": "GBR",  # EU GB EUROPEAN
    "E1": "GBR",  # EO GB EURO
    "XL": "GBR",  # EO GB LONDON
    "LO": "GBR",  # LI GB LSE
    "PZ": "GBR",  # PZ GB ISDX
    "XP": "GBR",  # EO GB PLUS
    "XE": "GBR",  # EO GB EURONEXT
    "S2": "GBR",  # S2 GB UBS
    "QX": "GBR",  # QX GB GBR
    "EB": "GBR",  # EB GB GBR
    "K3": "GBR",  # K3 GB GBR
    "B3": "GBR",  # B3 GB GBR
    "IX": "GBR",  # IX GB GBR
    "L3": "GBR",  # L3 GB GBR
    "ES": "GBR",  # ES GB GBR
    "NQ": "GBR",  # NQ GB GBR
    "S1": "GBR",  # S1 GB GBR
    "A0": "GBR",  # A0 GB GBR
    "DX": "GBR",  # DX GB GBR
    "TQ": "GBR",  # TQ GB GBR
    "LD": "GBR",  # LD GB GBR
    "LN": "GBR",  # LN GB GBR
    "LI": "GBR",  # LI GB GBR
    "EU": "GBR",  # EU GB GBR
    "PZ": "GBR",  # PZ GB GBR
    "S2": "GBR",  # S2 GB GBR
    "GG": "GEO",  # GG GE JSCGEORGIA
    "GG": "GEO",  # GG GE GEO
    "GU": "GGY",  # GU GG GUERNSEY
    "JY": "GGY",  # JY GG JERSEY
    "GU": "GGY",  # GU GG GGY
    "JY": "GGY",  # JY GG GGY
    "GN": "GHA",  # GN GH ACCRA
    "GN": "GHA",  # GN GH GHA
    "TL": "GIB",  # TL GI GIBRALTAR
    "TL": "GIB",  # TL GI GIB
    "AA": "GRC",  # GA GR ATHENS
    "XT": "GRC",  # EO GR ATHENS
    "AP": "GRC",  # GA GR ATHENS
    "GA": "GRC",  # GA GR ATHENS
    "GA": "GRC",  # GA GR GRC
    "GL": "GTM",  # GL GT GUATEMALA
    "GL": "GTM",  # GL GT GTM
    "H1": "HKG",  # H1 HK Sth
    "H2": "HKG",  # HK HK Sth
    "HK": "HKG",  # HK HK HONG
    "H1": "HKG",  # H1 HK HKG
    "HK": "HKG",  # HK HK HKG
    "HO": "HND",  # HO HN HONDURAS
    "HO": "HND",  # HO HN HND
    "ZA": "HRV",  # CZ HR ZAGREB
    "CZ": "HRV",  # CZ HR HRV
    "QM": "HUN",  # QM HU QUOTE
    "HB": "HUN",  # HB HU BUDAPEST
    "XH": "HUN",  # EO HU BUDAPEST
    "QM": "HUN",  # QM HU HUN
    "HB": "HUN",  # HB HU HUN
    "IJ": "IDN",  # IJ ID INDONESIA
    "IJ": "IDN",  # IJ ID IDN
    "IG": "IND",  # IN IN MCX
    "IB": "IND",  # IN IN BSE
    "IH": "IND",  # IN IN DELHI
    "IS": "IND",  # IN IN NATL
    "IN": "IND",  # IN IN IND
    "ID": "IRL",  # ID IE IRELAND
    "XF": "IRL",  # EO IE DUBLIN
    "PO": "IRL",  # PO IE ITG
    "ID": "IRL",  # ID IE IRL
    "PO": "IRL",  # PO IE IRL
    "IE": "IRN",  # IE IR TEHRAN
    "IE": "IRN",  # IE IR IRN
    "IQ": "IRQ",  # IQ IQ IRAQ
    "IQ": "IRQ",  # IQ IQ IRQ
    "RF": "ISL",  # IR IS FN
    "IR": "ISL",  # IR IS REYKJAVIK
    "IR": "ISL",  # IR IS ISL
    "IT": "ISR",  # IT IL TEL
    "IT": "ISR",  # IT IL ISR
    "TE": "ITA",  # TE IT EUROTLX
    "HM": "ITA",  # HM IT HI-MTF
    "IM": "ITA",  # IM IT BRSAITALIANA
    "IC": "ITA",  # IM IT MIL
    "XI": "ITA",  # EO IT BORSAITALOTC
    "IF": "ITA",  # IM IT MIL
    "TE": "ITA",  # TE IT ITA
    "HM": "ITA",  # HM IT ITA
    "IM": "ITA",  # IM IT ITA
    "JA": "JAM",  # JA JM KINGSTON
    "JA": "JAM",  # JA JM JAM
    "JR": "JOR",  # JR JO AMMAN
    "JR": "JOR",  # JR JO JOR
    "JI": "JPN",  # JP JP CHI-X
    "JD": "JPN",  # JP JP KABU.COM
    "JE": "JPN",  # JP JP SBIJAPANNEXT
    "JW": "JPN",  # JP JP SBIJNxt
    "JF": "JPN",  # JP JP FUKUOKA
    "JQ": "JPN",  # JP JP JASDAQ
    "JN": "JPN",  # JP JP NAGOYA
    "JO": "JPN",  # JP JP OSAKA
    "JS": "JPN",  # JP JP SAPPORO
    "JU": "JPN",  # JP JP SBI
    "JG": "JPN",  # JP JP TOKYO
    "JT": "JPN",  # JP JP TOKYO
    "JP": "JPN",  # JP JP JPN
    "KZ": "KAZ",  # KZ KZ KAZAKHSTAN
    "KZ": "KAZ",  # KZ KZ KAZ
    "KN": "KEN",  # KN KE NAIROBI
    "KN": "KEN",  # KN KE KEN
    "KB": "KGZ",  # KB KG KYRGYZSTAN
    "KB": "KGZ",  # KB KG KGZ
    "KH": "KHM",  # KH KH CAMBODIA
    "KH": "KHM",  # KH KH KHM
    "EK": "KNA",  # EK KN ESTN
    "AI": "KNA",  # AI KN ANGUILLA
    "NX": "KNA",  # NX KN ST
    "EK": "KNA",  # EK KN KNA
    "AI": "KNA",  # AI KN KNA
    "NX": "KNA",  # NX KN KNA
    "KF": "KOR",  # KF KR KOREAFRBMKT
    "KE": "KOR",  # KS KR KONEX
    "KP": "KOR",  # KS KR KOREA
    "KQ": "KOR",  # KS KR KOSDAQ
    "KF": "KOR",  # KF KR KOR
    "KS": "KOR",  # KS KR KOR
    "KK": "KWT",  # KK KW KUWAIT
    "KK": "KWT",  # KK KW KWT
    "LS": "LAO",  # LS LA LAOS
    "LS": "LAO",  # LS LA LAO
    "LB": "LBN",  # LB LB BEIRUT
    "LB": "LBN",  # LB LB LBN
    "LY": "LBY",  # LY LY LIBYANSTEXC
    "LY": "LBY",  # LY LY LBY
    "SL": "LKA",  # SL LK COLOMBO
    "SL": "LKA",  # SL LK LKA
    "LH": "LTU",  # LH LT VILNIUS
    "LH": "LTU",  # LH LT LTU
    "LX": "LUX",  # LX LU LUXEMBOURG
    "LX": "LUX",  # LX LU LUX
    "LG": "LVA",  # LR LV RIGA
    "LR": "LVA",  # LR LV LVA
    "MC": "MAR",  # MC MA CASABLANCA
    "MC": "MAR",  # MC MA MAR
    "MB": "MDA",  # MB MD MOLDOVA
    "MB": "MDA",  # MB MD MDA
    "MX": "MDV",  # MX MV MALDIVES
    "MX": "MDV",  # MX MV MDV
    "MM": "MEX",  # MM MX MEXICO
    "MM": "MEX",  # MM MX MEX
    "MS": "MKD",  # MS MK MACEDONIA
    "MS": "MKD",  # MS MK MKD
    "MV": "MLT",  # MV MT VALETTA
    "MV": "MLT",  # MV MT MLT
    "ME": "MNE",  # ME ME MONTENEGRO
    "ME": "MNE",  # ME ME MNE
    "MO": "MNG",  # MO MN MONGOLIA
    "MO": "MNG",  # MO MN MNG
    "MZ": "MOZ",  # MZ MZ MAPUTO
    "MZ": "MOZ",  # MZ MZ MOZ
    "MP": "MUS",  # MP MU SEM
    "MP": "MUS",  # MP MU MUS
    "MW": "MWI",  # MW MW MALAWI
    "MW": "MWI",  # MW MW MWI
    "MQ": "MYS",  # MQ MY MESDAQ
    "MK": "MYS",  # MK MY BURSA
    "MQ": "MYS",  # MQ MY MYS
    "MK": "MYS",  # MK MY MYS
    "NW": "NAM",  # NW NA WINDHOEK
    "NW": "NAM",  # NW NA NAM
    "NL": "NGA",  # NL NG LAGOS
    "NL": "NGA",  # NL NG NGA
    "NC": "NIC",  # NC NI NICARAGUA
    "NC": "NIC",  # NC NI NIC
    "MT": "NLD",  # MT NL TOM
    "NA": "NLD",  # NA NL EN
    "NR": "NLD",  # NR NL NYSE
    "MT": "NLD",  # MT NL NLD
    "NA": "NLD",  # NA NL NLD
    "NR": "NLD",  # NR NL NLD
    "NS": "NOR",  # NO NO NORWAY
    "NO": "NOR",  # NO NO OSLO
    "XN": "NOR",  # EO NO OSLO
    "NO": "NOR",  # NO NO NOR
    "NO": "NOR",  # NO NO NOR
    "NK": "NPL",  # NK NP NEPAL
    "NK": "NPL",  # NK NP NPL
    "NZ": "NZL",  # NZ NZ NZX
    "NZ": "NZL",  # NZ NZ NZL
    "OM": "OMN",  # OM OM MUSCAT
    "OM": "OMN",  # OM OM OMN
    "PK": "PAK",  # PA PK KARACHI
    "PA": "PAK",  # PA PK PAK
    "PP": "PAN",  # PP PA PANAMA
    "PP": "PAN",  # PP PA PAN
    "PE": "PER",  # PE PE LIMA
    "PE": "PER",  # PE PE PER
    "PM": "PHL",  # PM PH PHILIPPINES
    "PM": "PHL",  # PM PH PHL
    "PB": "PNG",  # PB PG PORT
    "PB": "PNG",  # PB PG PNG
    "PD": "POL",  # PW PL POLAND
    "PW": "POL",  # PW PL WARSAW
    "PW": "POL",  # PW PL POL
    "PX": "PRT",  # PX PT PEX
    "PL": "PRT",  # PL PT EN
    "PX": "PRT",  # PX PT PRT
    "PL": "PRT",  # PL PT PRT
    "PN": "PRY",  # PN PY ASUNCION
    "PN": "PRY",  # PN PY PRY
    "PS": "PSE",  # PS PS PALESTINE
    "PS": "PSE",  # PS PS PSE
    "QD": "QAT",  # QD QA QATAR
    "QD": "QAT",  # QD QA QAT
    "RZ": "ROU",  # RO RO SIBEX
    "RE": "ROU",  # RO RO BUCHAREST
    "RQ": "ROU",  # RO RO RASDAQ
    "RO": "ROU",  # RO RO ROU
    "RX": "RUS",  # RM RU MICEX
    "RN": "RUS",  # RM RU MICEX
    "RP": "RUS",  # RM RU MICEX
    "RR": "RUS",  # RU RU RTS
    "RT": "RUS",  # RU RU NP
    "RM": "RUS",  # RM RU RUS
    "RU": "RUS",  # RU RU RUS
    "RW": "RWA",  # RW RW RWANDA
    "RW": "RWA",  # RW RW RWA
    "AB": "SAU",  # AB SA SAUDI
    "AB": "SAU",  # AB SA SAU
    "SP": "SGP",  # SP SG SINGAPORE
    "SP": "SGP",  # SP SG SGP
    "EL": "SLV",  # EL SV EL
    "EL": "SLV",  # EL SV SLV
    "SG": "SRB",  # SG RS BELGRADE
    "SG": "SRB",  # SG RS SRB
    "SK": "SVK",  # SK SK BRATISLAVA
    "SK": "SVK",  # SK SK SVK
    "SV": "SVN",  # SV SI LJUBLJANA
    "XJ": "SVN",  # EO SI LJUB
    "SV": "SVN",  # SV SI SVN
    "BY": "SWE",  # BY SE BURGUNDY
    "SF": "SWE",  # SS SE FN
    "NG": "SWE",  # SS SE
    "XG": "SWE",  # EO SE NGM
    "XO": "SWE",  # EO SE OMX
    "KA": "SWE",  # SS SE AKTIE
    "SS": "SWE",  # SS SE NORDIC
    "BY": "SWE",  # BY SE SWE
    "SS": "SWE",  # SS SE SWE
    "SD": "SWZ",  # SD SZ MBABANE
    "SD": "SWZ",  # SD SZ SWZ
    "SZ": "SYC",  # SZ SC Seychelles
    "SZ": "SYC",  # SZ SC SYC
    "SY": "SYR",  # SY SY DAMASCUS
    "SY": "SYR",  # SY SY SYR
    "TB": "THA",  # TB TH BANGKOK
    "TB": "THA",  # TB TH THA
    "TP": "TTO",  # TP TT PORT
    "TP": "TTO",  # TP TT TTO
    "TU": "TUN",  # TU TN TUNIS
    "TU": "TUN",  # TU TN TUN
    "TI": "TUR",  # TI TR ISTANBUL
    "TF": "TUR",  # TI TR ISTN
    "TS": "TUR",  # TI TR ISTN
    "TI": "TUR",  # TI TR TUR
    "TT": "TWN",  # TT TW GRETAI
    "TT": "TWN",  # TT TW TAIWAN
    "TT": "TWN",  # TT TW TWN
    "TZ": "TZA",  # TZ TZ DAR
    "TZ": "TZA",  # TZ TZ TZA
    "UG": "UGA",  # UG UG UGANDA
    "UG": "UGA",  # UG UG UGA
    "UZ": "UKR",  # UZ UA PFTS
    "QU": "UKR",  # UZ UA PFTS
    "UK": "UKR",  # UZ UA RTS
    "UZ": "UKR",  # UZ UA UKR
    "UY": "URY",  # UY UY MONTEVIDEO
    "UY": "URY",  # UY UY URY
    "UP": "USA",  # US US NYSE
    "UF": "USA",  # US US BATS
    "VY": "USA",  # US US BATS
    "UO": "USA",  # US US CBSX
    "VJ": "USA",  # US US EDGA
    "VK": "USA",  # US US EDGX
    "UI": "USA",  # US US ISLAND
    "VF": "USA",  # US US INVESTOR
    "UV": "USA",  # US US OTC
    "PQ": "USA",  # US US OTC
    "UD": "USA",  # US US FINRA
    "UA": "USA",  # US US NYSE
    "UB": "USA",  # US US NSDQ
    "UM": "USA",  # US US CHICAGO
    "UC": "USA",  # US US NATIONAL
    "UL": "USA",  # US US ISE
    "UR": "USA",  # US US NASDAQ
    "UW": "USA",  # US US NASDAQ
    "UT": "USA",  # US US NASDAQ
    "UQ": "USA",  # US US NASDAQ
    "UN": "USA",  # US US NYSE
    "UU": "USA",  # US US OTC
    "UX": "USA",  # US US NSDQ
    "US": "USA",  # US US USA
    "ZU": "UZB",  # ZU UZ UZBEKISTAN
    "ZU": "UZB",  # ZU UZ UZB
    "VS": "VEN",  # VC VE CARACAS
    "VC": "VEN",  # VC VE VEN
    "VH": "VNM",  # VN VN HANOI
    "VU": "VNM",  # VN VN HANOI
    "VM": "VNM",  # VN VN HO
    "VN": "VNM",  # VN VN VNM
    "SJ": "ZAF",  # SJ ZA JOHANNESBURG
    "SJ": "ZAF",  # SJ ZA ZAF
    "ZL": "ZMB",  # ZL ZM LUSAKA
    "ZL": "ZMB",  # ZL ZM ZMB
    "ZH": "ZWE",  # ZH ZW HARARE
    "ZH": "ZWE",  # ZH ZW ZWE
}
