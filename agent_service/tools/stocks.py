# Author(s): Mohammad Zarei, David Grohmann
import json
from collections import defaultdict
from typing import Any, Dict, List, Optional, cast

import pandas as pd
from gbi_common_py_utils.numpy_common import NumpySheet
from gbi_common_py_utils.utils.environment import get_environment_tag

from agent_service.external.pa_backtest_svc_client import (
    universe_stock_factor_exposures,
)
from agent_service.external.stock_search_dao import async_sort_stocks_by_volume
from agent_service.GPT.constants import FILTER_CONCURRENCY, GPT35_TURBO, NO_PROMPT
from agent_service.GPT.requests import GPT
from agent_service.io_type_utils import TableColumnType
from agent_service.io_types.stock import StockID
from agent_service.io_types.stock_aligned_text import StockAlignedTextGroups
from agent_service.io_types.table import Table, TableColumnMetadata
from agent_service.io_types.text import Text
from agent_service.tool import ToolArgs, ToolCategory, ToolRegistry, tool
from agent_service.tools.stock_metadata import (
    GetStockDescriptionInput,
    get_company_descriptions,
)
from agent_service.tools.tool_log import tool_log
from agent_service.types import PlanRunContext
from agent_service.utils.async_utils import gather_with_concurrency
from agent_service.utils.gpt_logging import GptJobIdType, GptJobType, create_gpt_context
from agent_service.utils.logs import async_perf_logger
from agent_service.utils.postgres import get_psql
from agent_service.utils.prefect import get_prefect_logger
from agent_service.utils.prompt_utils import Prompt
from agent_service.utils.string_utils import (
    clean_to_json_if_needed,
    repair_json_if_needed,
)

STOCK_ID_COL_NAME_DEFAULT = "Security"
GROWTH_LABEL = "Growth"
VALUE_LABEL = "Value"

# constants used to massage the match scores for stocks to be in a good order

# this is needed to insert a perfect match score when we do an exact match sql
# and a score isn't automatically generated for us
PERFECT_TEXT_MATCH = 1.0

# matches that dont agree on the first word are less likely to be good
NON_PREFIX_PENALTY = 0.7

# SP CAPIQ's alt name data is useful to fill in the gaps
# but is not extremely consistent
# So we penalize matches on it vs the matches on the official name or boosted alt names
#
# Example: spiq has an alt name of 'Tesla' for this unheard of company
# 'gbi_security_id'	'symbol'	'isin'	'name'	'spiq_company_id'
# 25370	'TXL'	'CA8816011081'	'Tesla Exploration Ltd.'	30741740

# but not for the one you are thinking of.
# 25508	'TSLA'	'US88160R1014'	'Tesla, Inc.'	27444752
SPIQ_ALT_NAME_PENALTY = 0.8

# Any matches below this level will be discarded
MIN_ACCEPTABLE_MATCH_SCORE = 0.2

# when we are taking a GPT answer and pulling a gbi id using the name from gpt
# we want a confident text match
MIN_GPT_NAME_MATCH = 0.7


class StockIdentifierLookupInput(ToolArgs):
    # name or symbol of the stock to lookup
    stock_name: str


STOCK_CONFIRMATION_PROMPT = Prompt(
    name="STOCK_CONFIRMATION_PROMPT",
    template="""
Your task is to determine which company, stock, ETF or stock index the search term refers to,
 the search term might be a common word,
 but in this case you should try really hard to associate it with a stock or etf
 If the search term does NOT refer to a company, stock, ETF or stock index,
 then return an empty json: '{{}}'
 If the search term DOES refer to a company, stock, ETF or stock index, then return
 a json with at least the full_name field.
 Optionally, it is not required but it would be helpful if you
 can also return optional fields: ticker_symbol, isin, common_name, match_type, and country_iso3,
 match_type is 'stock', 'etf', or 'index'
 country_iso3 is the 3 letter iso country code where it is most often traded or originated from
 common_name is a name people would typically use to refer to it,
 it would often be a substring of the full_name (dropping common suffixes for example),
 an acronym, an abreviation, or a nickname.
 Only return each of the optional fields if you are confident they are correct,
 it is ok if you do not know some or all of them.
 you can return some and not others, the only required field is full_name.
 in the special case of a stock index, you should return the most popular ETF
 that is tracking that index, instead of the index itself
 for example SP500 should return the ETF: SPY.
 You should also return a confidence number between 1-10 :
 where 10 is extremely confident and 1 is a hallucinated random guess.
 You must also return a reason field to
 explain why the answer is a good match for the search term.
 You will also be given a list of potential search results in json format
 from another search system.
 The search results may or may not contain the correct entity,
 but you should consider them as a potential answer.
 If you are picking from the list, you should have a slight preference for
 matches with a higher "volume" field,
 you should prefer to pick stocks with volume greater than 1000000
 and stocks with volume less than 1000000 should be avoided
 If you strongly believe that none of the potential search results are correct,
 then please suggest your own match.
 If the search term is a product name or brand name,
 it is extremely unlikely to be in the list and you will need to suggest your own match
 for the company that is most associated with that product or brand.

 Make sure to return this in JSON.
 ONLY RETURN IN JSON. DO NOT RETURN NON-JSON.
 The potential search results in json are:
 {results}
 The search term is: '{search}'
 REMEMBER! If your reason is something like: "the search term closely resembles the ticker symbol"
 then you should reconsider as ticker typos are not common, and your answer is
 extremely likely to be wrong.
 In that case you should find a different answer with a better reason.
 You will be fired if your reason is close to: "the search term closely resembles the ticker symbol"
""",
)


@async_perf_logger
async def stock_confirmation_by_gpt(
    context: PlanRunContext, search: str, results: List[Dict]
) -> Optional[Dict[str, Any]]:
    logger = get_prefect_logger(__name__)
    gpt_context = create_gpt_context(
        GptJobType.AGENT_TOOLS, context.agent_id, GptJobIdType.AGENT_ID
    )
    # we can try different models
    # HAIKU, SONNET, GPT4_TURBO,
    llm = GPT(context=gpt_context, model=GPT35_TURBO)

    prompt = STOCK_CONFIRMATION_PROMPT.format(search=search, results=json.dumps(results))

    result_str = await llm.do_chat_w_sys_prompt(
        main_prompt=prompt,
        sys_prompt=NO_PROMPT,
        output_json=True,
    )

    result_str = result_str.strip().lower()
    clean_result_str = clean_to_json_if_needed(result_str, repair=False)
    result_dict = repair_json_if_needed(clean_result_str, json_load=True)

    logger.info(f"search: '{search}', gpt {result_dict=}")
    if isinstance(result_dict, dict):
        return result_dict

    return None


@tool(
    description=(
        "This function takes a string (microsoft, apple, AAPL, TESLA, META, e.g.) "
        "which refers to a stock, and converts it to an integer identifier."
    ),
    category=ToolCategory.STOCK,
    tool_registry=ToolRegistry,
    is_visible=False,
)
@async_perf_logger
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

    # first we check if the search string is in a format that leads to an unambiguous match
    exact_rows = await stock_lookup_exact(args, context)
    if exact_rows:
        logger.info(f"found {len(exact_rows)} exact matches")
        if len(exact_rows) > 1:
            exact_rows = await augment_stock_rows_with_volume(exact_rows)
            exact_rows = sorted(exact_rows, key=lambda x: x.get("volume", 0), reverse=True)
        stock = exact_rows[0]
        logger.info(f"found exact match {stock=}")
        return StockID(
            gbi_id=stock["gbi_security_id"],
            symbol=stock["symbol"],
            isin=stock["isin"],
            company_name=stock["name"],
        )

    # next we check for best matches by text similarity
    rows = await stock_lookup_by_text_similarity(args, context)
    logger.info(f"found {len(rows)} best potential matching stocks")
    rows = await augment_stock_rows_with_volume(rows)

    orig_stocks_sorted_by_match = sorted(rows, key=lambda x: x["final_match_score"], reverse=True)
    orig_stocks_sorted_by_volume = sorted(
        rows, key=lambda x: (x.get("volume", 0), x["final_match_score"]), reverse=True
    )

    # send the top 3 by text match and volume to GPT for review
    MAX_MATCH_STOCKS = 3
    MAX_VOLUME_STOCKS = 3
    ask_gpt = {}
    # this sql is a UNION so it can have multiple copies of the same stock,
    # so only keep the strongest matching one by inserting in match strength order first
    for x in orig_stocks_sorted_by_match[:MAX_MATCH_STOCKS]:
        if x["gbi_security_id"] not in ask_gpt:
            ask_gpt[x["gbi_security_id"]] = x

    for x in orig_stocks_sorted_by_volume[:MAX_VOLUME_STOCKS]:
        if x["gbi_security_id"] not in ask_gpt:
            ask_gpt[x["gbi_security_id"]] = x

    search = args.stock_name
    logger.info(
        f"{search=} , sending these possible matches to gpt: " f"{json.dumps(ask_gpt, indent=4)}"
    )
    gpt_answer = await stock_confirmation_by_gpt(
        context, search=args.stock_name, results=list(ask_gpt.values())
    )

    if gpt_answer and gpt_answer.get("full_name"):
        # map the GPT answer back to a gbi_id

        gpt_answer_full_name = cast(str, gpt_answer["full_name"])

        gpt_stock: Optional[Dict[str, Any]] = None
        # first check if gpt answer is in the original set:
        for s in orig_stocks_sorted_by_volume:
            if gpt_answer_full_name.lower() == s.get("name", "").lower():
                gpt_stock = s
                logger.info(f"found gpt answer in original set: {gpt_stock=}, {gpt_answer=}")
                break

        if gpt_stock:
            confirm_rows = [gpt_stock]
        else:
            # next do a full text search to map the gpt answer back to a gbi_id
            confirm_args = StockIdentifierLookupInput(stock_name=gpt_answer_full_name)
            confirm_rows = await stock_lookup_by_text_similarity(
                confirm_args, context, min_match_strength=MIN_GPT_NAME_MATCH
            )
            logger.info(f"found {len(rows)} best matching stock to the gpt answer")
            confirm_rows = await augment_stock_rows_with_volume(confirm_rows)

    else:
        # TODO, should we assume GPT is correct that this is not a stock,
        # or should we instead use our best db match?
        confirm_rows = []

    if confirm_rows:
        stock = confirm_rows[0]
        confirm_stocks_sorted_by_volume = sorted(
            confirm_rows, key=lambda x: (x.get("volume", 0), x["final_match_score"]), reverse=True
        )
        # we have multiple matches, lets use dollar trading volume to choose the most likely match
        if confirm_stocks_sorted_by_volume[0].get("volume"):
            stock = confirm_stocks_sorted_by_volume[0]
    elif orig_stocks_sorted_by_volume and orig_stocks_sorted_by_volume[0].get("volume"):
        stock = orig_stocks_sorted_by_volume[0]
    elif orig_stocks_sorted_by_match:
        stock = orig_stocks_sorted_by_match[0]
    else:
        raise ValueError(f"Could not find any stocks related to: '{args.stock_name}'")

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


async def stock_lookup_by_exact_gbi_alt_name(
    args: StockIdentifierLookupInput, context: PlanRunContext
) -> List[Dict[str, Any]]:
    """Returns the stocks with an exact match to gbi id alt names table

    Args:
        args (StockIdentifierLookupInput): The input arguments for the stock lookup.
        context (PlanRunContext): The context of the plan run.

    Returns:
        List[Dict[str, Any]]: DB rows representing the potentially matching stocks.
    """
    logger = get_prefect_logger(__name__)
    db = get_psql()

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
    AND source_id = 0 -- boosted custom alt_name entries
    """
    rows = db.generic_read(sql, {"search_term": args.stock_name})
    if rows:
        logger.info("found exact gbi alt name")
        return rows

    return []


async def stock_lookup_by_isin(
    args: StockIdentifierLookupInput, context: PlanRunContext
) -> List[Dict[str, Any]]:
    """Returns the stocks whose ISIN exactly matches the search string

    Args:
        args (StockIdentifierLookupInput): The input arguments for the stock lookup.
        context (PlanRunContext): The context of the plan run.

    Returns:
        List[Dict[str, Any]]: DB rows representing the potentially matching stocks.
    """
    logger = get_prefect_logger(__name__)
    db = get_psql()

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

    return []


async def augment_stock_rows_with_volume(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Returns the input row dicts augmented with a new 'volume' field

    Returns:
        List[Dict[str, Any]]: DB rows representing the potentially matching stocks.
    """
    logger = get_prefect_logger(__name__)
    gbiid2stocks = {r["gbi_security_id"]: r for r in rows}
    gbi_ids = list(gbiid2stocks.keys())
    stocks_sorted_by_volume = await async_sort_stocks_by_volume(gbi_ids)

    if stocks_sorted_by_volume:
        logger.info(f"Top stock volumes: {stocks_sorted_by_volume[:10]}")
        for gbi_id, volume in stocks_sorted_by_volume:
            stock = gbiid2stocks.get(gbi_id)
            if stock:
                stock["volume"] = volume
            else:
                logger.warning("Logic error!")
                # should not be possible
    return rows


@async_perf_logger
async def stock_lookup_by_text_similarity(
    args: StockIdentifierLookupInput,
    context: PlanRunContext,
    min_match_strength: float = MIN_ACCEPTABLE_MATCH_SCORE,
) -> List[Dict[str, Any]]:
    """Returns the stocks with the closest text similarity match in the various name fields
    to the input string
    such as name, isin or symbol (JP3633400001, microsoft, apple, AAPL, TESLA, META, e.g.).

    Args:
        args (StockIdentifierLookupInput): The input arguments for the stock lookup.
        context (PlanRunContext): The context of the plan run.
        min_match_strength (float 0.0-1.0): the minimum text match score

    Returns:
        List[Dict[str, Any]]: DB rows representing the potentially matching stocks.
    """
    logger = get_prefect_logger(__name__)
    db = get_psql()

    prefix = args.stock_name.split()[0]
    # often the most important word in a company name is the first word,
    # Samsung electronics, co. ltd, Samsung is important everything after is not
    logger.info(f"checking for text similarity of {args.stock_name=} and {prefix=}")
    # Word similarity name match

    # should we also allow 'Depositary Receipt (Common Stock)') ?
    sql = f"""
    SELECT *,

    -- we have to use the trigram indexes to speed up the qry
    -- but need a low enough value so we can find the stuff we are looking for
    -- should just be this SET cmd but it fails
    -- SET pg_trgm.similarity_threshold = 0.2;
    -- use set_limit() instead
    set_limit(0.2)
    FROM (
    SELECT *,
    -- penalize non-prefix matches
    CASE
        WHEN match_text ilike  %(prefix)s || '%%' THEN text_sim_score
    ELSE text_sim_score * {NON_PREFIX_PENALTY}
    END
    AS final_match_score
 FROM (

    -- ticker symbol (exact match only)
    SELECT gbi_security_id, ms.symbol, ms.isin, ms.security_region, ms.currency,
    ms.name, 'ticker symbol' as match_col, ms.symbol as match_text,
    {PERFECT_TEXT_MATCH} AS text_sim_score
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
    (strict_word_similarity(ms.name, %(search_term)s) +
    strict_word_similarity(%(search_term)s, ms.name)) / 2
    AS text_sim_score
    FROM master_security ms
    WHERE
    ms.asset_type  in ('Common Stock', 'Depositary Receipt (Common Stock)')
    AND ms.is_public
    AND ms.is_primary_trading_item = true
    AND ms.to_z is null

    -- this uses the trigram index which speeds up the qry
    -- https://www.postgresql.org/docs/current/pgtrgm.html#PGTRGM-OP-TABLE
    AND name %% %(search_term)s
    AND %(search_term)s %% name

    UNION

    -- custom boosted db entries -  company alt name * 1.0
    SELECT gbi_security_id, symbol, ms.isin, ms.security_region, ms.currency,
    name, 'comp alt name' as match_col, alt_name as match_text,

    -- lower the score for spiq matches
    CASE
        WHEN can.source_id = 0
        THEN     (strict_word_similarity(alt_name, %(search_term)s) +
                  strict_word_similarity(%(search_term)s, alt_name)) / 2
    ELSE (strict_word_similarity(alt_name, %(search_term)s) +
          strict_word_similarity(%(search_term)s, alt_name)) / 2 * {SPIQ_ALT_NAME_PENALTY}
    END
    AS text_sim_score
    FROM master_security ms
    JOIN spiq_security_mapping ssm ON ssm.gbi_id = ms.gbi_security_id
    JOIN "data".company_alt_names can ON ssm.spiq_company_id = can.spiq_company_id
    WHERE
    ms.asset_type  in ('Common Stock', 'Depositary Receipt (Common Stock)')
    AND can.enabled
    AND ms.is_public
    AND ms.is_primary_trading_item = true
    AND ms.to_z is null
    AND alt_name %% %(search_term)s
    AND %(search_term)s %% alt_name

    UNION

    -- gbi alt name
    SELECT gbi_security_id, symbol, ms.isin, ms.security_region, ms.currency,
    name, 'gbi alt name' as match_col, alt_name as match_text,
    (strict_word_similarity(alt_name, %(search_term)s) +
    strict_word_similarity(%(search_term)s, alt_name)) / 2
    AS text_sim_score
    FROM master_security ms
    JOIN "data".gbi_id_alt_names gan ON gan.gbi_id = ms.gbi_security_id
    WHERE
    ms.asset_type  in ('Common Stock', 'Depositary Receipt (Common Stock)')
    AND gan.enabled
    AND ms.is_public
    AND ms.is_primary_trading_item = true
    AND ms.to_z is null
    AND alt_name %% %(search_term)s
    AND %(search_term)s %% alt_name

    ORDER BY text_sim_score DESC
    LIMIT 200
    ) as text_scores
    ) as final_scores -- word similarity score
    WHERE
    final_scores.final_match_score >= {min_match_strength}  -- score including prefix match
    ORDER BY final_match_score DESC
    LIMIT 100
    """
    rows = db.generic_read(sql, {"search_term": args.stock_name, "prefix": prefix})
    if rows:
        # the weaker the match the more results to be
        # considered for trading volume tie breaker
        matches = [r for r in rows if r["final_match_score"] >= 0.9]
        if matches:
            matches = [r for r in rows if r["final_match_score"] >= 0.60]
            logger.info(f"found {len(matches)} very strong matches: {matches[:4]}")
            return matches[:20]

        matches = [r for r in rows if r["final_match_score"] >= 0.6]
        if matches:
            matches = [r for r in rows if r["final_match_score"] >= 0.50]
            logger.info(f"found {len(matches)} strong matches: {matches[:4]}")
            return matches[:20]

        matches = [r for r in rows if r["final_match_score"] >= 0.4]
        if matches:
            matches = [r for r in rows if r["final_match_score"] >= 0.30]
            logger.info(f"found {len(matches)} medium matches")
            return matches[:30]

        matches = [r for r in rows if r["final_match_score"] >= 0.3]
        if matches:
            matches = [r for r in rows if r["final_match_score"] >= 0.20]
            logger.info(f"found {len(matches)} weak matches")
            return matches[:40]

        # very weak text match
        matches = [r for r in rows if r["final_match_score"] > 0.2]
        if matches:
            logger.info(f"found {len(matches)} very weak matches")
            return matches[:50]

        logger.info(
            f"{args.stock_name=} found {len(rows)} potential matches but "
            "they were all likely unrelated to the user intent. "
            f"here are a few: {rows[:4]}"
        )

    logger.info(f"{args.stock_name=} found no textual matches")
    return []


async def stock_lookup_exact(
    args: StockIdentifierLookupInput, context: PlanRunContext
) -> List[Dict[str, Any]]:
    """Returns the stocks with an exact match to various fields that are unambiguous

    Args:
        args (StockIdentifierLookupInput): The input arguments for the stock lookup.
        context (PlanRunContext): The context of the plan run.

    Returns:
        List[Dict[str, Any]]: DB rows representing the potentially matching stocks.
    """
    logger = get_prefect_logger(__name__)

    bloomberg_rows = await stock_lookup_by_bloomberg_parsekey(args, context)
    if bloomberg_rows:
        logger.info("found bloomberg parsekey")
        return bloomberg_rows

    isin_rows = await stock_lookup_by_isin(args, context)
    if isin_rows:
        logger.info("found isin match")
        return isin_rows

    gbi_alt_name_rows = await stock_lookup_by_exact_gbi_alt_name(args, context)
    if gbi_alt_name_rows:
        logger.info("found gbi alt name match")
        return gbi_alt_name_rows

    return []


async def raw_stock_identifier_lookup(
    args: StockIdentifierLookupInput, context: PlanRunContext
) -> List[Dict[str, Any]]:
    """Returns the stocks with the closest text match to the input string
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

    exact_rows = await stock_lookup_exact(args, context)
    if exact_rows:
        logger.info(f"found  {len(exact_rows)} exact matches: {args=}")
        return exact_rows

    similar_rows = await stock_lookup_by_text_similarity(args, context)
    if similar_rows:
        logger.info(
            f"found {len(similar_rows)} text similarity matches: {args=}," f" {similar_rows[:4]}"
        )
        return similar_rows

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
    WHERE spiq_company_id = %s AND ms.is_public
    AND euh.to_z > NOW()
    """
    rows = db.generic_read(sql, [universe_spiq_company_id])

    return [
        StockID(
            gbi_id=row["gbi_id"], symbol=row["symbol"], isin=row["isin"], company_name=row["name"]
        )
        for row in rows
    ]


async def get_stock_ids_from_company_ids(
    context: PlanRunContext,
    spiq_company_ids: List[int],
    prefer_gbi_ids: Optional[List[int]] = None,
) -> Dict[int, StockID]:
    """Returns the list of stock identifiers given a stock universe's company id.

    Args:
        spiq_company_ids: List[int]
        context (PlanRunContext): The context of the plan run.

    Returns:
        Dict[int, StockID]: Mapping from company ID to output stock ID
    """
    db = get_psql()

    # Find the stocks in the universe
    sql = """
    SELECT
        ssm.gbi_id, ms.symbol, ms.isin, ms.name, ssm.spiq_company_id
    FROM master_security ms
    JOIN spiq_security_mapping ssm on ssm.gbi_id = ms.gbi_security_id
    WHERE ssm.spiq_company_id = ANY(%s)
        AND ms.is_public
    ORDER BY ssm.gbi_id, ms.from_z ASC
    """
    rows = db.generic_read(sql, [spiq_company_ids])

    cid_to_rows = defaultdict(list)
    for row in rows:
        cid_to_rows[row["spiq_company_id"]].append(row)

    cid_to_stock = {}
    prefer_gbi_ids_set = set(prefer_gbi_ids or [])
    for cid, rows in cid_to_rows.items():
        for row in rows:
            if row["gbi_id"] in prefer_gbi_ids_set:
                cid_to_stock[cid] = StockID(
                    gbi_id=row["gbi_id"],
                    symbol=row["symbol"],
                    isin=row["isin"],
                    company_name=row["name"],
                )
                break
        else:
            pick_row = rows[0]
            cid_to_stock[cid] = StockID(
                gbi_id=pick_row["gbi_id"],
                symbol=pick_row["symbol"],
                isin=pick_row["isin"],
                company_name=pick_row["name"],
            )

    return cid_to_stock


class GetRiskExposureForStocksInput(ToolArgs):
    stock_list: List[StockID]


@tool(
    description=(
        "This function takes a list of stock ids"
        " and returns a table of named factor exposure values for each stock."
        " The table has the following factors: value, leverage, growth, volatility, "
        "momentum, trading activity, size, and market. The values in the table are "
        "normalized, you can filter this table with the `transform_table' tool as follows: "
        "use > 1 to get 'high' scores, > 2 to get 'very high' scores "
        " < -1 to get 'low' scores, and -2 to get 'very low' scores. "
        "Use this tool if the user asks to filter/rank by one of these factors specifically, "
        "but you must never use it unless the client says corresponds exactly or almost "
        "exactly to one of the relevant factors, you must use one or more of the provided "
        "factor names exactly in your instructions to the transform_table table, if you cannot do that "
        "you should probably filter by profile instead."
        "Do not use this tool if the user asks for a particular statistic, even if it is"
        "closely related to one of these factors, use the get_statistic tool instead."
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
        " you must only use this function if the client specifically asks to filter by"
        " growth, do not use it for specific statistics, even if they are growth-related"
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
        " you must only use this function if the client specifically asks to filter by"
        " value, do not use it for specific statistics, even if they are value-related"
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
    strict_word_similarity(gsu.name, %s) AS ws,
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
    rows = await augment_stock_rows_with_volume(rows)
    orig_stocks_sorted_by_volume = sorted(
        rows, key=lambda x: (x.get("volume", 0), x.get("final_match_score", 0)), reverse=True
    )

    stock = orig_stocks_sorted_by_volume[0]

    return stock


async def stock_lookup_by_bloomberg_parsekey(
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

    iso3 = bloomberg_exchange_to_country_iso3.get(exch_code.upper())

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


class StockMarketSegmentFilterInput(ToolArgs):
    stock_ids: List[StockID]
    segment_text: str


STOCK_MARKET_SEGMENT_FILTER_MAIN_PROMPT = Prompt(
    name="STOCK_MARKET_SEGMENT_FILTER_MAIN_PROMPT",
    template=(
        "Your task to determine whether the following stock is a good match for the given market segment. "
        "You can use the stock description to help you make this determination. "
        "If you think the stock is a good match, return 'yes'. "
        "If you think the stock is not a good match, return 'no'. "
        "You answer MUST be one of the following: <yes, no>. "
        "\n###Stock Description\n"
        "{stock_description}"
        "\nMarket Segment\n"
        "{segment_text}"
        "\nNow is this stock a good match for the given market segment? (yes/no)"
    ),
)


@tool(
    description=(
        "This function takes a list of stock ids and a market segment text "
        "such as tech, health, gaming, ai chips, organic groceries, semiconductor, etc., "
        "and filters them acccording to how well they match the given market segment. "
        "Filtering is done based on the company/stock description text. "
    ),
    category=ToolCategory.STOCK,
    tool_registry=ToolRegistry,
    is_visible=True,
)
async def market_segment_filter(
    args: StockMarketSegmentFilterInput, context: PlanRunContext
) -> List[StockID]:
    # get company/stock descriptions
    description_texts = await get_company_descriptions(
        GetStockDescriptionInput(
            stock_ids=args.stock_ids,
        ),
        context,
    )

    # create aligned stock text groups and get all the text strings
    aligned_text_groups = StockAlignedTextGroups.from_stocks_and_text(
        args.stock_ids, description_texts  # type: ignore
    )
    stock_description_map: Dict[StockID, str] = await Text.get_all_strs(  # type: ignore
        aligned_text_groups.val, include_header=True, text_group_numbering=True
    )
    # filter out those with no data
    stocks = [stock for stock in args.stock_ids if stock in stock_description_map]

    # initiate GPT context and llm model
    llm = GPT(model=GPT35_TURBO)
    # create GPT call tasks
    tasks = []
    for stock in stocks:
        tasks.append(
            llm.do_chat_w_sys_prompt(
                STOCK_MARKET_SEGMENT_FILTER_MAIN_PROMPT.format(
                    stock_description=stock_description_map[stock],
                    segment_text=args.segment_text,
                ),
                sys_prompt=NO_PROMPT,
            )
        )
    results = await gather_with_concurrency(tasks, n=FILTER_CONCURRENCY)

    # filter out stocks that are not a good match
    filtered_stocks = []
    for stock, result in zip(stocks, results):
        if result.lower() == "yes":
            filtered_stocks.append(stock)

    if not filtered_stocks:
        raise ValueError(
            f"No stocks are a good match for the given market segment: '{args.segment_text}'"
        )

    return filtered_stocks


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
