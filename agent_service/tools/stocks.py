# Author(s): Mohammad Zarei, David Grohmann
import inspect
import json
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, cast

import pandas as pd
from gbi_common_py_utils.numpy_common import NumpySheet
from gbi_common_py_utils.utils.environment import PROD_TAG, get_environment_tag

from agent_service.external.pa_backtest_svc_client import (
    universe_stock_factor_exposures,
)
from agent_service.external.stock_search_dao import async_sort_stocks_by_volume
from agent_service.GPT.constants import GPT4_O_MINI, NO_PROMPT
from agent_service.GPT.requests import GPT
from agent_service.io_types.dates import DateRange
from agent_service.io_types.stock import StockID
from agent_service.io_types.table import (
    StockTable,
    Table,
    TableColumnMetadata,
    TableColumnType,
)
from agent_service.planner.errors import EmptyInputError, NotFoundError
from agent_service.tool import (
    TOOL_DEBUG_INFO,
    ToolArgs,
    ToolCategory,
    default_tool_registry,
    tool,
)
from agent_service.tools.lists import CombineListsInput, add_lists
from agent_service.tools.tool_log import tool_log
from agent_service.types import PlanRunContext
from agent_service.utils.async_db import get_async_db
from agent_service.utils.async_postgres_base import DEFAULT_ASYNCDB_GATHER_CONCURRENCY
from agent_service.utils.async_utils import gather_with_concurrency
from agent_service.utils.cache_utils import PostgresCacheBackend
from agent_service.utils.date_utils import get_now_utc
from agent_service.utils.gpt_logging import GptJobIdType, GptJobType, create_gpt_context
from agent_service.utils.logs import async_perf_logger
from agent_service.utils.pagerduty import pager_wrapper
from agent_service.utils.prefect import get_prefect_logger
from agent_service.utils.prompt_utils import Prompt
from agent_service.utils.stock_metadata import get_stock_metadata_rows
from agent_service.utils.string_utils import (
    clean_to_json_if_needed,
    repair_json_if_needed,
)
from agent_service.utils.tool_diff import (
    add_task_id_to_stocks_history,
    get_prev_run_info,
)

MAX_SYMBOL_LEN = 8

ACCEPTABLE_ASSET_CLASSES = """'Common Stock',
 'Depositary Receipt (Common Stock)',
 'Debt/Equity Composite Units'"""

STOCK_ID_COL_NAME_DEFAULT = "Security"
GROWTH_LABEL = "Growth"
VALUE_LABEL = "Value"

UNIVERSE_ADD_STOCK_DIFF = "{company} was added to the {universe} stock universe"
UNIVERSE_REMOVE_STOCK_DIFF = "{company} was removed from the {universe} stock universe"

ETF_ADD_STOCK_DIFF = "{company} was added to the universe of ETFs"
ETF_REMOVE_STOCK_DIFF = "{company} was removed from the universe of ETFs"

FACTOR_ADD_STOCK_DIFF = "{company} was added to the {factor} list"
FACTOR_REMOVE_STOCK_DIFF = "{company} was removed from the {factor} list"

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
    company_integer_id: Optional[int] = None
    stock_name: Optional[str] = None
    prefer_etfs: Optional[bool] = False


DEFAULT_CONFIRMATION_TARGET_TYPE = "company, stock, or ETF"
STOCK_CONFIRMATION_PROMPT_STR = """
Your task is to determine which {target_type} the search term refers to,
 the search term might be a common word,
 but in this case you should try really hard to associate it with a stock or etf
 If the search term does NOT refer to a company, stock, ETF,
 then return an empty json: '{{}}'
 If the search term DOES refer to a company, stock, ETF, then return
 a json with at least the full_name field.
 Optionally, it is not required but it would be helpful if you
 can also return optional fields: ticker_symbol, isin, match_type, and country_iso3,
 match_type is 'stock', 'etf'
 country_iso3 is the 3 letter iso country code where it is most often traded or originated from
 Only return each of the optional fields if you are confident they are correct,
 it is ok if you do not know some or all of them.
 you can return some and not others, the only required field is full_name.
 If you think the user is looking for a stock index, you should return the most popular ETF
 that is tracking that index, instead of the index itself
 for example instead of the SP500 index SPX you should return the most related ETF: SPY.
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
 Remember you are looking only for results of these types: {target_type}
 REMEMBER! If your reason is something like: "the search term closely resembles the ticker symbol"
 and the match is not an exact match for the ticker
 then you should reconsider as ticker typos are not common, and your answer is
 extremely likely to be wrong.
 In that case you should find a different answer with a better reason.
 You will be fired if your reason is close to: "the search term closely resembles the ticker symbol"
 When looking for ETFs, Your the search term must have some relevance to the ticker, name or match text
 besides just the word "ETF" itself.
"""


@async_perf_logger
async def stock_confirmation_by_gpt(
    context: PlanRunContext,
    search: str,
    results: List[Dict],
    target_type: str = DEFAULT_CONFIRMATION_TARGET_TYPE,
) -> Optional[Dict[str, Any]]:
    logger = get_prefect_logger(__name__)
    gpt_context = create_gpt_context(
        GptJobType.AGENT_TOOLS, context.agent_id, GptJobIdType.AGENT_ID
    )
    # we can try different models
    # HAIKU, SONNET, GPT4_TURBO,
    # should we switch this to 40mini?
    llm = GPT(context=gpt_context, model=GPT4_O_MINI)

    STOCK_CONFIRMATION_PROMPT = Prompt(
        name="STOCK_CONFIRMATION_PROMPT", template=STOCK_CONFIRMATION_PROMPT_STR
    )
    prompt = STOCK_CONFIRMATION_PROMPT.format(
        search=search, target_type=target_type, results=json.dumps(results)
    )

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


def get_stock_identifier_lookup_cache_key(
    tool_name: str, args: StockIdentifierLookupInput, context: PlanRunContext
) -> str:
    key = "_".join(
        [
            # TODO add something to indicate "SOFTWARE_VERSION"
            # as the cached values may not match the return vals for a newer implementation
            "TOOL",
            tool_name,
            "ARGS",
            args.stock_name.lower() if args.stock_name else str(args.company_integer_id),
            str(args.prefer_etfs),
        ]
    )
    return key


def could_be_ticker(s: str) -> bool:
    # check if 's' looks like it MIGHT be a ticker
    if len(s) > MAX_SYMBOL_LEN:
        return False
    if len(s.split()) > 1:
        return False
    return True


@tool(
    description=(
        "This function takes a string (microsoft, apple, AAPL, TESLA, META, SP 500, e.g.) "
        "which refers to a stock or ETF, and converts it to an identifier object. "
        "If the user has mentioned a stock with a Company Integer ID defined, "
        "pass that in instead. DO NOT PASS BOTH. One of stock_name or company_integer_id "
        "absolutely MUST be defined. If using the string lookup, pass as stock_name. "
        "If using the company integer ID lookup, pass as company_integer_id."
        "This tool MUST still be used even if the int ID is provided it may NOT be"
        " passed as a literal to other tools."
        " Other than simple spelling fixes you should try to take the stock_name field"
        " input directly from the original input text."
        " You MUST let this function interpret the meaning of the input text,"
        " the stock_name passed to this function should"
        " usually be copied verbatim from the client input, avoid"
        " paraphrasing or copying from a sample plan."
        " If the original input text contains a 2, 3, or 4 character stock ticker symbol,"
        " do not attempt to spell correct, modify, or alter it,"
        " do not replace it with a company name either,"
        " instead pass it directly as the stock_name and allow this function to interpret it."
        " If the user intent is to find an ETF, you may set prefer_etfs = True"
        " to enable some etf specific matching logic."
        " If the user specifies both a company name and a stock ticker"
        " like 'Name (TICKER)' or 'TICKER Name'"
        " then you should pass both in as having the name and ticker can help disambiguate."
    ),
    category=ToolCategory.STOCK,
    use_cache=True,
    cache_key_fn=get_stock_identifier_lookup_cache_key,
    cache_ttl=60 * 60 * 24,
    cache_backend=PostgresCacheBackend(),
    tool_registry=default_tool_registry(),
    is_visible=False,
)
@async_perf_logger
async def stock_identifier_lookup(
    args: StockIdentifierLookupInput, context: PlanRunContext
) -> StockID:
    """
    Returns the stock ID obj of a stock given its name or symbol (microsoft, apple, AAPL, TESLA, META, e.g.).

    This function performs a series of queries to find the stock's identifier. It starts with an exact symbol match,
    followed by a word similarity name match, and finally a word similarity symbol match. It only
    proceeds to the next query if the previous one returns no results.


    Args:
        args (StockIdentifierLookupInput): The input arguments for the stock lookup.
        context (PlanRunContext): The context of the plan run.

    Returns:
        int: The integer identifier of the stock.
    """
    return await stock_identifier_lookup_helper(args, context)


async def stock_identifier_lookup_helper(
    args: StockIdentifierLookupInput, context: PlanRunContext
) -> StockID:
    logger = get_prefect_logger(__name__)
    if not args.stock_name and not args.company_integer_id:
        raise ValueError("Must pass either a stock name or company integer id!!")

    # Tool level debug info
    real_debug_info: Dict[str, Any] = {}
    real_debug_info = TOOL_DEBUG_INFO.get()

    debug_info: Dict[str, Any] = {}
    # create one debug entry per unique call to this tool
    # useful for multistock lookup and other tools that invoke this tool
    if args.stock_name is not None:
        real_debug_info[args.stock_name] = debug_info

    try:  # since everything associated with diffing/rerun cache is optional, put in try/except
        # Update mode
        logger.info("Checking previous run info...")
        prev_run_info = await get_prev_run_info(context, "stock_identifier_lookup")
        if prev_run_info is not None:
            prev_args = StockIdentifierLookupInput.model_validate_json(prev_run_info.inputs_str)
            if args.stock_name == prev_args.stock_name:
                prev_output: StockID = prev_run_info.output  # type:ignore
                logger.info(f"using {prev_output=}")
                await tool_log(
                    log=f"Interpreting '{args.stock_name}' as {prev_output.symbol}: {prev_output.company_name}",
                    context=context,
                )

                return prev_output

    except Exception as e:
        logger.exception(f"Error using previous run cache: {e}")
        pager_wrapper(
            current_frame=inspect.currentframe(),
            module_name=__name__,
            context=context,
            e=e,
            classt="AgentUpdateError",
            summary="Failed to get previous run info",
        )

    logger.info(f"Attempting to map '{args.stock_name}' to a stock")

    # If a GBI ID has been entered, use that as long as it's valid
    if args.company_integer_id:
        gbi_id_row = await stock_lookup_by_gbi_id(gbi_id=args.company_integer_id)
        if gbi_id_row:
            logger.info(f"Found exact DB match for {args.company_integer_id=}")
            return StockID(
                gbi_id=gbi_id_row["gbi_security_id"],
                symbol=gbi_id_row["symbol"],
                isin=gbi_id_row["isin"],
                company_name=gbi_id_row["company_name"],
            )

    # We only get here if the GBI ID is invalid or not included
    if not args.stock_name:
        raise ValueError(f"Could not find stock with ID={args.company_integer_id}")
    # first we check if the search string is in a format that leads to an unambiguous match
    exact_rows = await stock_lookup_exact(args, context)
    if exact_rows:
        logger.info(f"found {len(exact_rows)} exact matches")
        if len(exact_rows) > 1:
            exact_rows = await augment_stock_rows_with_volume(context.user_id, exact_rows)
            exact_rows = sorted(exact_rows, key=lambda x: x.get("volume", 0), reverse=True)
        stock = exact_rows[0]
        logger.info(f"found exact match {stock=}")
        return StockID(
            gbi_id=stock["gbi_security_id"],
            symbol=stock["symbol"],
            isin=stock["isin"],
            company_name=stock["name"],
        )

    # if this is an ISIN, we should not allow partial matches on non-ISIN company text
    if is_isin(args.stock_name):
        logger.info(f"No acceptable stock found with isin match: {args.stock_name}")
        raise NotFoundError(f"Could not find any stocks related to: '{args.stock_name}'")

    # next we check for best matches by text similarity
    rows = await stock_lookup_by_text_similarity(args, context)
    logger.info(f"found {len(rows)} best potential matching stocks")
    rows = await augment_stock_rows_with_volume(context.user_id, rows)

    orig_stocks_sorted_by_match = sorted(
        rows, key=lambda x: (x["final_match_score"], x.get("volume", 0)), reverse=True
    )
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

    debug_info["matches_for_confirmation"] = ask_gpt

    target_type = DEFAULT_CONFIRMATION_TARGET_TYPE
    if args.prefer_etfs:
        target_type = "ETF"
    gpt_answer = await stock_confirmation_by_gpt(
        context, search=args.stock_name, results=list(ask_gpt.values()), target_type=target_type
    )

    debug_info["stock_confirmation_gpt_answer"] = gpt_answer

    if gpt_answer and gpt_answer.get("full_name"):
        # map the GPT answer back to a gbi_id

        gpt_answer_full_name = cast(str, gpt_answer["full_name"])

        gpt_stock: Optional[Dict[str, Any]] = None
        ticker_match: Optional[Dict[str, Any]] = None
        # first check if gpt answer is in the original set:
        for s in orig_stocks_sorted_by_volume:
            s_name = s.get("name", "")
            s_name = s_name.lower() if s_name else ""

            s_symbol = s.get("symbol", "")
            s_symbol = s_symbol.lower() if s_symbol else ""

            if gpt_answer_full_name.lower() == s_name:
                gpt_stock = s
                logger.info(f"found gpt answer in original set: {gpt_stock=}, {gpt_answer=}")
                break
            if not ticker_match and args.stock_name and args.stock_name.lower() == s_symbol:
                # we will need this for a corner case where GPT likes to lock on to ticker typos
                ticker_match = s
                logger.info(
                    f"found potential ticker match: {args.stock_name=}, {s=}, {gpt_answer=}"
                )

        if gpt_stock:
            # we found the same name stock in original result set as GPT
            confirm_rows = [gpt_stock]
        else:
            # we need to remap the gpt answer back to a gbiid
            reason = gpt_answer.get("reason", "")
            if (
                ticker_match
                and ("ticker" in reason or "symbol" in reason)
                and ("closely matches" in reason or "typo" in reason or "misspell" in reason)
            ):
                logger.info(
                    f"using ticker match instead of typo reasoning: {args.stock_name=}, {ticker_match=}, {gpt_answer=}"
                )
                # GPT has been told not to use ticker typos as it's reason but it doesnt always listen
                # so if we could not find the company name in the original search list, and GPT thinks it
                # is a ticker typo and the original search text matches the ticker of one of the original
                # matches, we will assume that is the correct stock.
                confirm_rows = [ticker_match]
            else:
                # GPT either used an alternative name for one of the stocks we found
                # or it is deciding that our suggested answers are all wrong, and it came up with
                # its own likely company match
                # so now we do a full text search to map the gpt answer back to a gbi_id
                confirm_args = StockIdentifierLookupInput(stock_name=gpt_answer_full_name)
                confirm_rows = await stock_lookup_by_text_similarity(
                    confirm_args, context, min_match_strength=MIN_GPT_NAME_MATCH
                )
                logger.info(f"found {len(confirm_rows)} best matching stock to the gpt answer")
                if len(confirm_rows) > 1:
                    confirm_rows = await augment_stock_rows_with_volume(
                        context.user_id, confirm_rows
                    )

                logger.info(f"confirmed rows found by gpt: {confirm_rows}")

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
    else:
        fall_back_stock = None
        if ask_gpt:
            for gbi_id, stock in ask_gpt.items():
                # these magic numbers were arbitrarily picked
                if stock.get("final_match_score", 0) >= 0.7 and stock.get("volume", 0) >= 10000:
                    fall_back_stock = stock
                    break

        if fall_back_stock:
            stock = fall_back_stock
            logger.info(
                "GPT answer could not be mapped back to a gbiid,"
                f"falling back to best original match: {stock} for '{args.stock_name}'"
            )
        else:
            raise NotFoundError(f"Could not find any stocks related to: '{args.stock_name}'")

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

    # Exact gbi alt name match
    # these are hand crafted strings to be used only when needed
    sql = f"""
    SELECT gbi_security_id, ms.symbol, ms.isin, ms.security_region, ms.currency,
    ms.name, gan.alt_name as gan_alt_name
    FROM master_security ms
    JOIN "data".gbi_id_alt_names gan ON gan.gbi_id = ms.gbi_security_id
    WHERE
    upper(gan.alt_name) = upper(%(search_term)s)
    AND gan.enabled
    AND ms.is_public
    AND ms.asset_type in ( {ACCEPTABLE_ASSET_CLASSES} )
    AND ms.is_primary_trading_item = true
    AND ms.to_z is null
    AND source_id = 0 -- boosted custom alt_name entries
    """
    db = get_async_db(read_only=True)
    rows = await db.generic_read(sql, {"search_term": args.stock_name})
    if rows:
        logger = get_prefect_logger(__name__)
        logger.info("found exact gbi alt name")
        return rows

    return []


def is_isin(search: str) -> bool:
    if 12 == len(search) and search[0:2].isalpha() and search[2:].isalnum():
        return True

    return False


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
    if not args.stock_name:
        raise ValueError("Cannot look up by ISIN if no stock name present")

    # ISINs are 12 chars long, 2 chars, 10 digits
    if is_isin(args.stock_name):
        # Exact ISIN match
        sql = f"""
        SELECT gbi_security_id, ms.symbol, ms.isin, ms.security_region, ms.currency, name,
        'ms.isin' as match_col, ms.isin as match_text
        FROM master_security ms
        WHERE ms.isin = upper(%(search_term)s)
        AND ms.is_public
        AND ms.asset_type  in ( {ACCEPTABLE_ASSET_CLASSES} )
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
        AND ms.asset_type  in ( {ACCEPTABLE_ASSET_CLASSES} )
        AND ms.is_primary_trading_item = true
        AND ms.to_z is null
        """
        db = get_async_db(read_only=True)
        rows = await db.generic_read(sql, {"search_term": args.stock_name})
        if rows:
            # useful for debugging
            # print("isin match: ", rows)
            logger = get_prefect_logger(__name__)
            logger.info("found by ISIN")
            return rows

    return []


async def augment_stock_rows_with_volume(
    user_id: str, rows: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Returns the input row dicts augmented with a new 'volume' field

    Returns:
        List[Dict[str, Any]]: DB rows representing the potentially matching stocks.
    """
    logger = get_prefect_logger(__name__)

    # dedupe by gbi_id and keep the first item (already presorted by match strength)
    gbiid2stocks = {}
    for r in rows:
        if r["gbi_security_id"] not in gbiid2stocks:
            gbiid2stocks[r["gbi_security_id"]] = r

    gbi_ids = list(gbiid2stocks.keys())
    stocks_sorted_by_volume = await async_sort_stocks_by_volume(user_id, gbi_ids)

    if stocks_sorted_by_volume:
        if len(stocks_sorted_by_volume) != len(gbiid2stocks):
            logger.warning(
                f"Some stock volumes not found, expected: {len(gbiid2stocks)}"
                f" but got: {len(stocks_sorted_by_volume)}"
            )

        logger.info(f"Top stock volumes: {stocks_sorted_by_volume[:10]}")
        for gbi_id, volume in stocks_sorted_by_volume:
            stock = gbiid2stocks.get(gbi_id)
            if stock:
                stock["volume"] = volume
            else:
                logger.warning("Logic error!")
                # should not be possible
    else:
        logger.warning(f"No stock volumes found for {gbi_ids=}")

    new_rows = list(gbiid2stocks.values())
    return new_rows


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
    db = get_async_db(read_only=True)
    if not args.stock_name:
        raise ValueError("Cannot look up by text similarity if no stock name present")

    prefix = args.stock_name.split()[0]
    # often the most important word in a company name is the first word,
    # Samsung electronics, co. ltd, Samsung is important everything after is not
    logger.info(f"checking for text similarity of {args.stock_name=} and {prefix=}")

    # normally we wont include this logic in the giant union below
    etf_union_sql = ""

    etf_match_str = args.stock_name
    if args.prefer_etfs or "etf" in args.stock_name.lower().split():
        # GNR ETF
        etf_match_str = etf_match_str.upper().replace(" ETF", "").replace("ETF ", "").strip()
        etf_union_sql = f"""
        UNION

        -- ETF ticker symbol (exact match only)
        SELECT gbi_security_id, ms.symbol, ms.isin, ms.security_region, ms.currency,
        ms.asset_type,
        ms.security_type,
        ms.name, 'ticker symbol' as match_col, ms.symbol as match_text,
        {PERFECT_TEXT_MATCH} AS text_sim_score
        FROM master_security ms
        WHERE
        ms.asset_type in ( {ACCEPTABLE_ASSET_CLASSES} )
        AND ms.is_public
        AND ms.is_primary_trading_item = true
        AND ms.to_z is null
        AND ms.symbol = upper(%(etf_match_str)s)
        -- double pct sign for python x sql special chars
        AND (ms.security_type like '%%ETF%%'  OR ms.security_type like '%%Fund%%')
        """

    ticker_and_name_match_sql = ""

    possible_ticker_str = args.stock_name.upper()

    # some tickers have a '.' in them
    possible_ticker_str = re.sub("[^A-Za-z0-9.]+", " ", possible_ticker_str)

    possible_ticker_tokens = possible_ticker_str.split()
    if len(possible_ticker_tokens) > 1:
        # sometimes users give us both the name and ticker often like "Name (Ticker)"
        # this will specifically search for that combination
        ticker_and_name_match_sql = f"""
        UNION

        -- stock ticker (exact) + name match (partial)
        SELECT gbi_security_id, ms.symbol, ms.isin, ms.security_region, ms.currency,
        ms.asset_type,
        ms.security_type,
        ms.name, 'ticker symbol & name' as match_col,
        ms.name || ' (' || ms.symbol || ')' as match_text,
        (strict_word_similarity(ms.name || ' ' || ms.symbol, %(search_term)s) +
        strict_word_similarity(%(search_term)s, ms.name || ' ' || ms.symbol)) / 2
        + 0.1 AS text_sim_score
        -- 0.1 is a magic number to boost this match since it covers both fields
        FROM master_security ms
        WHERE
        ms.asset_type in ( {ACCEPTABLE_ASSET_CLASSES} )
        AND ms.is_public
        AND ms.is_primary_trading_item = true
        AND ms.to_z is null
        AND ms.symbol = ANY(%(possible_ticker_tokens)s)

        -- there needs to be a not-terrible match against the company name by itself also
        -- 0.3 is a magic number to make sure that at least some of the search term matches
        -- a decent fraction of the company name
        AND  (strict_word_similarity(ms.name || ' ' || ms.symbol, %(search_term)s) +
        strict_word_similarity(%(search_term)s, ms.name || ' ' || ms.symbol)) / 2  > 0.3
        """

    ipo_dot_u_union_sql = ""
    dot_u_tickers = []
    if could_be_ticker(possible_ticker_str) and "." not in possible_ticker_str:
        dot_u_tickers.append(possible_ticker_str + ".U")
        if possible_ticker_str.endswith("U"):
            dot_u_tickers.append(possible_ticker_str[:-1] + ".U")
        ipo_dot_u_union_sql = f"""
        UNION

    -- ticker symbol (exact match only)
    SELECT gbi_security_id, ms.symbol, ms.isin, ms.security_region, ms.currency,
    ms.asset_type,
    ms.security_type,
    ms.name, 'ticker symbol dot u' as match_col, ms.symbol as match_text,
    -- this is an odd corner case for misquoted IPO table
    0.9 AS text_sim_score
    FROM master_security ms
    WHERE
    ms.asset_type in ( {ACCEPTABLE_ASSET_CLASSES} )
    AND ms.is_public
    AND ms.is_primary_trading_item = true
    AND ms.to_z is null
    AND ms.symbol = ANY(%(dot_u_tickers)s)"""

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
    ms.asset_type,
    ms.security_type,
    ms.name, 'ticker symbol' as match_col, ms.symbol as match_text,
    {PERFECT_TEXT_MATCH} AS text_sim_score
    FROM master_security ms
    WHERE
    ms.asset_type in ( {ACCEPTABLE_ASSET_CLASSES} )
    AND ms.is_public
    AND ms.is_primary_trading_item = true
    AND ms.to_z is null
    AND ms.symbol = upper(%(search_term)s)

{etf_union_sql}

{ticker_and_name_match_sql}

{ipo_dot_u_union_sql}

    UNION

(
    -- company name
    SELECT gbi_security_id, symbol, ms.isin, ms.security_region, ms.currency,
    ms.asset_type,
    ms.security_type,
    name, 'name' as match_col, ms.name as match_text,
    (strict_word_similarity(ms.name, %(search_term)s) +
    strict_word_similarity(%(search_term)s, ms.name)) / 2
    AS text_sim_score
    FROM master_security ms
    WHERE
    ms.asset_type  in ( {ACCEPTABLE_ASSET_CLASSES} )
    AND ms.is_public
    AND ms.is_primary_trading_item = true
    AND ms.to_z is null

    -- this uses the trigram index which speeds up the qry
    -- https://www.postgresql.org/docs/current/pgtrgm.html#PGTRGM-OP-TABLE
    AND name %% %(search_term)s
    ORDER BY text_sim_score DESC
    LIMIT 100
)

    UNION

(
    -- custom boosted db entries -  company alt name * 1.0
    SELECT gbi_security_id, symbol, ms.isin, ms.security_region, ms.currency,
    ms.asset_type,
    ms.security_type,
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
    ms.asset_type  in ( {ACCEPTABLE_ASSET_CLASSES} )
    AND can.enabled
    AND ms.is_public
    AND ms.is_primary_trading_item = true
    AND ms.to_z is null
    AND alt_name %% %(search_term)s
    AND alt_name not ilike 'ETF' --SPIQ spammed every ETF with this!
    ORDER BY text_sim_score DESC
    LIMIT 100
)

    UNION

(
    -- gbi alt name
    SELECT gbi_security_id, symbol, ms.isin, ms.security_region, ms.currency,
    ms.asset_type,
    ms.security_type,
    name, 'gbi alt name' as match_col, alt_name as match_text,
    (strict_word_similarity(alt_name, %(search_term)s) +
    strict_word_similarity(%(search_term)s, alt_name)) / 2
    AS text_sim_score
    FROM master_security ms
    JOIN "data".gbi_id_alt_names gan ON gan.gbi_id = ms.gbi_security_id
    WHERE
    ms.asset_type  in ( {ACCEPTABLE_ASSET_CLASSES} )
    AND gan.enabled
    AND ms.is_public
    AND ms.is_primary_trading_item = true
    AND ms.to_z is null
    AND alt_name %% %(search_term)s
    AND %(search_term)s %% alt_name
    ORDER BY text_sim_score DESC
    LIMIT 100
)

    ORDER BY text_sim_score DESC
    LIMIT 200
    ) as text_scores
    ) as final_scores -- word similarity score
    WHERE
    final_scores.final_match_score >= {min_match_strength}  -- score including prefix match
    ORDER BY final_match_score DESC
    LIMIT 100
    """
    rows = await db.generic_read(
        sql,
        params={
            "search_term": args.stock_name,
            "prefix": prefix,
            "etf_match_str": etf_match_str,
            "possible_ticker_tokens": possible_ticker_tokens,
            "dot_u_tickers": dot_u_tickers,
        },
    )

    # this shouldn't matter but at least once on dev the volume lookup silently failed
    # this will break ties in favor of Common stock and against Depositary Receipts
    for r in rows:
        if "Common Stock" != r["asset_type"]:
            r["final_match_score"] *= 0.9

    for r in rows:
        sec_type = str(r.get("security_type")).lower()
        if "etf" in sec_type or "fund" in sec_type:
            r["is_etf"] = True
            r["is_company"] = False
        else:
            r["is_etf"] = False
            r["is_company"] = True

    rows.sort(key=lambda x: x["final_match_score"], reverse=True)

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


async def stock_lookup_by_gbi_id(gbi_id: int) -> Optional[Dict[str, Any]]:
    db = get_async_db(read_only=True)
    sql = """
    SELECT gbi_security_id, ms.symbol, ms.isin, ms.security_region, ms.currency,
    ms.name AS company_name
    FROM master_security ms
    WHERE gbi_security_id = %(gbi_id)s
    """
    rows = await db.generic_read(sql, {"gbi_id": gbi_id})
    if not rows:
        return None
    return rows[0]


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

    if not args.stock_name:
        raise ValueError("Cannot run stock lookup exact without stock name")
    bloomberg_rows = await stock_lookup_by_bloomberg_parsekey(args, context)
    if bloomberg_rows:
        logger.info("found bloomberg parsekey")
        return bloomberg_rows

    ric_yahoo_rows = await stock_lookup_by_ric_yahoo_codes(args, context)
    if ric_yahoo_rows:
        logger.info("found RIC/Yahoo code")
        return ric_yahoo_rows

    if is_isin(args.stock_name):
        isin_rows = await stock_lookup_by_isin(args, context)
        if isin_rows:
            logger.info("found isin match")
            return isin_rows
        else:
            logger.info(f"No acceptable stock found with isin match: {args.stock_name}")
            return []

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
    if not args.stock_name:
        raise ValueError("Cannot look up by raw if no stock name present")
    logger = get_prefect_logger(__name__)

    exact_rows = await stock_lookup_exact(args, context)
    if exact_rows:
        logger.info(f"found  {len(exact_rows)} exact matches: {args=}")
        return exact_rows
    elif not is_isin(args.stock_name):
        similar_rows = await stock_lookup_by_text_similarity(args, context)
        if similar_rows:
            logger.info(
                f"found {len(similar_rows)} text similarity matches: {args=},"
                f" {similar_rows[:4]}"
            )
            return similar_rows

    raise NotFoundError(f"Could not find any stocks related to: '{args.stock_name}'")


class MultiStockIdentifierLookupInput(ToolArgs):
    company_integer_ids: List[int] = []
    # name or symbol of the stock to lookup
    stock_names: List[str] = []


@tool(
    description=(
        "This function takes a list of strings e.g. ['microsoft', 'apple', 'TESLA', 'META'] "
        "which refer to stocks, and converts them to a list of identifier objects. "
        "It may also take a list of Company Integer ID's, if present for some or all of the stocks."
        " ONLY pass in integer ID's when the Company Integer ID is EXPLICITLY given by the user."
        " If the Company Integer ID is given, ONLY USE THAT. DO NOT USE THE NAME ALSO."
        " For example, if the user inputs 'Microsoft (Company Integer ID: 123), ONLY 123 should be used."
        " Each stock should be present once, EITHER as a name OR an integer ID."
        " For stocks to be looked up by name, make sure you use the stock_names list."
        " For stocks to be looked up by integer ID, make sure you use the company_integer_ids list."
        " Make sure there is at least one value in either of these lists."
        "This tool MUST still be used even if int ID's are provided, they may NOT be"
        " passed as literals to other tools."
        "Since most other tools take lists of stocks, you should generally use this function "
        "to look up stocks mentioned by the client (instead of stock_identifier_lookup), "
        "even when there is only one stock."
        "However, you must NEVER use this tool when you need to use the resulting identifiers in separate "
        "tool calls, (e.g., you are creating multiple separate graphs or summmaries, one for each of the stocks)"
        "since you are NOT allowed to access stock ids in the resulting list using indexing, "
        "i.e. stock_ids[0] is NOT allowed, in those circumstances you should make separate calls to "
        "the stock_indentifier_lookup, for each stock, do not use this tool."
        " You MUST let this function interpret the meaning of the input text,"
        " the stock_names passed to this function should"
        " usually be copied verbatim from the client input, avoid"
        " paraphrasing or copying from a sample plan."
        " If the original input text contains a 2, 3, or 4 character stock ticker symbol,"
        " do not attempt to spell correct, modify, or alter it,"
        " do not replace it with a company name either,"
        " instead pass them directly as the stock_names and allow this function to interpret each of them."
        " If the user specifies both a company name and a stock ticker next to each other"
        " like 'Name (TICKER)' or 'TICKER Name'"
        " then you should pass both in as having the name and ticker can help disambiguate."
    ),
    category=ToolCategory.STOCK,
    tool_registry=default_tool_registry(),
    is_visible=False,
)
async def multi_stock_identifier_lookup(
    args: MultiStockIdentifierLookupInput, context: PlanRunContext
) -> List[StockID]:
    logger = get_prefect_logger(__name__)

    # Just runs stock identifier look up below for each stock in the list
    # Probably can be done more efficiently

    try:  # since everything associated with diffing/rerun cache is optional, put in try/except\
        # Update mode
        prev_run_info = await get_prev_run_info(context, "multi_stock_identifier_lookup")
        if prev_run_info is not None:
            prev_args = MultiStockIdentifierLookupInput.model_validate_json(
                prev_run_info.inputs_str
            )
            if sorted(args.stock_names) == sorted(prev_args.stock_names):
                prev_output: List[StockID] = prev_run_info.output  # type:ignore
                return prev_output

    except Exception as e:
        logger.exception(f"Error using previous run cache: {e}")
        pager_wrapper(
            current_frame=inspect.currentframe(),
            module_name=__name__,
            context=context,
            e=e,
            classt="AgentUpdateError",
            summary="Failed to get previous run info",
        )

    tasks = [
        stock_identifier_lookup_helper(
            (StockIdentifierLookupInput(stock_name=stock_name)),
            context,
        )
        for stock_name in args.stock_names
    ] + [
        stock_identifier_lookup_helper(
            StockIdentifierLookupInput(company_integer_id=gbi_id), context
        )
        for gbi_id in args.company_integer_ids
    ]

    # each of these tasks uses at least 1 sql,
    # but only max_pool sqls can run concurrently
    output: List[StockID] = await gather_with_concurrency(
        tasks,
        n=DEFAULT_ASYNCDB_GATHER_CONCURRENCY,
    )
    return list(set(output))


class GetETFUniverseInput(ToolArgs):
    pass


@tool(
    description=(
        "This function takes no arguments."
        " It returns the full list of StockIds for the entire universe of supported ETFs"
        " You should call this function to get a list of ETFs to be filtered or sorted later."
        " If the client wants to filter over ETFs but does not specify an initial set"
        " of stocks, you should call this tool first."
    ),
    category=ToolCategory.STOCK,
    tool_registry=default_tool_registry(),
    is_visible=True,
)
async def get_etf_list(args: GetETFUniverseInput, context: PlanRunContext) -> List[StockID]:
    """Returns the full list of ETFs.

    Args:
        args (GetETFUniverseInput): The input arguments for the ETF universe lookup.
        context (PlanRunContext): The context of the plan run.

    Returns:
        list[StockID]: The list of ETF identifiers in the universe.
    """
    logger = get_prefect_logger(__name__)

    db = get_async_db(read_only=True)

    sql = """
    SELECT gbi_id, spiq_company_id, name
    FROM "data".etf_universes
    """

    # TODO: cache this
    etf_rows = await db.generic_read(sql)
    gbi_ids = [r["gbi_id"] for r in etf_rows]
    stock_list = await StockID.from_gbi_id_list(gbi_ids)

    log_str = f"found {len(stock_list)} ETFs"
    logger.info(log_str)
    await tool_log(
        log=log_str,
        context=context,
    )

    return stock_list


class GetCountryInput(ToolArgs):
    stock_ids: List[StockID]


@tool(
    description=(
        "This function takes a list of StockIds"
        " and returns a table with columns named: 'security' and 'country'."
        " The 'security' column contains the original input stock_ids,"
        " The country column contains 3-character ISO country codes"
        " for each of the input stocks."
        " Use this function to add country to a dataset before display"
        " or for further processing."
        " If you want to filter on a country or region do not use this function,"
        " you should use the filter_stocks_by_region instead"
    ),
    category=ToolCategory.STOCK_GROUPS,
    tool_registry=default_tool_registry(),
    is_visible=True,
)
async def get_country_for_stocks(args: GetCountryInput, context: PlanRunContext) -> StockTable:
    """Returns the country for each stock

    Returns:
        StockTable: a table of stock identifiers and countries.
    """
    df = await get_metadata_for_stocks(args.stock_ids, context)

    # we might not have found metadata for all the gbi_ids (rare)
    # so lets select the ones we found and keep them for history
    orig_stock_ids = {s.gbi_id: s for s in args.stock_ids}
    new_stock_ids = [orig_stock_ids.get(id) for id in list(df["gbi_id"])]
    df["Security"] = new_stock_ids

    table = StockTable.from_df_and_cols(
        data=df,
        columns=[
            TableColumnMetadata(label="Security", col_type=TableColumnType.STOCK),
            TableColumnMetadata(label="Country", col_type=TableColumnType.STRING),
        ],
    )

    return table


@tool(
    description=(
        "This function takes a list of StockIds"
        " and returns a table with columns named: 'Security' and 'Country of Domicile'."
        " The 'security' column contains the original input stock_ids,"
        " The country of domicile column contains the name of the country"
        " for each of the input stocks."
        " Use this function to add country of domicile to a dataset before display"
        " or for further processing."
        " If you want to filter on a country or region or country of domicile do not use this function,"
        " you should use the filter_stocks_by_country_of_domicile instead"
    ),
    category=ToolCategory.STOCK_GROUPS,
    tool_registry=default_tool_registry(),
    is_visible=True,
)
async def get_country_of_domicile_for_stocks(
    args: GetCountryInput, context: PlanRunContext
) -> StockTable:
    """Returns the country for each stock

    Returns:
        StockTable: a table of stock identifiers and countries.
    """
    df = await get_metadata_for_stocks(args.stock_ids, context)

    # we might not have found metadata for all the gbi_ids (rare)
    # so lets select the ones we found and keep them for history
    orig_stock_ids = {s.gbi_id: s for s in args.stock_ids}
    new_stock_ids = [orig_stock_ids.get(id) for id in list(df["gbi_id"])]
    df["Security"] = new_stock_ids

    table = StockTable.from_df_and_cols(
        data=df,
        columns=[
            TableColumnMetadata(label="Security", col_type=TableColumnType.STOCK),
            TableColumnMetadata(label="Country of Domicile", col_type=TableColumnType.STRING),
        ],
    )

    return table


class GetCurrencyInput(ToolArgs):
    stock_ids: List[StockID]


@tool(
    description=(
        "This function takes a list of StockIds"
        " and returns a table with columns named: 'Security' and 'currency'."
        " The 'Security' column contains the original input stock_ids,"
        " The currency column contains 3-character ISO currency codes"
        " for each of the input stocks."
        " Use this function to add currency to a dataset before display"
        " or for further processing"
    ),
    category=ToolCategory.STOCK_GROUPS,
    tool_registry=default_tool_registry(),
    is_visible=True,
)
async def get_currency_for_stocks(args: GetCurrencyInput, context: PlanRunContext) -> StockTable:
    """Returns the currency for each stock

    Returns:
        StockTable: a table of stock identifiers and currencies.
    """

    df = await get_metadata_for_stocks(args.stock_ids, context)
    new_stock_ids = await StockID.from_gbi_id_list(list(df["gbi_id"]))
    df["Security"] = new_stock_ids

    table = StockTable.from_df_and_cols(
        data=df,
        columns=[
            TableColumnMetadata(label="Security", col_type=TableColumnType.STOCK),
            TableColumnMetadata(label="Currency", col_type=TableColumnType.STRING),
        ],
    )

    return table


class GetISINInput(ToolArgs):
    stock_ids: List[StockID]


@tool(
    description=(
        "This function takes a list of StockIds"
        " and returns a table with columns named: 'Security' and 'ISIN'."
        " The 'Security' column contains the original input stock_ids,"
        " The ISIN column contains the 12-character alphanumeric ISIN code"
        " for each of the input stocks."
        " Use this function to add ISIN to a dataset before display"
        " or for further processing"
    ),
    category=ToolCategory.STOCK,
    tool_registry=default_tool_registry(),
    is_visible=True,
)
async def get_ISIN_for_stocks(args: GetISINInput, context: PlanRunContext) -> StockTable:
    """Returns the ISIN for each stock

    Returns:
        StockTable: a table of stock identifiers and isin.
    """

    df = await get_metadata_for_stocks(args.stock_ids, context)
    new_stock_ids = await StockID.from_gbi_id_list(list(df["gbi_id"]))
    df["Security"] = new_stock_ids

    table = StockTable.from_df_and_cols(
        data=df,
        columns=[
            TableColumnMetadata(label="Security", col_type=TableColumnType.STOCK),
            TableColumnMetadata(label="ISIN", col_type=TableColumnType.STRING),
        ],
    )

    return table


class GetSectorInput(ToolArgs):
    stock_ids: List[StockID]


@tool(
    description=(
        "This function takes a list of StockIds"
        " and returns a table with columns named: 'Security' and 'Sector'."
        " The'Security' column contains the original input stock_ids,"
        " The Sector column contains the name of the GICS level 1 Sector"
        " that the stock belongs to."
        " Use this function to add Sector to a dataset before display"
        " or for further processing."
        " If you want to filter on a sector do not use this function,"
        " you should use the sector_filter instead"
    ),
    category=ToolCategory.STOCK_GROUPS,
    tool_registry=default_tool_registry(),
    is_visible=True,
)
async def get_sector_for_stocks(args: GetSectorInput, context: PlanRunContext) -> StockTable:
    """Returns the Sector for each stock

    Returns:
        StockTable: a table of stock identifiers and sectors.
    """

    df = await get_metadata_for_stocks(args.stock_ids, context)
    new_stock_ids = await StockID.from_gbi_id_list(list(df["gbi_id"]))
    df["Security"] = new_stock_ids

    table = StockTable.from_df_and_cols(
        data=df,
        columns=[
            TableColumnMetadata(label="Security", col_type=TableColumnType.STOCK),
            TableColumnMetadata(label="Sector", col_type=TableColumnType.STRING),
        ],
    )

    return table


class GetIndustryGroupInput(ToolArgs):
    stock_ids: List[StockID]


@tool(
    description=(
        "This function takes a list of StockIds"
        " and returns a table with columns named: 'Security' and 'Industry Group'."
        " The'Security' column contains the original input stock_ids,"
        " The Industry Group column contains the name of the GICS level 2 Industry Group"
        " that the stock belongs to."
        " Use this function to add Industry Group to a dataset before display"
        " or for further processing."
        " 'Industry Groups' are also known as 'Sub Sectors' or 'subsectors'."
        " If you want to filter on an Industry group do not use this function,"
        " you should use the sector_filter instead"
    ),
    category=ToolCategory.STOCK_GROUPS,
    tool_registry=default_tool_registry(),
    is_visible=True,
)
async def get_industry_group_for_stocks(
    args: GetIndustryGroupInput, context: PlanRunContext
) -> StockTable:
    """Returns the Industry Group for each stock

    Returns:
        StockTable: a table of stock identifiers and industry groups.
    """

    df = await get_metadata_for_stocks(args.stock_ids, context)
    new_stock_ids = await StockID.from_gbi_id_list(list(df["gbi_id"]))
    df["Security"] = new_stock_ids

    table = StockTable.from_df_and_cols(
        data=df,
        columns=[
            TableColumnMetadata(label="Security", col_type=TableColumnType.STOCK),
            TableColumnMetadata(label="Industry Group", col_type=TableColumnType.STRING),
        ],
    )

    return table


class GetIndustryInput(ToolArgs):
    stock_ids: List[StockID]


@tool(
    description=(
        "This function takes a list of StockIds"
        " and returns a table with columns named: 'Security' and 'Industry '."
        " The'Security' column contains the original input stock_ids,"
        " The Industry column contains the name of the GICS level 3 Industry"
        " that the stock belongs to."
        " Use this function to add Industry to a dataset before display"
        " or for further processing."
        " If you want to filter on an Industry do not use this function,"
        " you should use the sector_filter instead"
    ),
    category=ToolCategory.STOCK_GROUPS,
    tool_registry=default_tool_registry(),
    is_visible=True,
)
async def get_industry_for_stocks(args: GetIndustryInput, context: PlanRunContext) -> StockTable:
    """Returns the Industry  for each stock

    Returns:
        StockTable: a table of stock identifiers and industries.
    """

    df = await get_metadata_for_stocks(args.stock_ids, context)
    new_stock_ids = await StockID.from_gbi_id_list(list(df["gbi_id"]))
    df["Security"] = new_stock_ids

    table = StockTable.from_df_and_cols(
        data=df,
        columns=[
            TableColumnMetadata(label="Security", col_type=TableColumnType.STOCK),
            TableColumnMetadata(label="Industry", col_type=TableColumnType.STRING),
        ],
    )

    return table


class GetSubIndustryInput(ToolArgs):
    stock_ids: List[StockID]


@tool(
    description=(
        "This function takes a list of StockIds"
        " and returns a table with columns named: 'Security' and 'Sub Industry '."
        " The'Security' column contains the original input stock_ids,"
        " The Sub Industry column contains the name of the GICS level 4 Sub Industry"
        " that the stock belongs to."
        " Use this function to add Sub Industry to a dataset before display"
        " or for further processing."
        " If you want to filter on a Sub Industry do not use this function,"
        " you should use the sector_filter instead"
    ),
    category=ToolCategory.STOCK_GROUPS,
    tool_registry=default_tool_registry(),
    is_visible=True,
)
async def get_sub_industry_for_stocks(
    args: GetSubIndustryInput, context: PlanRunContext
) -> StockTable:
    """Returns the Sub Industry  for each stock

    Returns:
        StockTable: a table of stock identifiers and sub industries.
    """

    df = await get_metadata_for_stocks(args.stock_ids, context)
    new_stock_ids = await StockID.from_gbi_id_list(list(df["gbi_id"]))
    df["Security"] = new_stock_ids

    table = StockTable.from_df_and_cols(
        data=df,
        columns=[
            TableColumnMetadata(label="Security", col_type=TableColumnType.STOCK),
            TableColumnMetadata(label="Sub Industry", col_type=TableColumnType.STRING),
        ],
    )

    return table


async def get_metadata_for_stocks(
    stock_ids: List[StockID], context: PlanRunContext
) -> pd.DataFrame:
    """Returns the metadata for each stock

    Args:
        stock_ids: List[StockID]
        context (PlanRunContext): The context of the plan run.

    Returns:
        dataframe: a table of stock identifiers and all metadata columns.
    """

    db = get_async_db(read_only=True)

    gbi_ids = await StockID.to_gbi_id_list(stock_ids)

    rows = await get_stock_metadata_rows(gbi_ids=gbi_ids, pg=db.pg)
    df = pd.DataFrame(rows)

    # the column names need to be changed to match the output column labels
    df.rename(
        columns={
            "country": "Country",
            "country_of_domicile": "Country of Domicile",
            "currency": "Currency",
            "isin": "ISIN",
            "gics1_name": "Sector",
            "gics2_name": "Industry Group",
            "gics3_name": "Industry",
            "gics4_name": "Sub Industry",
        },
        inplace=True,
    )
    return df


class GetInternationalCapStartingUniverseInput(ToolArgs):
    pass


@tool(
    description=(
        "This function only gives you the stocks you need to start with to"
        " later filter on by region and market cap,"
        " they are not pre-filtered to meet that criteria."
        " This function should only be used when you need a starting list of stocks to be later"
        " filtered by non-USA country, region or countries that will then also later be filtered by"
        " small, medium, or mid cap stocks by market cap range."
        " If the user does not mention a country or region, do not use this function!"
        " I repeat the output list MUST be further filtered by a country first using region_filter"
        " and then filtered by a market cap range after the region filter."
        " The filter by market cap can be achieved by using the get_statistic_data_for_companies"
        " function to get market cap and followed by transform_table to filter to the"
        " correct market cap range such as small cap, mid cap, medium cap."
        " These steps MUST happen immediatelty after calling this function and before calling "
        " prepare_output or you will be FIRED!!"
        " I REPEAT this function does not do any country or market cap filtering itself,"
        " you ABSOLUTELY MUST perform the market cap and country filtering in"
        " additional steps in the plan!"
    ),
    category=ToolCategory.STOCK,
    tool_registry=default_tool_registry(),
    enabled=False,
)
async def get_international_cap_starting_universe(
    args: GetInternationalCapStartingUniverseInput, context: PlanRunContext
) -> List[StockID]:
    target_tickers = [
        "IWSZ LN Equity",  # iShares MSCI World Mid-Cap Equal Weight UCITS ETF
        "WSML LN Equity",  # iShares MSCI World Small Cap UCITS ETF
        "VSS US Equity",  # Vanguard FTSE All-World ex-US Small-Cap ETF
        "VT US Equity",  # Vanguard International Equity Index Funds - Vanguard Total World Stock ETF
        "PDN US Equity",  # Invesco FTSE RAFI Developed Markets ex-U.S. Small-Mid ETF
        "IEUS US Equity",  # iShares Trust - iShares MSCI Europe Small-Cap ETF
        "ISFE LN Equity",  # iShares II Public Limited Company - iShares MSCI AC Far East ex-Japan Small Cap UCITS ETF
        "SCJ US Equity",  # iShares, Inc. - iShares MSCI Japan Small-Cap ETF
    ]

    await tool_log(log="Getting a list of international stocks to start with", context=context)

    current: List[StockID] = []
    for tkr in target_tickers:
        arg1 = GetStockUniverseInput(universe_name=tkr)
        stocks: List[StockID] = await get_stock_universe(args=arg1, context=context)  # type:ignore
        arg2 = CombineListsInput(list1=current, list2=stocks)  # type:ignore
        current = await add_lists(args=arg2, context=context)  # type:ignore

    return current


async def is_etf(gbi_id: int) -> bool:
    db = get_async_db(read_only=True)
    sql = """
    SELECT
    gbi_security_id
    FROM master_security ms
    WHERE
    gbi_security_id = %(gbi_id)s
    AND (ms.security_type like '%%ETF%%' OR ms.security_type like '%%Fund%%')
    """
    params = {"gbi_id": gbi_id}
    rows = await db.generic_read(
        sql,
        params=params,
    )

    if not rows:
        return False

    return True


class GetStockUniverseInput(ToolArgs):
    # name of the universe to lookup
    universe_name: str
    date_range: Optional[DateRange] = None
    dedup_companies: bool = False


@tool(
    description=(
        "This function takes a string and an optional date range"
        " which refers to a stock universe, and converts it to a string identifier "
        " and then returns the list of stock identifiers in the universe."
        " Stock universes are generally major market indexes like the S&P 500 or the"
        " Stoxx 600 or the similar names of ETFs or the 3-6 letter ticker symbols for ETFs"
        " You should use this tool if you want to filter or otherwise process individual"
        " stocks in an ETF, but note that you should NOT look up the identifier for the ETF"
        " using stock_identifier_lookup before calling this tool, pass the name of the ETF"
        " as a string directly."
        " If a client asks for the companies in the universe "
        " you must set dedup_companies to True. If a client asks for the stocks in a universe "
        " you must set dedup_companies to False. It is of the utmost importance that if a client "
        " asks for stocks in a universe (like stocks in the SPY ETF), you set dedup_companies "
        " to false."
        " If the client wants to filter over stocks but does not specify an initial set"
        " of stocks, you should call this tool with 'S&P 500'."
        " You can also use this tool to get the holdings of an ETF or stock."
        " But not the holdings of a user's portfolio, "
        " if you do need portfolio holdings then use get_portfolio_holdings tool instead."
        " If you need the weights for the stocks in the universe "
        " then you should call get_universe_holdings instead. "
        " \n - Some example phrases that imply you should use this function are:"
        " 'stocks in', 'companies in', 'holdings of'"
        " Please be careful not to confuse the r1k (Russell 1000), the r2k (Russell 2000),"
        " and the r3k (Russell 3000) when using this tool, these are very different universes!"
        " Please be careful to run this tool as a separate step in your plan!"
    ),
    category=ToolCategory.STOCK,
    tool_registry=default_tool_registry(),
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

    if "INTERNATIONAL_SMALL" == args.universe_name.upper().replace(" ", "_"):
        await tool_log(
            log="Looking up a list of international stocks including small & midcap",
            context=context,
        )

        results = await get_international_cap_starting_universe(
            args=GetInternationalCapStartingUniverseInput(), context=context
        )
        # List[StockID]
        return results  # type: ignore

    etf_stock = await get_stock_info_for_universe(args, context)
    universe_spiq_company_id = etf_stock["spiq_company_id"]
    stock_universe_table = await get_stock_universe_table_from_universe_company_id(
        universe_spiq_company_id=universe_spiq_company_id,
        date_range=args.date_range,
        dedup_companies=args.dedup_companies,
        context=context,
    )
    stock_universe_list = stock_universe_table.to_df()[STOCK_ID_COL_NAME_DEFAULT].tolist()

    if not stock_universe_list:
        gbi_id = etf_stock["gbi_security_id"]
        if not await is_etf(gbi_id):
            logger.warning(f"not an ETF: {etf_stock}")
            raise NotFoundError(f"{etf_stock['symbol']} - {etf_stock['name']} is not an ETF")

    date_clause = ""
    if args.date_range:
        if args.date_range.start_date == args.date_range.end_date:
            date_clause = f" on {args.date_range.start_date.isoformat()}"
        else:
            date_clause = (
                f" between {args.date_range.start_date.isoformat()}"
                f" and {args.date_range.end_date.isoformat()}"
            )

    logger.info(
        f"found {len(stock_universe_list)} holdings{date_clause} in ETF: {etf_stock} from '{args.universe_name}'"
    )

    await tool_log(
        log=f"Found {len(stock_universe_list)} holdings{date_clause} in {etf_stock['symbol']}: {etf_stock['name']}",
        context=context,
    )

    try:  # since everything associated with diffing is optional, put in try/except
        # Update mode
        if context.task_id:
            # we need to add the task id to all runs, including the first one, so we can track changes
            stock_universe_list = add_task_id_to_stocks_history(
                stock_universe_list, context.task_id
            )
            if context.diff_info is not None:
                prev_run_info = await get_prev_run_info(context, "get_stock_universe")
                if prev_run_info is not None:
                    prev_output: List[StockID] = prev_run_info.output  # type:ignore
                    curr_stock_set = set(stock_universe_list)
                    prev_stock_set = set(prev_output)
                    added_stocks = curr_stock_set - prev_stock_set
                    removed_stocks = prev_stock_set - curr_stock_set
                    context.diff_info[context.task_id] = {
                        "added": {
                            added_stock: UNIVERSE_ADD_STOCK_DIFF.format(
                                company=added_stock.company_name, universe=args.universe_name
                            )
                            for added_stock in added_stocks
                        },
                        "removed": {
                            removed_stock: UNIVERSE_REMOVE_STOCK_DIFF.format(
                                company=removed_stock.company_name, universe=args.universe_name
                            )
                            for removed_stock in removed_stocks
                        },
                    }

    except Exception as e:
        logger.exception(f"Error using previous run cache: {e}")
        pager_wrapper(
            current_frame=inspect.currentframe(),
            module_name=__name__,
            context=context,
            e=e,
            classt="AgentUpdateError",
            summary="Failed to get stock universe",
        )

    return stock_universe_list


async def get_stock_info_for_universe(args: GetStockUniverseInput, context: PlanRunContext) -> Dict:
    """Returns the company id of the best match universe.

    Args:
        args (GetStockUniverseInput): The input arguments for the stock universe lookup.
        context (PlanRunContext): The context of the plan run.

    Returns:
        Dict: gbi_id, company id,  name
    """
    logger = get_prefect_logger(__name__)

    # Find the universe id/name by reusing the stock lookup, and then filter by ETF
    etf_stock_match = await get_stock_universe_from_etf_stock_match(args, context)

    if etf_stock_match:
        logger.info(f"Found ETF directly for '{args.universe_name}'")
        stock = etf_stock_match
    else:
        logger.info(f"Could not find ETF directly for '{args.universe_name}'")
        gbi_uni_row = await get_stock_universe_gbi_stock_universe(args, context)
        if gbi_uni_row:
            stock = gbi_uni_row
        else:
            raise NotFoundError(
                f"Could not find any stock universe related to: '{args.universe_name}'"
            )

    return stock


async def get_stock_universe_table_from_universe_company_id(
    universe_spiq_company_id: int,
    date_range: Optional[DateRange],
    context: PlanRunContext,
    dedup_companies: bool = False,
) -> StockTable:
    """Returns the list of stock identifiers given a stock universe's company id.

    Args:
        universe_spiq_company_id: int
        date_range: DateRange
        context (PlanRunContext): The context of the plan run.

    Returns:
        StockTable: The table of stock identifiers and weights in the universe.
    """
    # logger = get_prefect_logger(__name__)
    db = get_async_db(read_only=True)
    gbi_ids = []
    rows = []
    deduped_rows_by_company = []

    if date_range:
        start_date = date_range.start_date
        end_date = date_range.end_date

        # Find the stocks in the universe
        query = """
        SELECT DISTINCT ON (gbi_id)
        gbi_id, symbol, ms.isin, name, weight
        FROM "data".etf_universe_holdings euh
        JOIN master_security ms ON ms.gbi_security_id = euh.gbi_id
        WHERE spiq_company_id = %(spiq_company_id)s AND ms.is_public
        AND
        (
        -- the start date lands inside of a row's from/to daterange
        (euh.from_z <= %(start_date)s AND %(start_date)s <= euh.to_z  )

        OR

        -- the row's date range is
        -- entirely more than the start date
        -- and entirely less than the end date
        -- start date <= row_date_range <= end_date
        (%(start_date)s <= euh.from_z AND euh.to_z <= %(end_date)s)

        OR

        -- the end_date lands inside of a row's daterange
        (euh.from_z <= %(end_date)s AND %(end_date)s <= euh.to_z  )
        )
        """

        params = {
            "spiq_company_id": universe_spiq_company_id,
            "start_date": start_date,
            "end_date": end_date,
        }

        full_rows = await db.generic_read(
            query,
            params=params,
        )

        if dedup_companies:
            dedup_query = f"""
                SELECT DISTINCT ON (name)
                gbi_id, symbol, isin, name, weight
                FROM ({query})
                AS subquery
            """
            deduped_rows_by_company = await db.generic_read(
                dedup_query,
                params=params,
            )
    else:
        # Find the stocks in the universe
        query = """
        SELECT DISTINCT ON (gbi_id)
        gbi_id, symbol, ms.isin, name, weight
        FROM "data".etf_universe_holdings euh
        JOIN master_security ms ON ms.gbi_security_id = euh.gbi_id
        WHERE spiq_company_id =  %(spiq_company_id)s AND ms.is_public
        AND euh.to_z > %(now_utc)s
        """

        now_utc = get_now_utc()
        params = {"spiq_company_id": universe_spiq_company_id, "now_utc": now_utc}
        full_rows = await db.generic_read(query, params=params)

        if dedup_companies:
            dedup_query = f"""
                SELECT DISTINCT ON (name)
                gbi_id, symbol, isin, name, weight
                FROM ({query})
                AS subquery
            """
            deduped_rows_by_company = await db.generic_read(dedup_query, params=params)

    if dedup_companies:
        await tool_log(
            f"Retrieved {len(deduped_rows_by_company)} unique companies,"
            f" ignoring {len(full_rows) - len(deduped_rows_by_company)} redundant stocks",
            context,
        )
        rows = deduped_rows_by_company
    else:
        await tool_log(
            f"Retrieved {len(full_rows)} stocks, may contain multiple stocks for the same company",
            context,
        )
        rows = full_rows

    gbi_ids = [row["gbi_id"] for row in rows]
    stock_ids = await StockID.from_gbi_id_list(gbi_ids)
    data = {
        STOCK_ID_COL_NAME_DEFAULT: stock_ids,
        "Weight": [row["weight"] / 100 for row in rows],
    }
    df = pd.DataFrame(data)
    table = StockTable.from_df_and_cols(
        data=df,
        columns=[
            TableColumnMetadata(label=STOCK_ID_COL_NAME_DEFAULT, col_type=TableColumnType.STOCK),
            TableColumnMetadata(label="Weight", col_type=TableColumnType.PERCENT),
        ],
    )

    return table


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
    # logger = get_prefect_logger(__name__)
    db = get_async_db(read_only=True)

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
    rows = await db.generic_read(sql, [spiq_company_ids])

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
        " < -1 to get 'low' scores, and < -2 to get 'very low' scores. "
        " This are also known as risk factors or factor weights, you should get this table "
        " if someone asks for numerical risk factors associated with stocks. "
        "Use this tool if the user asks to filter/rank by one of these factors specifically, "
        "but you must never use it unless what the client says corresponds exactly or almost "
        "exactly to one of the relevant factors, you must use one or more of the provided "
        "factor names exactly in your instructions to the transform_table tool, if you cannot do that "
        "you should probably filter by profile instead. "
        "You must not use this tool if the user asks for something related to the particular value "
        "of particular statistic, even if it is closely related to one of these factors, use the "
        "get_statistic tool instead. For example you would never use this tool for queries "
        "asking for filtering based on market cap, which is a specific statistic."
        "Do not use this tool to try to filter by Growth at a Reasonable Price (GARP) "
        "instead use a combination of growth_filter and value_filter. "
        "Do not use this tool to determine if any risk factor exposure"
        " has increased or decreased across time"
        " it can only tell you the current exposure."
        "The output table will always include all the factors. If the user is not interested in all "
        "the factors, but only a specific one (like momentum, or leverage) you must immediately call "
        "transform_table and explicitly ask to remove all factors but the one(s) the user wants. It "
        "is extremely important that you do not display factors the user is not interested in!!!!! "
        "To help you remember this step, make sure you always call the output of this tool 'factor_table' "
        "and then call transform tool to turn it into a table with the factor you need."
    ),
    category=ToolCategory.STATISTICS,
    tool_registry=default_tool_registry(),
    is_visible=True,
)
async def get_risk_exposure_for_stocks(
    args: GetRiskExposureForStocksInput, context: PlanRunContext
) -> Table:
    if len(args.stock_list) == 0:
        raise EmptyInputError("Cannot get risk exposure for empty list of stocks")

    # logger = get_prefect_logger(__name__)

    def format_column_name(col_name: str) -> str:
        if "_" in col_name:
            return col_name.replace("_", " ").title()
        else:
            return col_name.title()

    env = get_environment_tag()
    # TODO when risk model ism integration is complete
    # accept a risk model id as input and default to NA model
    # Default to SP 500
    DEV_SP500_UNIVERSE_ID = "249a293b-d9e2-4905-94c1-c53034f877c9"
    PROD_SP500_UNIVERSE_ID = "4e5f2fd3-394e-4db9-aad3-5c20abf9bf3c"

    universe_id = DEV_SP500_UNIVERSE_ID
    if env == PROD_TAG:  # AKA ALPHA
        # If we are in Prod use SPY
        universe_id = PROD_SP500_UNIVERSE_ID

    exposures = await universe_stock_factor_exposures(
        user_id=context.user_id,
        universe_id=universe_id,
        # NA risk model id is 5 in both DEV and PROD
        risk_model_id=5,
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
    df.columns = pd.Index([format_column_name(col) for col in df.columns])
    df = df.drop("Idiosyncratic", axis=1)

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
    max_value: float = 10
    stocks_to_keep: Optional[int] = None


@tool(
    description=(
        "This function takes a list of stock ids"
        " and filters them acccording to how growth-y they are"
        " if no stock_list is provided, a default list will be used"
        " min_value will default to 1 standard deviation,"
        " the larger the value then the filterd stocks will be even more growthy"
        " you must only use this function if the client specifically asks to filter by"
        " growth, do not use it for specific statistics, even if they are growth-related"
        " set min_value 1 to get 'high growth', 2 to get 'very high growth'"
        " 3 to get 'extremely high growth'."
        " stocks_to_keep will trim the final list of stocks to length stocks_to_keep and"
        " is useful when the client wants a specific number of growth stocks. It defaults"
        " to None, in which case the list is not trimmed."
        " This tool is useful for finding growth stocks."
        " It can be combined with 'value_filter' tool to find GARP/Growth at a Reasonable Price."
        " This function should only be used to determining the general growthiness of a stock."
        "\n- When not to use this function:"
        " This function MUST NOT be used to get 'expected growth' of something"
        " or 'stocks expected to grow'"
        " or specific types of growth like 'revenue growth' ."
        " In those cases use the get_statistic_data_for_companies function instead ."
        " 'low growth' should be interpreted as min_value = -10, max_value = -1 ."
        " 'good growth' should be interpreted the same as high growth."
    ),
    category=ToolCategory.STOCK_FILTERS,
    tool_registry=default_tool_registry(),
    is_visible=True,
)
async def growth_filter(args: GrowthFilterInput, context: PlanRunContext) -> List[StockID]:
    logger = get_prefect_logger(__name__)
    stock_ids = args.stock_ids
    if stock_ids == []:
        # degenerate case should i log or throw?
        await tool_log(log="No stocks left to filter by 'growth'", context=context)
        return []

    if stock_ids is None:
        stock_uni_args = GetStockUniverseInput(universe_name="S&P 500")
        stock_ids = await get_stock_universe(stock_uni_args, context)  # type: ignore
        if not stock_ids:
            raise RuntimeError("could not retrieve default stock list")

    if stock_ids is None:
        logger = get_prefect_logger(__name__)
        logger.info("we need universe stocks to proceed")
        return []

    risk_args = GetRiskExposureForStocksInput(stock_list=stock_ids)

    risk_table = await get_risk_exposure_for_stocks(risk_args, context)
    # mypy thinks this is not a table but a generic ComplexIO Base
    df = risk_table.to_df()  # type: ignore
    filtered_df = df.loc[
        (df[GROWTH_LABEL] >= args.min_value) & (df[GROWTH_LABEL] <= args.max_value)
    ]
    stock_list = filtered_df[STOCK_ID_COL_NAME_DEFAULT].squeeze().to_list()
    await tool_log(
        log=f"Filtered {len(stock_ids)} stocks down to {len(stock_list)}", context=context
    )

    try:  # since everything associated with diffing is optional, put in try/except
        # we need to add the task id to all runs, including the first one, so we can track changes
        # Update Mode
        if context.task_id:
            stock_list = add_task_id_to_stocks_history(stock_list, context.task_id)
            if context.diff_info is not None:
                # 2nd arg is the name of the function we are in
                prev_run_info = await get_prev_run_info(context, "growth_filter")
                if prev_run_info is not None:
                    prev_input = GrowthFilterInput.model_validate_json(prev_run_info.inputs_str)
                    prev_output: List[StockID] = prev_run_info.output  # type:ignore
                    # corner case here where S&P 500 change causes output to change, but not going to
                    # bother with it on first pass
                    if args.stock_ids and prev_input.stock_ids:
                        # we only care about stocks that were inputs for both
                        shared_inputs = set(prev_input.stock_ids) & set(args.stock_ids)
                    else:
                        shared_inputs = set()
                    curr_stock_set = set(stock_list)
                    prev_stock_set = set(prev_output)
                    added_stocks = (curr_stock_set - prev_stock_set) & shared_inputs
                    removed_stocks = (prev_stock_set - curr_stock_set) & shared_inputs
                    context.diff_info[context.task_id] = {
                        "added": {
                            added_stock: FACTOR_ADD_STOCK_DIFF.format(
                                company=added_stock.company_name, factor="growth"
                            )
                            for added_stock in added_stocks
                        },
                        "removed": {
                            removed_stock: FACTOR_REMOVE_STOCK_DIFF.format(
                                company=removed_stock.company_name, factor="growth"
                            )
                            for removed_stock in removed_stocks
                        },
                    }

    except Exception as e:
        logger.exception(f"Error creating diff info from previous run: {e}")
        pager_wrapper(
            current_frame=inspect.currentframe(),
            module_name=__name__,
            context=context,
            e=e,
            classt="AgentUpdateError",
            summary="Failed to update per stock summary",
        )

    if args.stocks_to_keep:
        stock_list = stock_list[: args.stocks_to_keep]

    return stock_list


class ValueFilterInput(ToolArgs):
    stock_ids: Optional[List[StockID]] = None
    min_value: float = 1
    max_value: float = 10
    stocks_to_keep: Optional[int] = None


@tool(
    description=(
        "This function takes a list of stock ids"
        " and filters them acccording to how value-y they are"
        " if no stock_list is provided, a default list will be used"
        " min_value will default to 1 standard deviation,"
        " the larger the value then the filtered stocks will be even more valuey"
        " you must only use this function if the client specifically asks to filter by"
        " value, do not use it for specific statistics, even if they are value-related"
        " set min_value 1 to get 'high value', 2 to get 'very high value'"
        " 3 to get 'extremely high value'."
        " stocks_to_keep will trim the final list of stocks to length stocks_to_keep and"
        " is useful when the client wants a specific number of value stocks. It defaults"
        " to None, in which case the list is not trimmed."
        " This tool is useful for finding value stocks."
        " It can be combined with 'growth_filter' tool to find GARP/Growth at a Reasonable Price."
        " 'overvalued stocks' should be interpreted as min_value = -10, max_value = -1 ."
        " 'low value' should be interpreted as min_value = -10, max_value = -1 ."
        " 'undervalued' should be interpreted as min_value = 1, max_value = 10 ."
        " 'good value' should be interpreted the same as high value."
    ),
    category=ToolCategory.STOCK_FILTERS,
    tool_registry=default_tool_registry(),
    is_visible=True,
)
async def value_filter(args: ValueFilterInput, context: PlanRunContext) -> List[StockID]:
    logger = get_prefect_logger(__name__)
    stock_ids = args.stock_ids
    if stock_ids == []:
        # degenerate case should i log or throw?
        await tool_log(log="No stocks left to filter by 'value'", context=context)
        return []

    if stock_ids is None:
        stock_uni_args = GetStockUniverseInput(universe_name="S&P 500")
        stock_ids = await get_stock_universe(stock_uni_args, context)  # type: ignore
        if not stock_ids:
            raise RuntimeError("could not retrieve default stock list")

    if stock_ids is None:
        logger = get_prefect_logger(__name__)
        logger.info("we need universe stocks to proceed")
        return []

    risk_args = GetRiskExposureForStocksInput(stock_list=stock_ids)

    risk_table = await get_risk_exposure_for_stocks(risk_args, context)

    # mypy thinks this is not a table but a generic ComplexIO Base
    df = risk_table.to_df()  # type: ignore
    filtered_df = df.loc[(df[VALUE_LABEL] >= args.min_value) & (df[VALUE_LABEL] <= args.max_value)]
    stock_list = filtered_df[STOCK_ID_COL_NAME_DEFAULT].squeeze().to_list()

    await tool_log(
        log=f"Filtered {len(stock_ids)} stocks down to {len(stock_list)}", context=context
    )

    try:  # since everything associated with diffing is optional, put in try/except
        # we need to add the task id to all runs, including the first one, so we can track changes
        if context.task_id:
            stock_list = add_task_id_to_stocks_history(stock_list, context.task_id)
            if context.diff_info is not None:
                # 2nd arg is the name of the function we are in
                prev_run_info = await get_prev_run_info(context, "value_filter")
                if prev_run_info is not None:
                    prev_input = ValueFilterInput.model_validate_json(prev_run_info.inputs_str)
                    prev_output: List[StockID] = prev_run_info.output  # type:ignore
                    # corner case here where S&P 500 change causes output to change, but not going to
                    # bother with it on first pass
                    if args.stock_ids and prev_input.stock_ids:
                        # we only care about stocks that were inputs for both
                        shared_inputs = set(prev_input.stock_ids) & set(args.stock_ids)
                    else:
                        shared_inputs = set()
                    curr_stock_set = set(stock_list)
                    prev_stock_set = set(prev_output)
                    added_stocks = (curr_stock_set - prev_stock_set) & shared_inputs
                    removed_stocks = (prev_stock_set - curr_stock_set) & shared_inputs
                    context.diff_info[context.task_id] = {
                        "added": {
                            added_stock: FACTOR_ADD_STOCK_DIFF.format(
                                company=added_stock.company_name, factor="value"
                            )
                            for added_stock in added_stocks
                        },
                        "removed": {
                            removed_stock: FACTOR_REMOVE_STOCK_DIFF.format(
                                company=removed_stock.company_name, factor="value"
                            )
                            for removed_stock in removed_stocks
                        },
                    }

    except Exception as e:
        logger.exception(f"Error creating diff info from previous run: {e}")
        pager_wrapper(
            current_frame=inspect.currentframe(),
            module_name=__name__,
            context=context,
            e=e,
            classt="AgentUpdateError",
            summary="Failed to get previous run info or getting default stock list",
        )

    if args.stocks_to_keep:
        stock_list = stock_list[: args.stocks_to_keep]

    return stock_list


async def get_stock_universe_gbi_stock_universe(
    args: GetStockUniverseInput, context: PlanRunContext
) -> Optional[Dict]:
    """
    Returns an optional dict representing the best gbi universe match
    """
    logger = get_prefect_logger(__name__)
    db = get_async_db(read_only=True)

    logger.info(f"looking in gbi_stock_universe for: '{args.universe_name}'")
    sql = """
    SELECT * FROM (
    SELECT etfs.spiq_company_id, etfs.name, ms.gbi_security_id, ms.symbol,
    strict_word_similarity(gsu.name, %s) AS ws,
    gsu.name as gsu_name
    FROM "data".etf_universes etfs
    JOIN gbi_stock_universe gsu
    ON
    etfs.spiq_company_id = (gsu.ingestion_configuration->'ownerObjectId')::INT
    JOIN master_security ms ON ms.gbi_security_id = etfs.gbi_id
    ) as tmp
    ORDER BY ws DESC
    LIMIT 20
    """
    rows = await db.generic_read(sql, [args.universe_name])
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
    db = get_async_db(read_only=True)

    # Find the universe id/name by reusing the stock lookup, and then filter by ETF
    logger.info(f"Attempting to map '{args.universe_name}' to a stock universe")
    stock_args = StockIdentifierLookupInput(stock_name=args.universe_name, prefer_etfs=True)
    stock_rows = []
    try:
        stock_id = await stock_identifier_lookup(stock_args, context)
        stock_id = cast(StockID, stock_id)

        # make this look like a sql row
        stock_row = {
            "gbi_security_id": stock_id.gbi_id,
            "symbol": stock_id.symbol,
            "isin": stock_id.isin,
            "name": stock_id.company_name,
        }
        stock_rows.append(stock_row)
    except ValueError as e:
        logger.info(e)
    except NotFoundError as e:
        logger.info(e)

    if not stock_rows:
        # fall back to the old logic
        raw_stock_rows = await raw_stock_identifier_lookup(stock_args, context)
        stock_rows.extend(raw_stock_rows)

    if not stock_rows:
        return None

    # get company ID
    sql = """
    SELECT ms.gbi_security_id AS gbi_id, ssm.spiq_company_id, ms.name
    FROM master_security ms
    JOIN spiq_security_mapping ssm ON ssm.gbi_id = ms.gbi_security_id
    WHERE ms.gbi_security_id = ANY ( %s )
    """
    potential_etf_gbi_ids = [r["gbi_security_id"] for r in stock_rows]
    company_id_rows = await db.generic_read(sql, [potential_etf_gbi_ids])
    if not company_id_rows:
        # should not be possible
        logger.error("could not find company IDs")
        return None

    gbiid2stock = {r["gbi_security_id"]: r for r in stock_rows}

    # add the company id
    for r in company_id_rows:
        if r["gbi_id"] in gbiid2stock:
            gbiid2stock[r["gbi_id"]]["spiq_company_id"] = r["spiq_company_id"]

    rows = [r for r in gbiid2stock.values() if r.get("spiq_company_id")]
    if not rows:
        logger.error("could not find company IDs")
        return None

    if len(rows) < 5:
        logger.info(f"found {len(rows)} best potential matching ETFs: {rows=}")
    else:
        logger.info(f"found {len(rows)} best potential matching ETFs")

    if 1 == len(rows):
        return rows[0]

    # we have multiple matches, lets use dollar trading volume to choose the most likely match
    rows = await augment_stock_rows_with_volume(context.user_id, rows)
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
    if not args.stock_name:
        raise ValueError("Cannot look up by BBG parsekey if no stock name present")
    logger = get_prefect_logger(__name__)
    db = get_async_db(read_only=True)

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

    sql = f"""
    -- ticker symbol + country (exact match only)
    SELECT gbi_security_id, ms.symbol, ms.isin, ms.security_region, ms.currency,
    ms.name, 'ticker symbol' as match_col, ms.symbol || ' ' || ms.security_region as match_text,
    1.0 AS ws
    FROM master_security ms
    WHERE
    ms.asset_type  in ( {ACCEPTABLE_ASSET_CLASSES} )
    AND ms.is_public
    AND ms.to_z is null
    AND ms.symbol = upper(%(symbol)s)
    AND ms.security_region = upper(%(iso3)s)
    """

    rows = await db.generic_read(sql, {"symbol": symbol, "iso3": iso3})
    if rows:
        logger.info("found bloomberg parsekey match")
        return rows

    logger.info(f"Looks like a bloomberg parsekey but couldn't find a match: '{args.stock_name}'")
    return []


async def stock_lookup_by_ric_yahoo_codes(
    args: StockIdentifierLookupInput, context: PlanRunContext
) -> List[Dict[str, Any]]:
    """Returns the stocks with matching ticker and RIC/Yahoo exchange code
    by mapping the exchange code to a country iso code and matching against our DB

    Examples: "IBM.L", "LNR.TO"

    Args:
        args (StockIdentifierLookupInput): The input arguments for the stock lookup.
        context (PlanRunContext): The context of the plan run.

    Returns:
        List of potential db matches
    """

    logger = get_prefect_logger(__name__)
    db = get_async_db(read_only=True)

    if not args.stock_name:
        raise ValueError("Cannot look up by RIC Yahoo if no stock name present")
    search_term = args.stock_name
    search_term = search_term.strip()

    if "." not in search_term:
        logger.info("not RIC/Yahoo format")
        return []

    # not really needed but just in case
    if search_term.endswith("quity") or search_term.endswith("QUITY"):
        search_term = search_term.replace(" Equity", "")
        search_term = search_term.replace(" equity", "")
        search_term = search_term.replace(" EQUITY", "")

    search_term = search_term.strip()
    search_terms = search_term.split(".")  # example: LNR.TO

    MAX_TOKENS = 2
    MAX_EXCH_CODE_LEN = 4
    MAX_SYMBOL_LEN = 8

    symbol = search_terms[0].upper()
    exch_code = search_terms[-1]
    if (
        len(search_terms) != MAX_TOKENS
        or len(exch_code) > MAX_EXCH_CODE_LEN
        or len(symbol) > MAX_SYMBOL_LEN
    ):
        logger.info("not RIC/Yahoo format")
        return []

    # RIC has some upper and lower case versions
    iso3 = ric_yahoo_exchange_to_country_iso3.get(exch_code.upper())

    if not iso3:
        iso3 = ric_yahoo_exchange_to_country_iso3.get(exch_code.lower())

    if not iso3:
        logger.info(
            f"either '{args.stock_name}' just looked similar to a yahoo/RIC code"
            f" or we are missing an exchange code mapping for: '{exch_code}'"
        )
        return []

    logger.info(f"searching for {symbol=}  {iso3=}")

    sql = f"""
    -- ticker symbol + country (exact match only)
    SELECT gbi_security_id, ms.symbol, ms.isin, ms.security_region, ms.currency,
    ms.name, 'ticker symbol' as match_col, ms.symbol || ' ' || ms.security_region as match_text,
    1.0 AS ws
    FROM master_security ms
    WHERE
    ms.asset_type  in ( {ACCEPTABLE_ASSET_CLASSES} )
    AND ms.is_public
    AND ms.to_z is null
    AND ms.symbol = upper(%(symbol)s)
    AND ms.security_region = upper(%(iso3)s)
    """

    rows = await db.generic_read(sql, {"symbol": symbol, "iso3": iso3})
    if rows:
        logger.info("found RIC/Yahoo code match")

    # SPIQ stores some tickers in format XXXX.YY, so lets find those also just in case
    sql2 = f"""
    -- ticker symbol + country (exact match only)
    SELECT gbi_security_id, ms.symbol, ms.isin, ms.security_region, ms.currency,
    ms.name, 'ticker symbol' as match_col, ms.symbol as match_text,
    1.0 AS ws
    FROM master_security ms
    WHERE
    ms.asset_type  in ( {ACCEPTABLE_ASSET_CLASSES} )
    AND ms.is_public
    AND ms.to_z is null
    AND ms.symbol = upper(%(symbol)s)
    """

    symbol_dot_ex = symbol.upper() + "." + exch_code.upper()

    logger.info(f"searching for {symbol_dot_ex=}")
    rows2 = await db.generic_read(sql2, {"symbol": symbol_dot_ex})

    if rows2:
        logger.info("found SPIQ ticker XXXX.YY format match")

    if not rows:
        rows = []

    if not rows2:
        rows2 = []

    all_rows = rows + rows2
    if all_rows:
        return all_rows

    logger.info(f"Looks like a RIC/Yahoo code but couldn't find a match: '{args.stock_name}'")
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
    "AL": "ALB",  # AL AL ALB
    "DU": "ARE",  # DU AE NASDAQ
    "DH": "ARE",  # UH AE ABU
    "DB": "ARE",  # DU AE ARE
    "UH": "ARE",  # UH AE ARE
    "AM": "ARG",  # AR AR MENDOZA
    "AF": "ARG",  # AR AR BUENOS
    "AC": "ARG",  # AR AR BUENOS
    "AS": "ARG",  # AR AR BUENOS
    "AR": "ARG",  # AR AR ARG
    "AY": "ARM",  # AY AM ARM
    "PF": "AUS",  # AU AU ASIA
    "AQ": "AUS",  # AU AU ASX
    "AH": "AUS",  # AU AU CHIX
    "SI": "AUS",  # AU AU SIM
    "AT": "AUS",  # AU AU ASE
    "AO": "AUS",  # AU AU NSX
    "AU": "AUS",  # AU AU AUS
    "AV": "AUT",  # AV AT VIENNA
    "XA": "AUT",  # AV AT AUT
    "AZ": "AZE",  # AZ AZ AZE
    "BB": "BEL",  # BB BE BEL
    "BD": "BGD",  # BD BD BGD
    "BU": "BGR",  # BU BG BGR
    "BI": "BHR",  # BI BH BHR
    "BM": "BHS",  # BM BS BHS
    "BK": "BIH",  # BK BA BANJA
    "BT": "BIH",  # BT BA BIH
    "RB": "BLR",  # RB BY BLR
    "BH": "BMU",  # BH BM BMU
    "VB": "BOL",  # VB BO BOL
    "BN": "BRA",  # BZ BR SAO
    "BS": "BRA",  # BZ BR BM&FBOVESPA
    "BV": "BRA",  # BZ BR BOVESPA
    "BR": "BRA",  # BZ BR RIO
    "BO": "BRA",  # BZ BR SOMA
    "BZ": "BRA",  # BZ BR BRA
    "BA": "BRB",  # BA BB BRB
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
    "SW": "CHE",  # VX CH CHE
    "CE": "CHL",  # CI CL SAINT
    "CC": "CHL",  # CI CL SANT.
    "CI": "CHL",  # CI CL CHL
    "C2": "CHN",  # CH CN Nrth
    "CS": "CHN",  # CH CN SHENZHEN
    "CG": "CHN",  # CH CN SHANGHAI
    "C1": "CHN",  # C1 CN Nth
    "CH": "CHN",  # C1 CN CHN
    "IA": "CIV",  # IA CI ABIDJAN
    "BC": "CIV",  # BC CI BRVM
    "ZS": "CIV",  # ZS CI CIV
    "DE": "CMR",  # DE CM CMR
    "CX": "COL",  # CB CO BOLSA
    "CB": "COL",  # CB CO COL
    "VR": "CPV",  # VR CV CPV
    "CR": "CRI",  # CR CR CRI
    "KY": "CYM",  # KY KY CYM
    "CY": "CYP",  # CY CY NICOSIA
    "YC": "CYP",  # CY CY CYP
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
    "GR": "DEU",  # QT DE DEU
    "DD": "DNK",  # DC DK DANSK
    "DC": "DNK",  # DC DK COPENHAGEN
    "DF": "DNK",  # DC DK DNK
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
    "ET": "EST",  # ET EE EST
    "FF": "FIN",  # FH FI FN
    "FH": "FIN",  # FH FI FIN
    "FS": "FJI",  # FS FJ FJI
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
    "S2": "GBR",  # S2 GB GBR
    "GG": "GEO",  # GG GE GEO
    "GU": "GGY",  # GU GG GUERNSEY
    "JY": "GGY",  # JY GG GGY
    "GN": "GHA",  # GN GH GHA
    "TL": "GIB",  # TL GI GIB
    "AA": "GRC",  # GA GR ATHENS
    "XT": "GRC",  # EO GR ATHENS
    "AP": "GRC",  # GA GR ATHENS
    "GA": "GRC",  # GA GR GRC
    "GL": "GTM",  # GL GT GTM
    "H1": "HKG",  # H1 HK Sth
    "H2": "HKG",  # HK HK Sth
    "HK": "HKG",  # HK HK HKG
    "HO": "HND",  # HO HN HND
    "ZA": "HRV",  # CZ HR ZAGREB
    "CZ": "HRV",  # CZ HR HRV
    "QM": "HUN",  # QM HU QUOTE
    "HB": "HUN",  # HB HU BUDAPEST
    "XH": "HUN",  # HB HU HUN
    "IJ": "IDN",  # IJ ID IDN
    "IG": "IND",  # IN IN MCX
    "IB": "IND",  # IN IN BSE
    "IH": "IND",  # IN IN DELHI
    "IS": "IND",  # IN IN NATL
    "IN": "IND",  # IN IN IND
    "ID": "IRL",  # ID IE IRELAND
    "XF": "IRL",  # EO IE DUBLIN
    "PO": "IRL",  # PO IE IRL
    "IE": "IRN",  # IE IR IRN
    "IQ": "IRQ",  # IQ IQ IRQ
    "RF": "ISL",  # IR IS FN
    "IR": "ISL",  # IR IS ISL
    "IT": "ISR",  # IT IL ISR
    "TE": "ITA",  # TE IT EUROTLX
    "HM": "ITA",  # HM IT HI-MTF
    "IM": "ITA",  # IM IT BRSAITALIANA
    "IC": "ITA",  # IM IT MIL
    "XI": "ITA",  # EO IT BORSAITALOTC
    "IF": "ITA",  # IM IT ITA
    "JA": "JAM",  # JA JM JAM
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
    "KZ": "KAZ",  # KZ KZ KAZ
    "KN": "KEN",  # KN KE KEN
    "KB": "KGZ",  # KB KG KGZ
    "KH": "KHM",  # KH KH KHM
    "EK": "KNA",  # EK KN ESTN
    "AI": "KNA",  # AI KN ANGUILLA
    "NX": "KNA",  # NX KN KNA
    "KF": "KOR",  # KF KR KOREAFRBMKT
    "KE": "KOR",  # KS KR KONEX
    "KP": "KOR",  # KS KR KOREA
    "KQ": "KOR",  # KF KR KOR
    "KS": "KOR",  # KS KR KOR
    "KK": "KWT",  # KK KW KWT
    "LS": "LAO",  # LS LA LAO
    "LB": "LBN",  # LB LB LBN
    "LY": "LBY",  # LY LY LBY
    "SL": "LKA",  # SL LK LKA
    "LH": "LTU",  # LH LT LTU
    "LX": "LUX",  # LX LU LUX
    "LG": "LVA",  # LR LV RIGA
    "LR": "LVA",  # LR LV LVA
    "MC": "MAR",  # MC MA MAR
    "MB": "MDA",  # MB MD MDA
    "MX": "MDV",  # MX MV MDV
    "MM": "MEX",  # MM MX MEX
    "MS": "MKD",  # MS MK MKD
    "MV": "MLT",  # MV MT MLT
    "ME": "MNE",  # ME ME MNE
    "MO": "MNG",  # MO MN MNG
    "MZ": "MOZ",  # MZ MZ MOZ
    "MP": "MUS",  # MP MU MUS
    "MW": "MWI",  # MW MW MWI
    "MQ": "MYS",  # MQ MY MESDAQ
    "MK": "MYS",  # MK MY MYS
    "NW": "NAM",  # NW NA NAM
    "NL": "NGA",  # NL NG NGA
    "NC": "NIC",  # NC NI NIC
    "MT": "NLD",  # MT NL TOM
    "NA": "NLD",  # NA NL EN
    "NR": "NLD",  # NR NL NLD
    "NS": "NOR",  # NO NO NORWAY
    "NO": "NOR",  # NO NO OSLO
    "XN": "NOR",  # NO NO NOR
    "NK": "NPL",  # NK NP NPL
    "NZ": "NZL",  # NZ NZ NZL
    "OM": "OMN",  # OM OM OMN
    "PK": "PAK",  # PA PK KARACHI
    "PA": "PAK",  # PA PK PAK
    "PP": "PAN",  # PP PA PAN
    "PE": "PER",  # PE PE PER
    "PM": "PHL",  # PM PH PHL
    "PB": "PNG",  # PB PG PNG
    "PD": "POL",  # PW PL POLAND
    "PW": "POL",  # PW PL POL
    "PX": "PRT",  # PX PT PEX
    "PL": "PRT",  # PL PT PRT
    "PN": "PRY",  # PN PY PRY
    "PS": "PSE",  # PS PS PSE
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
    "RW": "RWA",  # RW RW RWA
    "AB": "SAU",  # AB SA SAU
    "SP": "SGP",  # SP SG SGP
    "EL": "SLV",  # EL SV SLV
    "SG": "SRB",  # SG RS SRB
    "SK": "SVK",  # SK SK SVK
    "SV": "SVN",  # SV SI LJUBLJANA
    "XJ": "SVN",  # SV SI SVN
    "BY": "SWE",  # BY SE BURGUNDY
    "SF": "SWE",  # SS SE FN
    "NG": "SWE",  # SS SE
    "XG": "SWE",  # EO SE NGM
    "XO": "SWE",  # EO SE OMX
    "KA": "SWE",  # SS SE AKTIE
    "SS": "SWE",  # SS SE SWE
    "SD": "SWZ",  # SD SZ SWZ
    "SZ": "SYC",  # SZ SC SYC
    "SY": "SYR",  # SY SY SYR
    "TB": "THA",  # TB TH THA
    "TP": "TTO",  # TP TT TTO
    "TU": "TUN",  # TU TN TUN
    "TI": "TUR",  # TI TR ISTANBUL
    "TF": "TUR",  # TI TR ISTN
    "TS": "TUR",  # TI TR TUR
    "TT": "TWN",  # TT TW TWN
    "TZ": "TZA",  # TZ TZ TZA
    "UG": "UGA",  # UG UG UGA
    "UZ": "UKR",  # UZ UA PFTS
    "QU": "UKR",  # UZ UA PFTS
    "UK": "UKR",  # UZ UA UKR
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
    "ZU": "UZB",  # ZU UZ UZB
    "VS": "VEN",  # VC VE CARACAS
    "VC": "VEN",  # VC VE VEN
    "VH": "VNM",  # VN VN HANOI
    "VU": "VNM",  # VN VN HANOI
    "VM": "VNM",  # VN VN HO
    "VN": "VNM",  # VN VN VNM
    "SJ": "ZAF",  # SJ ZA ZAF
    "ZL": "ZMB",  # ZL ZM ZMB
    "ZH": "ZWE",  # ZH ZW ZWE
}


# this was built by merging a list of yahoo exchange suffixes and Reuters/Refinitive RIC code suffixes
# RIC codes took precedence and only used yahoo list if not already in RIC list
#
# https://en.wikipedia.org/wiki/Ticker_symbol
# https://en.wikipedia.org/wiki/Refinitiv_Identification_Code
#
# yahoo codes
# https://in.help.yahoo.com/kb/finance-app-for-ios/exchanges-data-providers-yahoo-finance-sln2310.html # noqa
# https://lists.gnucash.org/docs/C/gnucash-manual/fq-spec-yahoo.html
#
# RIC codes (could not find an official source, but I notices a few errors that I corrected.. could be more though!)
# some have a duplicate code, leaving them here for documentation purposes
# I confirmed that all such dupes map to the same country
# https://community.developers.refinitiv.com/questions/92114/how-to-find-the-list-with-the-exchange-for-which-t.html
ric_yahoo_exchange_to_country_iso3 = {
    # EXCH_ID :	ISO3 | Mnemonic |  Exchange Name | Country
    "AD": "ARE",  # ABD	Abu Dhabi Securities Exch	UAE
    "DI": "ARE",  # DIX	Nasdaq Dubai(ex-DIFX)	UAE
    "DU": "ARE",  # DBX	Dubai Financial	UAE
    "AE": "ARE",  # Yahoo	Dubai Financial Market	United Arab Emirates
    "BA": "ARG",  # BUE	Buenos Aires	Argentina
    "MZA": "ARG",  # MEN	MENDOZA STOCK EXCHANGE	Argentina
    "RF": "ARG",  # RFX	ROSARIO FUTURES EXCHANGE	Argentina
    "XA": "ARM",  # ARM	NASDAQ OMX Armenia	Armenia
    "SF": "AUS",  # SFE	Sydney Futures Exch	Aus/New Zealand
    "AUX": "AUS",  # AUX	Australia Consolidated	Australia
    "AX": "AUS",  # ASX	Australian SE	Australia
    "AXP": "AUS",  # AXP	ASX Pure Match	Australia
    "CHA": "AUS",  # CHA	Chi-X AUS Securities Exch	Australia
    "NH": "AUS",  # NSX	National SE for Australia	Australia
    "v": "AUT",  # OTB	(OETOB) Austria	Austria
    "VI": "AUT",  # VIE	Vienna SE	Austria
    "AZ": "AZE",  # BAE	Baku Stock Exchange	Azerbaijan
    # b" : "BEL", # BFX	Brussels Deriv Exchange	Belgium
    "BR": "BEL",  # BRU	Euronext Brussels	Belgium
    "CJ1": "BGD",  # CHT	Chittagong SE	Bangladesh
    "DH": "BGD",  # DSE	Dhaka Stock Exchange	Bangladesh
    "BB": "BGR",  # BLG	Bulgarian SE	Bulgaria
    "BH": "BHR",  # BAH	Bahrain Bourse	Bahrain
    "HW": "BHR",  # BHX	Bahrain Financial Exchange	Bahrain
    "BJ": "BIH",  # BNL	Banja Luka SE	Bosnia & Herzegovina
    "SJ": "BIH",  # SJR	Sarajevo SE	Bosnia & Herzegovina
    "BSX": "BMU",  # BSX	Bermuda Stock Exchange	Bermuda
    "SA": "BRA",  # SAO	Sao Paulo SE	Brazil
    "SO": "BRA",  # SOMA	SOMA	Brazil
    "BT": "BWA",  # BSM	Botswana SE	Botswana
    "CAB": "CAN",  # CAQ	Canadian BBO Direct	Canada
    "CCP": "CAN",  # CAQ	Canadian BBO Direct	Canada
    "CD": "CAN",  # CNQ	CNSX-Canadian National	Canada
    "CXC": "CAN",  # CXC	NASDAQ CX	Canada
    "CXX": "CAN",  # CXX	NASDAQ CX2	Canada
    "GO": "CAN",  # PTX	Pure Trading	Canada
    "M": "CAN",  # MON	Montreal Exchange	Canada
    "NBC": "CAN",  # NBC	Nasdaq Basic Canada	Canada
    "NEO": "CAN",  # NEO	Aequitas Neo	Canada
    "NLB": "CAN",  # NLB	Aequitas Neo Lit	Canada
    "OMG": "CAN",  # OMG	OMEGA ATS	Canada
    "TMX": "CAN",  # TMX	TMX Select	Canada
    "TO": "CAN",  # TOR	Toronto SE	Canada
    "V": "CAN",  # NEX	TSX Venture-NEX	Canada
    "NE": "CAN",  # Yahoo	Cboe Canada	Canada
    "ALP": "CAN",  # ALP	Alpha Trading Systems	Canada (Toronto)
    "ALV": "CAN",  # ALV	Alpha Trading Systems	Canada (Ventures)
    "BN": "CHE",  # BRN	Berne SE	Switzerland
    # EX" : "CHE", # EUX	Eurex Switzerland	Switzerland
    "S": "CHE",  # VTX	Swiss Blue Chip Segment	Switzerland
    "SW": "CHE",  # Yahoo	Swiss Exchange (SIX)	Switzerland
    "CE": "CHL",  # BEC	Electronic Exchange	Chile
    "SN": "CHL",  # SGO	Santiago SE	Chile
    "SS": "CHN",  # SHH	Shanghai SE	China
    "SZ": "CHN",  # SHZ	Shenzhen SE	China
    "CI": "CIV",  # ABJ	BRVM Ivory	Coast
    "CN": "COL",  # COL	Colombia SE	Colombia
    "CJ": "CRI",  # CRI	Bolsa de Valores Nacional,SA	Costa Rica
    "CY": "CYP",  # CYS	Cyprus SE	Cyprus
    "PR": "CZE",  # PRA	Prague SE	Czech
    "BE": "DEU",  # BER	Berlin SE	Germany
    "D": "DEU",  # DUS	RWB Germany	Germany
    "DE": "DEU",  # GER	Xetra Germany	Germany
    "EW": "DEU",  # EWX	Euwax Germany	Germany
    # EX" : "DEU", # EUX	Eurex Deutschland	Germany
    "F": "DEU",  # FRA	Frankfurt SE	Germany
    "H": "DEU",  # HAM	Hamburg SE	Germany
    "HA": "DEU",  # HAN	Hanover SE	Germany
    "MU": "DEU",  # MUN	Munich SE	Germany
    "SG": "DEU",  # STU	Stuttgart SE	Germany
    "TG": "DEU",  # TDG	Tradegate SE	Germany
    "XR": "DEU",  # XIM	Xetra International Market	Germany
    "BM": "DEU",  # Yahoo	Bremen Stock Exchange	Germany
    "CO": "DNK",  # CPH	Copenhagen SE	Denmark
    "GQ": "ECU",  # GYQ	BOLSA DE	Ecuador
    "QU": "ECU",  # QTO	BOLSA DE QUITO	Ecuador
    "CA": "EGY",  # CAI	Egyptian SE	Egypt
    "PD": "EGY",  # EDP	Primary Dealers Bond Market	Egypt
    "BC": "ESP",  # BAR	Barcelona SE	Spain
    "BI": "ESP",  # BIL	Bilbao SE	Spain
    "ES": "ESP",  # FDI	Spainish Investment Funds	SPAIN
    "i": "ESP",  # MRV	MEFF (Renta Variable)	Spain
    "LA": "ESP",  # LAT	Latino American Market	Spain
    "MA": "ESP",  # MAD	Madrid SE	Spain
    "MC": "ESP",  # MCE	Mercado Continuo	Spain
    "SCT": "ESP",  # SOE	Infobolsa-SpanishOutcry Eq	Spain
    "VA": "ESP",  # VLN	Valencia SE	Spain
    "TL": "EST",  # TLX	Tallinn SE	Estonia
    # h" : "FIN", # FOM	Finnish Options Market	Finland
    "HE": "FIN",  # HEX	Helsinki SE	Finland
    "LN": "FRA",  # LNM	Le Nouveau Marche	France
    "p": "FRA",  # PAR	MONEP France	France
    "PA": "FRA",  # PAR	Euronext Paris	France
    "LT": "GBR",  # LSE	London Latest Touch system	Britain
    "EA": "GBR",  # EDX	European Derivative	UK
    "ED": "GBR",  # EQD	Equiduct UK	UK
    "BCO": "GBR",  # BCO	BATS CHI-X OTC	United Kingdom
    "BS": "GBR",  # BTE	BATS Europe	United Kingdom
    "CH": "GBR",  # CISX	Channel Islands SE	United Kingdom
    "CHI": "GBR",  # CHI	CHI-X Europe	United Kingdom
    "CSX": "GBR",  # CSX	Cayman Island Stock Exchange	United Kingdom
    # Ip" : "GBR", # ISE	Irish Mifid	United Kingdom
    "ISD": "GBR",  # ISD	ICAP Sec & Deriv Exc(ISDX)	United Kingdom
    "L": "GBR",  # LIF	LIFFE United Kingdom	United Kingdom
    "PZ": "GBR",  # PLU	PLUS Markets Group Plc	United Kingdom
    "UP": "GBR",  # UKPX	UK Power Exchange	United Kingdom
    "XC": "GBR",  # Yahoo	Cboe UK	United Kingdom
    "IL": "GBR",  # Yahoo	London Stock Exchange	United Kingdom
    "GH": "GHA",  # GSE	Ghana SE	Ghana
    "AT": "GRC",  # ATH/ADE	Athens SE/Derivative	Greece
    "GG": "GRC",  # BKG	Bank of	Greece
    "GK": "GRC",  # MTF	MTSGreece Greece	Greece
    "HK": "HKG",  # HKG	Hong Kong	Hong
    "HF": "HKG",  # HFE	Hong Kong Futures Exchange	Hong Kong
    "IXH": "HKG",  # IXH	Instinet HK	Hong Kong
    "ZA": "HRV",  # ZAG	Zagreb SE	Croatia
    "BU": "HUN",  # BUD	Budapest SE	Hungary
    "QMF": "HUN",  # QMF	Quote MTF Ltd	Hungary
    "JK": "IDN",  # JKT	Indonesia SE (formerly JSX)	Indonesia
    "BO": "IND",  # BSE	Bombay SE	India
    "CL": "IND",  # CAL	Calcutta SE	India
    "DL": "IND",  # DES	Delhi Stock	India
    "NS": "IND",  # NSI	National SE	India
    "I": "IRL",  # ISE	Irish SE	Ireland
    "Ip": "IRL",  # ISE	The Irish Stock Exchange	Ireland
    "IR": "IRL",  # Yahoo	Euronext Dublin	Ireland
    "IQ": "IRQ",  # ISX	Iraq Stock Exchange	Iraq
    "IC": "ISL",  # ICX	Iceland SE	Iceland
    "TA": "ISR",  # TLV	Tel Aviv SE	Israel
    "TV": "ISR",  # MTT	MTSIsrael	Israel
    "MI": "ITA",  # MIL	Milan SE	Italy
    "TI": "ITA",  # ETX	Euro TLX	Italy
    "TX": "ITA",  # ETX	Euro TLX	ITALY
    "JS": "JAM",  # JAM	Jamaica Stock Exchange	Jamaica
    "AM": "JOR",  # AMM	Amman SE	Jordan
    "FU": "JPN",  # FKA	Fukuoka SE	Japan
    "JA": "JPN",  # JNA	Tokyo AIM	Japan
    "KY": "JPN",  # KYO	Kyoto SE	Japan
    "NG": "JPN",  # NGO	Nagoya SE	Japan
    "OS": "JPN",  # OSA	Osaka SE	Japan
    "SP": "JPN",  # SAP	Sapporo Stock Exchange	Japan
    "T": "JPN",  # TYO	Tokyo SE	Japan
    "KZ": "KAZ",  # KAZ	Kazakhstan Stock Exchange	Kazakhstan
    "NR": "KEN",  # NAI	Nairobi SE	Kenya
    "KE": "KOR",  # KFE	KOFEX - KODAQ 50	South Korea
    "KN": "KOR",  # KNX	KRX - KONEX Market	South Korea
    "KQ": "KOR",  # KOE	KOSDAQ	South Korea
    "KS": "KOR",  # KSC	Korea SE (Koscom)	South Korea
    "KW": "KWT",  # KUW	Kuwait SE	Kuwait
    "LK": "LAO",  # LSX	Lao Securities Exchange	Laos
    "BY": "LBN",  # BDB	Beirut SE	Lebanon
    "CM": "LKA",  # CSE	Colombo SE	Sri Lanka
    "VL": "LTU",  # VLX	Vilnus SE	Lithuania
    "VS": "LTU",  # Yahoo	Nasdaq OMX Vilnius	Lithuania
    "LU": "LUX",  # LUX	Luxembourg SE	Luxembourg
    "LUF": "LUX",  # RCT	Luxembourg Domiciled Funds	Luxembourg
    "RI": "LVA",  # RIX	Riga Stock Exchange	Latvia
    "RG": "LVA",  # Yahoo	Nasdaq OMX Riga	Latvia
    "CS": "MAR",  # CAS	Casablanca SE	Morocco
    "MX": "MEX",  # MEX	Mexico SE	Mexico
    "MKE": "MKD",  # MKE	Macedonia Stock Exchange	Macedonia
    "MT": "MLT",  # MLT	Malta SE	Malta
    "YG": "MMR",  # YSX	Yangon Stock Exchange	Myanmar
    "MOT": "MNE",  # MOT	Montenegro SE	Montenegro
    "MNE": "MNG",  # MGS	Mongolia Stock Exchange	Mongolia
    "MZ": "MUS",  # MAU	Mauritius SE	Mauritius
    "MV": "MWI",  # MLS	Malawi SE	Malawi
    "KF": "MYS",  # MDX	Bursa Msia Derivatives Ex	Malaysia
    "KL": "MYS",  # KLS	Bursa Msia Securities Ex	Malaysia
    "NM": "NAM",  # NSE	Namibian SE	Namibia
    "LG": "NGA",  # LAG	Nigeria Stock Exchange	Nigeria
    "APX": "NLD",  # APX	APX Netherlands	Netherlands
    "AS": "NLD",  # AEX	Euronext Amsterdam	Netherlands
    "E": "NLD",  # EOE/AEX	AEX-Options & Futures	Netherlands
    "NFF": "NOR",  # NFF	Norwegian Fund Broker Asstn	Norway
    "NP": "NOR",  # NWX	Nordpool Energy Exch Options	Norway
    "OL": "NOR",  # OSL	Oslo SE	Norway
    "OLT": "NOR",  # OSL	Oslo trades and broker info	Norway
    "NZ": "NZL",  # NZC	New Zealand SE	New Zealand
    "OM": "OMN",  # MUS	Muscat Sercuities Market	Oman
    "KA": "PAK",  # KAR	Karachi SE	Pakistan
    "LM": "PER",  # LMA	Lima SE	Peru
    "PS": "PHL",  # PHS	Philippine SE	Philippines
    "CT": "POL",  # CT1	Ceto OTC Regulated Market	Poland
    "WA": "POL",  # WSE	Warsaw SE	Poland
    "LS": "PRT",  # LIS	Euronext Lisbon	Portugal
    "PL": "PSE",  # PLS	Palestinian Securities Exch	Palestinian
    "QA": "QAT",  # DSM	Qatar Exchange	Qatar
    "BRQ": "ROU",  # BUH	RASDAQ Listed/RSQ Traded	Romania
    "BX": "ROU",  # BUH	Bucharest SE	Romania
    "NL": "ROU",  # BUH	Romanian Equities TNL	Romania
    "RO": "ROU",  # Yahoo	Bucharest Stock Exchange	Romania
    "MM": "RUS",  # MCX	MICEX Russia	Russia
    "MO": "RUS",  # MSE	Moscow SE	Russia
    "PC": "RUS",  # SPC	Saint-Petersburg Currency	Russia
    "PE": "RUS",  # SPS	Saint-Petersburg	Russia
    "RTS": "RUS",  # RTS	Russian Trading System	Russia
    "RW": "RWA",  # RSE	Rwanda Stock Exchange	Rwanda
    "SE": "SAE",  # SAU	Saudi SE	Arabia
    "SAU": "SAU",  # Yahoo	Saudi Stock Exchange (Tadawul)	Saudi Arabia
    "SI": "SGP",  # SES	SGX-ST Singapore	Singapore
    "BEL": "SRB",  # BEL	Belgrade Stock Exchange	Serbia
    "BV": "SVK",  # BRA	Bratislava SE	Slovakia
    "LJ": "SVN",  # LJU	Ljubljana SE	Slovenia
    "MANG": "SWE",  # BQT	BeQuoted	Sweden
    "NGM": "SWE",  # NGM	Nordic Growth Market	Sweden
    "SHBK": "SWE",  # BQT	Be Quoted	Sweden
    "ST": "SWE",  # STO	Stockholm Options	Sweden
    # TE" : "SWE", # AKT	Spotlight Stock Market	Sweden
    "DS": "SYR",  # DSX	Damascus Stock Exchange	Syria
    "BK": "THA",  # SET	Thailand SE	Thailand
    "FX": "THA",  # TFX	Thailand Futures	Thailand
    "TN": "TUN",  # TUN	Tunis SE	Tunisia
    "IS": "TUR",  # IST/TDE	Istanbul SE	Turkey
    "TR": "TUR",  # IST	ISE Settl and Custody Bank	Turkey
    "TC": "TWN",  # CBC	CBC,Taiwan	Taiwan
    "TE": "TWN",  # TEJ	Taiwan Economic Journal	Taiwan
    "TM": "TWN",  # TIM	Taiwan Futures Exchange	Taiwan
    "TW": "TWN",  # TAI	Taiwan SE	Taiwan
    "TWO": "TWN",  # TWO	ROC OTC SE	Taiwan
    "TZ": "TZA",  # DSS	Dar Es Salaam SE Ltd	Tanzania
    "PFT": "UKR",  # PFT	PFTS Stock Exchange	Ukraine
    "UAX": "UKR",  # UAX	Ukrainian Exchange	Ukraine
    "UG": "UKR",  # UGS	Uganda Stock Exchange	Ukraine
    "MN": "URY",  # MTV	MONTEVIDEO STOCK EXCHANGE	Uruguay
    "UE": "URY",  # UEX	URUGUAYAN ELECTRONIC EXCH	Uruguay
    "A": "USA",  # ASE	NYSE American(Equities)	United States
    "B": "USA",  # BOX	Boston Options Exchange	United States
    "C": "USA",  # LCC	NYSE National For NASDAQ LC	United States
    "DF": "USA",  # ADF	NASDAQ ADF	United States
    "DG": "USA",  # GCD	Direct Edge Holdings EDGX	United States
    "DY": "USA",  # GDA	Direct Edge Holdings	United States
    "EI": "USA",  # IEX	Investors Exchange	United States
    "K": "USA",  # PCQ	NYSE Arca Consolidated	United States
    "LTS": "USA",  # LTE	Long Term SE	United States
    "MW": "USA",  # MID	NYSE Chicago	United States
    "N": "USA",  # NYS	New York SE	United States
    "NB": "USA",  # NBN	Nasdaq Basic	United States
    "O": "USA",  # NSQ	Nasdaq Consolidated	United States
    "OB": "USA",  # OBB	OTC Bulletin Board	United States
    "OQ": "USA",  # NSM	NASDAQ Stock Market	United States
    "P": "USA",  # LCP	Pacific Exchange/ARCA	United States
    "PH": "USA",  # XPH	NASDAQ OMX PSX	United States
    "PK": "USA",  # PNK	OTC (Finra)	United States
    "TH": "USA",  # THM	Third Market Stock	United States
    "U": "USA",  # OPQ	OPRA NBBO Options	United States
    "W": "USA",  # WCB	Chicago Options	United States
    "X": "USA",  # PHO	Philadelphia Options	United States
    "Z": "USA",  # LCZ	BATS Trading For Nasdaq	United States
    "CBT": "USA",  # Yahoo	Chicago Board of Trade (CBOT)***	United States of America
    "CME": "USA",  # Yahoo	Chicago Mercantile Exchange (CME)***	United States of America
    "NYB": "USA",  # Yahoo	ICE Futures US	United States of America
    "CMX": "USA",  # Yahoo	New York Commodities Exchange (COMEX)***	United States of America
    "NYM": "USA",  # Yahoo	New York Mercantile Exchange (NYMEX)***	United States of America
    "II": "USA",  # ASQ	NYSE Amex Consolidated	USA
    "ZY": "USA",  # LCY	BATS Y Trading For Nasdaq OMX Global Market	USA
    "CR": "VEN",  # CCS	Caracas SE	Venezuela
    "HM": "VNM",  # HSX	Hochiminh S	Vietnam
    "HN": "VNM",  # HNX	Hanoi Stock Exchange	Vietnam
    "J": "ZAF",  # JNB/SFX	Johannesburg SE	South Africa
    "JO": "ZAF",  # Yahoo	Johannesburg Stock Exchange	South Africa
    "LZ": "ZMB",  # LUS	Lusaka Stock Exchange	Zambia
    "ZI": "ZWE",  # ZSE	Zimbabwe Stock Exchange	Zimbabwe
    # HNO" : "#N/A", # UPC	Unlisted Public Comp Mrkt	??
    # RCT" : "#N/A", # Reuters Contributed Exchange	code
    # BFC" : "#N/A", # TLX	Baltic Fund Market 	Estonia,Latvia,Lithuania
    # SIG" : "#N/A", # SIG	Sigma X	EUROPE
    # LP" : "#N/A", # LIP	Lipper	Global
    # TQ" : "#N/A", # TRQ	TURQUOISE	Kingdom
    # BD" : "#N/A", # BUR	Burgundy MTF	Nordic Region
    # FN" : "#N/A", # STO	OMX First North	Nordic Region
}
