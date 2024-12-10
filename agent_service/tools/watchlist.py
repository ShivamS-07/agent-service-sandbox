import inspect
import re
from typing import Dict, List, Optional, Tuple

from gbi_common_py_utils.utils.feature_flags import get_ld_flag

from agent_service.external.pa_svc_client import (
    get_all_stocks_in_all_watchlists,
    get_all_watchlists,
    get_watchlist_stocks,
)
from agent_service.GPT.constants import GPT4_O_MINI, NO_PROMPT
from agent_service.GPT.requests import GPT
from agent_service.io_types.stock import StockID
from agent_service.planner.errors import NotFoundError
from agent_service.tool import ToolArgs, ToolCategory, default_tool_registry, tool
from agent_service.tools.tool_log import tool_log
from agent_service.types import PlanRunContext
from agent_service.utils.gpt_logging import GptJobIdType, GptJobType, create_gpt_context
from agent_service.utils.pagerduty import pager_wrapper
from agent_service.utils.postgres import get_psql
from agent_service.utils.prefect import get_prefect_logger
from agent_service.utils.prompt_utils import Prompt
from agent_service.utils.tool_diff import (
    add_task_id_to_stocks_history,
    get_prev_run_info,
)


class GetUserWatchlistStocksInput(ToolArgs):
    watchlist_name: str
    watchlist_id: Optional[str] = None


class GetStocksForUserAllWatchlistsInput(ToolArgs):
    pass


WATCHLIST_ADD_STOCK_DIFF = "{company} was added to the watchlist: {watchlist}"
WATCHLIST_REMOVE_STOCK_DIFF = "{company} was removed from the watchlist: {watchlist}"


@tool(
    description=(
        "Given a watchlist name, and optionally a watchlist ID, this tool "
        "returns the list of stock identifiers that are inside "
        "the watchlist. It MUST be used when the client mentions 'watchlist' in the request. "
        "Watchlist ID should be included as well as name, but ONLY if the UUID of the "
        "watchlist is included in chat! "
        "E.g. 'My watchlist' (Watchlist ID: <some UUID>)."
    ),
    category=ToolCategory.PORTFOLIO,
    tool_registry=default_tool_registry(),
    is_visible=True,
)
async def get_user_watchlist_stocks(
    args: GetUserWatchlistStocksInput, context: PlanRunContext
) -> List[StockID]:
    # Use PA Service to get all accessible watchlists (including shared watchlists)

    logger = get_prefect_logger(__name__)

    resp = await get_all_watchlists(user_id=context.user_id)
    if not resp.watchlists:
        raise ValueError("User has no watchlists")

    # if there are watchlists with same name, we will return the one with the latest update
    all_watchlists = sorted(
        [w for w in resp.watchlists], key=lambda x: x.last_updated.ToDatetime(), reverse=True
    )
    watchlist_names = [watchlist.name for watchlist in all_watchlists]
    watchlist_ids = {watchlist.watchlist_id.id for watchlist in all_watchlists}

    name_to_id = {}
    for watchlist in all_watchlists:
        if watchlist.name not in name_to_id:  # only keep the first occurrence
            name_to_id[watchlist.name] = watchlist.watchlist_id.id

    if args.watchlist_id and args.watchlist_id in watchlist_ids:
        # ID passed in
        watchlist_id = args.watchlist_id
        watchlist_name = args.watchlist_name
    elif args.watchlist_name in name_to_id:
        # exact name match
        watchlist_id = name_to_id[args.watchlist_name]
        watchlist_name = args.watchlist_name
    elif not get_ld_flag("use-gpt-fallback-for-watchlist-search-term-matching", default=False):
        # Old logic for fallback, using just SQL
        sql = """
                SELECT watchlist_name
                FROM unnest(%(names)s::text[]) AS watchlist_name
                ORDER BY word_similarity(lower(watchlist_name), lower(%(target_name)s)) DESC
                LIMIT 1
                """
        rows = get_psql().generic_read(
            sql, {"names": watchlist_names, "target_name": args.watchlist_name}
        )
        watchlist_name = rows[0]["watchlist_name"]
        watchlist_id = name_to_id[rows[0]["watchlist_name"]]
    else:
        # Use SQL built-in function to find the best match
        sql = """
            SELECT watchlist_name
            FROM unnest(%(names)s::text[]) AS watchlist_name
            WHERE (
                (word_similarity(lower(watchlist_name), lower(%(target_name)s)) +
                 word_similarity(lower(%(target_name)s), lower(watchlist_name))) / 2
            ) >= %(threshold)s
            ORDER BY word_similarity(lower(watchlist_name), lower(%(target_name)s)) DESC
            LIMIT 1
        """
        threshold = 0.4

        rows = get_psql().generic_read(
            sql,
            {"names": watchlist_names, "target_name": args.watchlist_name, "threshold": threshold},
        )
        if rows:
            watchlist_name = rows[0]["watchlist_name"]
            watchlist_id = name_to_id[rows[0]["watchlist_name"]]
        else:
            # Let's try with GPT to find us a match
            result_name, result_id = await watchlist_match_by_gpt(
                context=context,
                watchlist_name=args.watchlist_name,
                watchlists_name_to_id=name_to_id,
            )
            if result_name and result_id:
                watchlist_name = result_name
                watchlist_id = result_id
            elif is_generic_watchlist_search(args.watchlist_name):
                # Final try - if a user is referring to a generic portfolio then return the most recently edited
                if len(all_watchlists) > 0:
                    # If no partial matches, return the user owned portfolio which was edited most recently
                    sorted_user_owned_watchlists = sorted(
                        all_watchlists,
                        key=lambda x: (
                            x.last_updated.seconds if x.last_updated else x.created_at.seconds
                        ),
                        reverse=True,
                    )
                    logger.info(
                        f"Generic watchlist requested, get most recent: {sorted_user_owned_watchlists[0]}"
                    )
                    watchlist_name = sorted_user_owned_watchlists[0].name
                    watchlist_id = sorted_user_owned_watchlists[0].watchlist_id.id
                else:
                    raise NotFoundError("User does not have access to any watchlists")
            else:
                raise NotFoundError(f"No watchlist found matching: '{args.watchlist_name}'")

    await tool_log(log=f"Found watchlist: <{watchlist_name}>", context=context)

    stock_list = await StockID.from_gbi_id_list(
        await get_watchlist_stocks(user_id=context.user_id, watchlist_id=watchlist_id)
    )

    try:  # since everything associated with diffing is optional, put in try/except
        # we need to add the task id to all runs, including the first one, so we can track changes
        # Update mode
        if context.task_id:
            stock_list = add_task_id_to_stocks_history(stock_list, context.task_id)
            if context.diff_info is not None:
                # 2nd arg is the name of the function we are in
                prev_run_info = await get_prev_run_info(context, "get_user_watchlist_stocks")
                if prev_run_info is not None:
                    prev_output: List[StockID] = prev_run_info.output  # type:ignore
                    # corner case here where S&P 500 change causes output to change, but not going to
                    # bother with it on first pass
                    curr_stock_set = set(stock_list)
                    prev_stock_set = set(prev_output)
                    added_stocks = curr_stock_set - prev_stock_set
                    removed_stocks = prev_stock_set - curr_stock_set
                    context.diff_info[context.task_id] = {
                        "added": {
                            added_stock: WATCHLIST_ADD_STOCK_DIFF.format(
                                company=added_stock.company_name, watchlist=watchlist_name
                            )
                            for added_stock in added_stocks
                        },
                        "removed": {
                            removed_stock: WATCHLIST_REMOVE_STOCK_DIFF.format(
                                company=removed_stock.company_name, watchlist=watchlist_name
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
            summary="Failed to get previous run info",
        )

    return stock_list


@tool(
    description=(
        "This tool returns the union of the stock identifiers in ALL watchlists of the user. "
        "It should ONLY be used when the user mentions 'all watchlists' or something similar. "
        "When a watchlist name is mentioned, you MUST use the tool 'get_user_watchlist_stocks'."
    ),
    category=ToolCategory.PORTFOLIO,
    tool_registry=default_tool_registry(),
    is_visible=True,
)
async def get_stocks_for_user_all_watchlists(
    args: GetStocksForUserAllWatchlistsInput, context: PlanRunContext
) -> List[StockID]:
    return await StockID.from_gbi_id_list(
        await get_all_stocks_in_all_watchlists(user_id=context.user_id)
    )


def is_generic_watchlist_search(search_str: Optional[str]) -> bool:
    """
    return true if search str generically refers to a portfolio an not a specific name
    """

    if not search_str:
        return True
    search_str = search_str.lower()
    search_str = re.sub(r"\bmy\b", " ", search_str)
    search_str = re.sub(r"\byour\b", " ", search_str)
    search_str = re.sub(r"\bour\b", " ", search_str)
    search_str = re.sub(r"\ba\b", " ", search_str)
    search_str = re.sub(r"\ban\b", " ", search_str)
    search_str = re.sub(r"\bany\b", " ", search_str)
    search_str = re.sub(r"\bthe\b", " ", search_str)
    search_str = re.sub(r"\bthat\b", " ", search_str)
    search_str = re.sub(r"\bthis\b", " ", search_str)
    search_str = re.sub(r"\bsome\b", " ", search_str)
    search_str = re.sub(r"\bwatchlists\b", " ", search_str)
    search_str = re.sub(r"\bwatchlist\b", " ", search_str)

    # remove any non-alphanumeric chars
    search_str = re.sub(r"\W+", "", search_str, flags=re.MULTILINE)
    search_str = search_str.replace("_", "")

    return not search_str


WATCHLIST_MATCH_PROMPT_STR = """
Given a watchlist name, find the best match from a list of watchlist options.
Consider possible typos, transposed letters, minor spelling errors, or related
meanings when finding the closest match.
Even if the match is not exact, return the watchlist that seems most relevant or
closest in meaning.
Only respond with "no_match" if no watchlist name reasonably aligns with the given name.
When returning "no_match," return it without any punctuation or anything.
Here are the watchlists to consider: {watchlists}.
Here is the text to potentially match to one of them: {watchlist_name}.
"""


async def watchlist_match_by_gpt(
    context: PlanRunContext,
    watchlist_name: str,
    watchlists_name_to_id: Dict[str, str],
) -> Tuple[Optional[str], Optional[str]]:
    """
    Match a specific "watchlist_name" mentioned by a prompt to the closest actual
    watchlist name object from the given dict of watchlists.
    If there is no connection to be made at all, just return None, None.

    If a match is found, return a two tuple which is the watchlist name, watchlist ID
    """
    if not watchlists_name_to_id:
        return None, None

    gpt_context = create_gpt_context(
        GptJobType.AGENT_TOOLS, context.agent_id, GptJobIdType.AGENT_ID
    )

    llm = GPT(context=gpt_context, model=GPT4_O_MINI)

    WATCHLIST_MATCH_PROMPT = Prompt(
        name="WATCHLIST_MATCH_PROMPT", template=WATCHLIST_MATCH_PROMPT_STR
    )

    watchlist_names = [name for name in watchlists_name_to_id.keys()]
    prompt = WATCHLIST_MATCH_PROMPT.format(
        watchlist_name=watchlist_name, watchlists=watchlist_names
    )

    result_str = await llm.do_chat_w_sys_prompt(
        main_prompt=prompt,
        sys_prompt=NO_PROMPT,
    )

    # First pass case sensitive
    for name in watchlist_names:
        if name == result_str:
            return name, watchlists_name_to_id[name]

    # Just in case GPT returned all lowercase, second pass case in-sensitive
    for name in watchlist_names:
        if name.lower() == result_str.lower():
            return name, watchlists_name_to_id[name]

    return None, None
