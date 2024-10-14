from typing import List

from agent_service.external.pa_svc_client import (
    get_all_stocks_in_all_watchlists,
    get_all_watchlists,
    get_watchlist_stocks,
)
from agent_service.io_types.stock import StockID
from agent_service.tool import ToolArgs, ToolCategory, ToolRegistry, tool
from agent_service.types import PlanRunContext
from agent_service.utils.postgres import get_psql
from agent_service.utils.prefect import get_prefect_logger
from agent_service.utils.tool_diff import (
    add_task_id_to_stocks_history,
    get_prev_run_info,
)


class GetUserWatchlistStocksInput(ToolArgs):
    watchlist_name: str


class GetStocksForUserAllWatchlistsInput(ToolArgs):
    pass


WATCHLIST_ADD_STOCK_DIFF = "{company} was added to the watchlist: {watchlist}"
WATCHLIST_REMOVE_STOCK_DIFF = "{company} was removed from the watchlist: {watchlist}"


@tool(
    description=(
        "This function returns a list of stock identifiers that are inside a given watchlist name "
        "or mention (e.g. my watchlist). "
        "It MUST be used when the client mentions any 'watchlist' in the request. "
        "This function will try to match the given name with the watchlist names for that clients "
        "and return the closest match. "
    ),
    category=ToolCategory.USER,
    tool_registry=ToolRegistry,
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

    name_to_id = {}
    for watchlist in all_watchlists:
        if watchlist.name not in name_to_id:  # only keep the first occurrence
            name_to_id[watchlist.name] = watchlist.watchlist_id.id

    # exact match
    if args.watchlist_name in name_to_id:
        watchlist_id = name_to_id[args.watchlist_name]
        watchlist_name = args.watchlist_name
    else:
        # Use SQL built-in function to find the best match

        # TODO we should add a minimum match strength of ~0.2
        # this should be changed to avg of word_sim(left, right) + word_sim(right, left)
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

    stock_list = await StockID.from_gbi_id_list(
        await get_watchlist_stocks(user_id=context.user_id, watchlist_id=watchlist_id)
    )

    try:  # since everything associated with diffing is optional, put in try/except
        # we need to add the task id to all runs, including the first one, so we can track changes
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
        logger.warning(f"Error creating diff info from previous run: {e}")

    return stock_list


@tool(
    description="This function returns the list of stock identifiers in all watchlists of the user.",
    category=ToolCategory.USER,
    tool_registry=ToolRegistry,
    is_visible=True,
)
async def get_stocks_for_user_all_watchlists(
    args: GetStocksForUserAllWatchlistsInput, context: PlanRunContext
) -> List[StockID]:
    return await StockID.from_gbi_id_list(
        await get_all_stocks_in_all_watchlists(user_id=context.user_id)
    )
