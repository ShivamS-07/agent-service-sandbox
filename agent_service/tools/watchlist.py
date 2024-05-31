from typing import List

from agent_service.external.pa_svc_client import (
    get_all_stocks_in_all_watchlists,
    get_all_watchlists,
    get_watchlist_stocks,
)
from agent_service.io_types.misc import StockID
from agent_service.tool import ToolArgs, ToolCategory, ToolRegistry, tool
from agent_service.types import PlanRunContext
from agent_service.utils.postgres import get_psql


class GetUserWatchlistStocksInput(ToolArgs):
    watchlist_name: str


class GetStocksForUserAllWatchlistsInput(ToolArgs):
    pass


@tool(
    description="This function takes a string which refers to a watchlist name, and returns the"
    " list of stock identifiers in the best matched watchlist. A watchlist is a list of"
    " stocks that the user cares about.",
    category=ToolCategory.USER,
    tool_registry=ToolRegistry,
    is_visible=True,
)
async def get_user_watchlist_stocks(
    args: GetUserWatchlistStocksInput, context: PlanRunContext
) -> List[StockID]:
    # Use PA Service to get all accessible watchlists (including shared watchlists)
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

    if args.watchlist_name in name_to_id:
        return await StockID.from_gbi_id_list(
            await get_watchlist_stocks(
                user_id=context.user_id, watchlist_id=name_to_id[args.watchlist_name]
            )
        )

    # Use SQL built-in function to find the best match
    sql = """
        SELECT watchlist_name
        FROM unnest(%(names)s::text[]) AS watchlist_name
        ORDER BY word_similarity(lower(watchlist_name), lower(%(target_name)s)) DESC
        LIMIT 1
    """
    rows = get_psql().generic_read(
        sql, {"names": watchlist_names, "target_name": args.watchlist_name}
    )
    watchlist_id = name_to_id[rows[0]["watchlist_name"]]

    return await StockID.from_gbi_id_list(
        await get_watchlist_stocks(user_id=context.user_id, watchlist_id=watchlist_id)
    )


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
