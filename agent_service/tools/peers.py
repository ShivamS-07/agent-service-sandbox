from collections import defaultdict
from typing import List

from agent_service.external.nlp_svc_client import (
    get_earnings_peers_impacted_by_stocks,
    get_earnings_peers_impacting_stocks,
)
from agent_service.io_types.stock import StockID
from agent_service.tool import ToolArgs, ToolCategory, ToolRegistry, tool
from agent_service.tools.tool_log import tool_log
from agent_service.types import PlanRunContext

# from agent_service.utils.prefect import get_prefect_logger


class PeersForStockInput(ToolArgs):
    stock_ids: List[StockID]


@tool(
    description="""
This function returns a list of peer companies for the input stocks.
 Peers are related to the input stock as competitors as well as
 other actors in similar business or market areas as the input stocks.
""",
    category=ToolCategory.STOCK,
    tool_registry=ToolRegistry,
)
async def get_peers(args: PeersForStockInput, context: PlanRunContext) -> List[StockID]:
    """
    Returns a list of Peers for the input company
    """
    if not args.stock_ids:
        await tool_log(
            log="No peers found due to no input stocks",
            context=context,
        )

        return await StockID.from_gbi_id_list([])

    impacted_by_peers = await get_earnings_peers_impacted_by_stocks(
        user_id=context.user_id, gbi_ids=[x.gbi_id for x in args.stock_ids]
    )

    impacting_peers = await get_earnings_peers_impacting_stocks(
        user_id=context.user_id, gbi_ids=[x.gbi_id for x in args.stock_ids]
    )

    source_gbi_to_impacted_by_peers = defaultdict(set)
    source_gbi_to_impacting_peers = defaultdict(set)

    for p in impacted_by_peers.peer_connections:
        source_gbi_to_impacted_by_peers[p.gbi_id].add(p.affected_gbi_id)

    for p in impacting_peers.peer_connections:
        # notice the key and valeus are reversed here
        source_gbi_to_impacting_peers[p.affected_gbi_id].add(p.gbi_id)

    # for this first version we are defining competitors as stocks that are
    # peers that impact each other, so get the intersection
    bidirectional_peers = {}
    for k, v in source_gbi_to_impacted_by_peers.items():
        bidirectional_peers[k] = v.intersection(source_gbi_to_impacting_peers[k])

    peer_gbi_ids = []
    for k, v in bidirectional_peers.items():
        peer_gbi_ids.extend(list(v))

    peer_gbi_ids = list(set(peer_gbi_ids))

    if len(args.stock_ids) == 1:
        await tool_log(
            log=f"Found {len(peer_gbi_ids)} peers for {args.stock_ids[0].symbol}: "
            f"{args.stock_ids[0].company_name}",
            context=context,
        )
    else:
        await tool_log(
            log=f"Found {len(peer_gbi_ids)} peers for {len(args.stock_ids)} input stocks",
            context=context,
        )

    peer_stocks = await StockID.from_gbi_id_list(peer_gbi_ids)
    return peer_stocks
