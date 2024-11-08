import asyncio
from collections import defaultdict
from typing import Dict, List, Optional

import pandas as pd

from agent_service.external.nlp_svc_client import (
    get_earnings_peers_impacted_by_stocks,
    get_earnings_peers_impacting_stocks,
)
from agent_service.GPT.constants import GPT4_O
from agent_service.GPT.requests import GPT
from agent_service.io_type_utils import HistoryEntry, TableColumnType
from agent_service.io_types.dates import DateRange
from agent_service.io_types.stock import StockID
from agent_service.io_types.stock_groups import StockGroup, StockGroups
from agent_service.io_types.table import Table, TableColumnMetadata
from agent_service.io_types.text import EarningsPeersText, Text, TextCitation
from agent_service.planner.errors import EmptyInputError, NotFoundError
from agent_service.tool import ToolArgs, ToolCategory, ToolRegistry, tool
from agent_service.tools.LLM_analysis.tools import SummarizeTextInput, summarize_texts
from agent_service.tools.stocks import (
    StockIdentifierLookupInput,
    stock_identifier_lookup,
)
from agent_service.tools.tool_log import tool_log
from agent_service.types import ChatContext, Message, PlanRunContext
from agent_service.utils.async_utils import gather_with_concurrency
from agent_service.utils.date_utils import get_now_utc
from agent_service.utils.postgres import SyncBoostedPG, get_psql
from agent_service.utils.prefect import get_prefect_logger
from agent_service.utils.prompt_utils import Prompt
from agent_service.utils.tool_diff import get_prev_run_info

DELIMITER = "\n\n********************\n\n"
SEPARATOR = "###"
GET_PEER_GROUP_FOR_STOCK_MAIN_PROMPT = Prompt(
    name="GET_PEER_GROUP_FOR_STOCK_MAIN_PROMPT",
    template="Here is the given stock:\n {stock_str}\n"
    "Provided is also some information about the stock "
    "that may be useful to supplement your knowledge "
    "of the stock:\n"
    "{input_stock_info}",
)
GET_PEER_GROUP_FOR_STOCK_SYS_PROMPT = Prompt(
    name="GET_PEER_GROUP_FOR_STOCK_SYS_PROMPT",
    template="""
    Here is the chat context:
    {chat_context}
    Your task is to identify a list of stocks that belong to peer groups for a given stock.
    A peer group refers to stocks that operate in a similar market, industry, or sector, including competitors.
    Do not limit the amount of peers in the peer group, provide as many as possible.
    You may also be provided with a category to help you focus the peer group.
    The given stocks will each be presented on its own line, starting with a numerical stock identifier,
    the name of the stock, and optionally, the focus category, all delineated with the separator {separator}.
    Aggregate peer stocks by company and stock rather than individual products or divisions.\n
    For the format of the output:
    Return the given stock identifier and then the stock name, delineated by the separator {separator}.
    Return the peer stocks line by line, starting with:
    the Name of the Peer Stock
    the ISIN of the stock if it is publicly traded, else return False
    the Stock Symbol if it is publicly traded, else return False
    a 2 to 3 sentence justification of why you have chosen them to belong to the peer group
    all on the same line, all delineated with the separator {separator}.
    Output only plain text.
    Do not number the list.
    Do not output any additional justification or explanation.
    """,
)
VALIDATE_INPUT_STOCK_MAIN_PROMPT = Prompt(
    name="VALIDATE_INPUT_STOCK_MAIN_PROMPT",
    template="""Consider the following companies that make up a peer group:
    {peers_str}
    {category_str}
    Would you consider {input_stock_str} to be a major player that belongs in this peer group?
    In the first line of the output, return True or False without any punctuation.
    In the second line of the output, return a brief justification for your answer.
    """,
)
VALIDATE_PEER_STOCK_MAIN_PROMPT = Prompt(
    name="VALIDATE_PEER_STOCK_MAIN_PROMPT",
    template="""Consider the following company: {peer_str}
    Should {peer_str} be considered a competitor or peer company to {input_stock_str}{category_str}?
    Take into consideration the following reasoning:
    {justification}
    Return True or False without any punctuation.
    """,
)
VALIDATION_SYS_PROMPT = Prompt(
    name="VALIDATION_SYS_PROMPT",
    template="A peer group refers to stocks that operate in a similar market, "
    "industry, or sector, including competitors.\n"
    "Provided is also some information about the stocks "
    "that may be useful to supplement your knowledge "
    "of the stock:\n"
    "{extra_company_info}",
)

AFFECTED_LABEL = "Security"
AFFECTING_LABEL = "Potentially Influencial Security"
CONNECTION_LABEL = "Potential Influence"


class PeersForStockInput(ToolArgs):
    stock_ids: List[StockID]
    date_range: Optional[DateRange] = None


@tool(
    description="""
    This tool specificially identifies Earnings Peers for a given stock.
    This function takes in a list of StockIDs and a date_range and returns lists
    of PeerConnections that are impacted by the input stocks.
    The resulting PeerConnections contain the impacted stock (peers),
    the impacting stock (which will be from the input),
    and the summarized connection between the two.
    The connection is a list of PeerRelation objects, which contain the relation and connection
    between the two stocks and comes from Earnings Reports.
    This tool already has summarized the connection between the two stocks.
    Thus get_earnings_call_summaries does not need to be called in order to get the summarized connection.
    Important!!! This tool should only be used when the user specifically mentions earnings.
    """,
    category=ToolCategory.STOCK,
    tool_registry=ToolRegistry,
    enabled=False,
)
async def get_affected_peers(
    args: PeersForStockInput, context: PlanRunContext
) -> List[EarningsPeersText]:
    if not args.stock_ids:
        await tool_log(
            log="No peers found due to no input stocks",
            context=context,
        )
        return []

    impacting_peers = await get_earnings_peers_impacted_by_stocks(
        user_id=context.user_id, gbi_ids=[x.gbi_id for x in args.stock_ids]
    )

    peer_stocks: List[EarningsPeersText] = []

    for p in impacting_peers.peer_connections:
        gbi_id = (await StockID.from_gbi_id_list([p.gbi_id]))[0]
        peer_id = (await StockID.from_gbi_id_list([p.affected_gbi_id]))[0]

        report_date = p.earnings_date.ToDatetime().date()
        if args.date_range is not None:
            if args.date_range.start_date <= report_date <= args.date_range.end_date:
                peer_stocks.extend(
                    EarningsPeersText(
                        stock_id=peer_id,
                        affecting_stock_id=gbi_id,
                        val=c.connection,
                        history=[
                            HistoryEntry(citations=[TextCitation(source_text=Text(val=c.remark))])
                        ],
                        year=p.year,
                        quarter=p.quarter,
                    )
                    for c in p.connections
                )
        else:
            peer_stocks.extend(
                EarningsPeersText(
                    stock_id=peer_id,
                    affecting_stock_id=gbi_id,
                    val=c.connection,
                    history=[
                        HistoryEntry(citations=[TextCitation(source_text=Text(val=c.remark))])
                    ],
                    year=p.year,
                    quarter=p.quarter,
                )
                for c in p.connections
            )

    await tool_log(
        log=f"Found {len(peer_stocks)} peers for {len(args.stock_ids)} input stocks",
        context=context,
    )
    return peer_stocks


@tool(
    description="""
    This tool specificially identifies Earnings Peers for a given stock.
    This function takes in a list of StockIDs and a date_range and returns lists
    of PeerConnections that are impacted by the input stocks.
    The resulting PeerConnections contain the impacted stock (which will be from the input),
    the impacting stock (peers), and the connection between the two.
    The connection is a list of PeerRelation objects, which contain the relation and
    connection between the two stocks and comes from Earnings Reports.
    This tool already has summarized the connection between the two stocks.
    Thus get_earnings_call_summaries does not need to be called in order to get the summarized connection.
    Important!!! This tool should only be used when the user specifically mentions earnings.
    """,
    category=ToolCategory.STOCK,
    tool_registry=ToolRegistry,
    enabled=False,
)
async def get_affecting_peers(
    args: PeersForStockInput, context: PlanRunContext
) -> List[EarningsPeersText]:
    if not args.stock_ids:
        await tool_log(
            log="No peers found due to no input stocks",
            context=context,
        )

        return []

    impacting_peers = await get_earnings_peers_impacting_stocks(
        user_id=context.user_id, gbi_ids=[x.gbi_id for x in args.stock_ids]
    )

    peer_stocks: List[EarningsPeersText] = []

    for p in impacting_peers.peer_connections:
        gbi_id = (await StockID.from_gbi_id_list([p.affected_gbi_id]))[0]
        peer_id = (await StockID.from_gbi_id_list([p.gbi_id]))[0]

        report_date = p.earnings_date.ToDatetime().date()
        if args.date_range is not None:
            if args.date_range.start_date <= report_date <= args.date_range.end_date:
                peer_stocks.extend(
                    EarningsPeersText(
                        stock_id=gbi_id,
                        affecting_stock_id=peer_id,
                        val=c.connection,
                        history=[
                            HistoryEntry(citations=[TextCitation(source_text=Text(val=c.remark))])
                        ],
                        year=p.year,
                        quarter=p.quarter,
                    )
                    for c in p.connections
                )

        else:
            peer_stocks.extend(
                [
                    EarningsPeersText(
                        stock_id=gbi_id,
                        affecting_stock_id=peer_id,
                        val=c.connection,
                        history=[
                            HistoryEntry(citations=[TextCitation(source_text=Text(val=c.remark))])
                        ],
                        year=p.year,
                        quarter=p.quarter,
                    )
                    for c in p.connections
                ]
            )

    await tool_log(
        log=f"Found {len(peer_stocks)} peers for {len(args.stock_ids)} input stocks",
        context=context,
    )
    return peer_stocks


class PeersConnections(ToolArgs):
    connections: List[EarningsPeersText]


@tool(
    description="""
    This tool takes in a List of EarningsPeersText objects and returns the information
    in Table format. This table includes the affected stock, the affecting stock,
    and the connection between the two.
    """,
    category=ToolCategory.STOCK,
    tool_registry=ToolRegistry,
    enabled=False,
)
async def get_earnings_peers_table(args: PeersConnections, context: PlanRunContext) -> Table:
    columns: List[TableColumnMetadata] = []
    columns.append(TableColumnMetadata(label=AFFECTED_LABEL, col_type=TableColumnType.STOCK))
    columns.append(TableColumnMetadata(label=AFFECTING_LABEL, col_type=TableColumnType.STOCK))
    columns.append(TableColumnMetadata(label=CONNECTION_LABEL, col_type=TableColumnType.STRING))

    data = []
    for connection in args.connections:
        data.append(
            {
                AFFECTED_LABEL: connection.stock_id,
                AFFECTING_LABEL: connection.affecting_stock_id,
                CONNECTION_LABEL: connection.val,
            }
        )
    df = pd.DataFrame(data)
    return Table.from_df_and_cols(data=df, columns=columns)


# disabling this tool for now as it falls back to earnings peers
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
        # notice the key and values are reversed here
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


class GeneralPeersForStockInput(ToolArgs):
    stock_id: StockID
    max_num_peers: Optional[int] = None


@tool(
    description="""
    This function returns a list of peer companies for the input stock.
     Peers are related to the input stock as competitors as well as
     other actors in similar business or market areas as the input stock.
     You must use this tool when users ask for competitors or peers of a stock,
     unless they express specific interest in competitive analysis. Never, ever
     use this tool together with competitive analysis tools, instead use the
     product_or_service_filter tool. For example, if the user asks:
     'give me a news summary for each of the competitors of apple', you would
     'use this tool, but if the user ask to 'compare apple with its key
     competitors' you must NOT use this tool.
     If the client mentions a specific number of peers that they want, include
     that as max_num_peers, otherwise all peers will be returned
    """,
    category=ToolCategory.STOCK,
    tool_registry=ToolRegistry,
)
async def get_general_peers(
    args: GeneralPeersForStockInput, context: PlanRunContext
) -> List[StockID]:
    if not args.stock_id:
        await tool_log(
            log="No peers found due to no input stocks",
            context=context,
        )
        return await StockID.from_gbi_id_list([])

    llm = GPT(context=None, model=GPT4_O)
    stock_ids = await get_peer_group_for_stock(
        stock=args.stock_id,
        llm=llm,
        context=context,
        category=None,
    )

    await tool_log(log=f"Found {len(stock_ids)} peers for {args.stock_id.symbol}", context=context)

    if args.max_num_peers and len(stock_ids) > args.max_num_peers:
        stock_ids = stock_ids[: args.max_num_peers]
        await tool_log(log=f"Filtered to top {args.max_num_peers} peers", context=context)

    return stock_ids


async def get_peer_stock_id(
    peer_info: Dict[str, str], context: PlanRunContext
) -> Optional[StockID]:
    stock_id: Optional[StockID] = None
    for lookup_str in ["isin", "company_name", "symbol"]:
        if lookup_str in peer_info:
            try:
                stock_id = await stock_identifier_lookup(  # type: ignore
                    StockIdentifierLookupInput(
                        stock_name=peer_info[lookup_str],
                    ),
                    context=context,
                )
            except NotFoundError:
                pass

        if stock_id:
            break
    return stock_id


async def get_peer_group_for_stock(
    stock: StockID,
    llm: GPT,
    context: PlanRunContext,
    category: Optional[str] = None,
) -> List[StockID]:
    logger = get_prefect_logger(__name__)
    stock_str = str(stock.gbi_id) + SEPARATOR + stock.company_name
    if category:
        stock_str += SEPARATOR + category

    db = get_psql()
    input_stock_info, _ = db.get_short_company_description(stock.gbi_id)

    # initial prompt for peers
    initial_peers_gpt_resp = await llm.do_chat_w_sys_prompt(
        main_prompt=GET_PEER_GROUP_FOR_STOCK_MAIN_PROMPT.format(
            stock_str=stock_str,
            input_stock_info=input_stock_info,
        ),
        sys_prompt=GET_PEER_GROUP_FOR_STOCK_SYS_PROMPT.format(
            separator=SEPARATOR,
            chat_context=context.chat,
        ),
    )

    initial_peer_group = []
    initial_peers = initial_peers_gpt_resp.split("\n")
    # first line should be input stock: gbi_id, company name
    input_stock = initial_peers[0].split(SEPARATOR)
    if int(input_stock[0]) == stock.gbi_id:
        for initial_peer in initial_peers[1:]:
            # check line is not empty
            if initial_peer:
                try:
                    # company name, ISIN, stock symbol, justification
                    peer_lst = initial_peer.split(SEPARATOR)
                    # filter out non-public companies
                    if peer_lst[1].lower() != "false" or peer_lst[2].lower() != "false":
                        peer_obj = {
                            "company_name": peer_lst[0],
                            "isin": peer_lst[1],
                            "symbol": peer_lst[2],
                            "justification": peer_lst[3],
                        }
                        initial_peer_group.append(peer_obj)
                except IndexError:
                    logger.warning(f"Peers parsing failed for line: {initial_peer}, skipping")
                    continue

    # Dict[gbi_id, StockID]
    peer_stock_ids: Dict[int, StockID] = {}
    # Dict[gbi_id, justification]
    peer_justifications: Dict[int, str] = {}

    # we don't want the internal decision making of stock_identifier_lookup
    # to pollute the work log for peers tool
    no_tool_log_context = context.model_copy(update={"skip_db_commit": True})

    lookup_tasks = []
    for peer in initial_peer_group:
        lookup_tasks.append(get_peer_stock_id(peer, no_tool_log_context))

    lookup_result = await gather_with_concurrency(lookup_tasks, n=len(initial_peer_group))

    for peer, stock_id in zip(initial_peer_group, lookup_result):
        if stock_id and stock_id.gbi_id not in peer_stock_ids:  # prevent dupes
            peer_stock_ids[stock_id.gbi_id] = stock_id
            peer_justifications[stock_id.gbi_id] = peer.get("justification", "")

    if len(peer_stock_ids) == 0:
        raise ValueError(f"Could not find peers for stock {stock.company_name}")

    # validate that input stock belongs in peer group
    company_descriptions = {
        peer.gbi_id: db.get_short_company_description(peer.gbi_id)[0]
        for peer in peer_stock_ids.values()
    }

    company_info_for_llm = (
        "You are going to some peer group analysis for various companies,"
        "here is some supporting information regarding the companies:\n"
    )
    company_info_for_llm += DELIMITER.join(
        [description for description in company_descriptions.values() if description is not None]
    )

    validate_input_stock_gpt_resp = await llm.do_chat_w_sys_prompt(
        main_prompt=VALIDATE_INPUT_STOCK_MAIN_PROMPT.format(
            peers_str="\n".join([peer.company_name for peer in peer_stock_ids.values()]),
            input_stock_str=stock.company_name,
            category_str=(
                f"which are a part of a peer group within the industry or sector: {category}"
                if category
                else ""
            ),
        ),
        sys_prompt=VALIDATION_SYS_PROMPT.format(
            extra_company_info=company_info_for_llm,
        ),
    )
    # [0]: True or False, [1] justification
    validate_input_stock = validate_input_stock_gpt_resp.split("\n")
    if len(validate_input_stock) > 0 and validate_input_stock[0].lower() == "false":
        raise ValueError(
            f"{stock.company_name} does not belong in the"
            f"generated peer group: {validate_input_stock[1]}"
        )

    validated_peers: List[StockID] = []
    validate_tasks = []
    for peer_stock_id in peer_stock_ids.values():

        validate_tasks.append(
            llm.do_chat_w_sys_prompt(
                main_prompt=VALIDATE_PEER_STOCK_MAIN_PROMPT.format(
                    peer_str=peer_stock_id.company_name,
                    input_stock_str=stock.company_name,
                    justification=peer_justifications.get(peer_stock_id.gbi_id, "No reasoning.\n"),
                    category_str=(f" in the field of: {category}" if category else ""),
                ),
                sys_prompt=VALIDATION_SYS_PROMPT.format(
                    extra_company_info=company_descriptions[peer_stock_id.gbi_id],
                ),
            )
        )

    validate_results = await gather_with_concurrency(validate_tasks, n=len(peer_stock_ids))

    for peer_stock_id, validate_peer_stock_gpt_resp in zip(
        peer_stock_ids.values(), validate_results
    ):
        validate_peer_stock = validate_peer_stock_gpt_resp.split("\n")
        if len(validate_peer_stock) > 0 and validate_peer_stock[0].lower() == "false":
            logger.info(
                f"Removing {peer_stock_id.company_name} from peer group due to failing validation"
            )
            continue
        validated_peers.append(peer_stock_id)

    for i, peer in enumerate(validated_peers):
        if peer.gbi_id in peer_justifications:
            validated_peers[i] = peer.inject_history_entry(
                HistoryEntry(
                    explanation=peer_justifications[peer.gbi_id],
                    title=f"Connection to '{stock.company_name}'",
                )
            )

    return validated_peers


class PerStockGeneralPeersInput(ToolArgs):
    stock_ids: List[StockID]
    max_num_peers: Optional[int] = None


@tool(
    description="""
    This function returns a StockGroups object where each of its StockGroups
    consists of a list of peer companies for one.
    Peers are related to the input stock as competitors as well as
    other actors in similar business or market areas as the input stock.
    You should use this tool when client ask for competitors or peers of a list
    of stocks, For example, if the client asks:
    'show me a list of competitors for each of my watchlist stocks',
    you would use this tool. One of the most important uses of this tool is
    preparing lists of competitors for use to generate a related discussion
    using the per_stock_group_summarize_tool.
    You must never use this tool when the client is interested in a single group of competitors
    in those cases use the filter_by_product_or_service tool (for competitive analysis or other
    single product market situations) or the get_general_peers tool (when a general list of
    competitors is needed, potentially involving multiple different product markets).
    If the client mentions a specific number of peers that they want, include
    that as max_num_peers, otherwise all peers will be returned for each stock
    """,
    category=ToolCategory.STOCK,
    tool_registry=ToolRegistry,
)
async def per_stock_get_general_peers(
    args: PerStockGeneralPeersInput, context: PlanRunContext
) -> StockGroups:
    if not args.stock_ids:
        raise EmptyInputError("No peers found due to no input stocks")

    logger = get_prefect_logger(__name__)
    stock_groups = []

    wanted_stocks = args.stock_ids

    try:  # since everything associated with diffing is optional, put in try/except
        prev_run_info = await get_prev_run_info(context, "per_stock_get_general_peers")
        if prev_run_info is not None:
            old_competitor_lookup = {
                stock_group.ref_stock: stock_group
                for stock_group in prev_run_info.output.stock_groups  # type: ignore
            }
            new_wanted_stocks = []
            old_stock_groups = []
            for stock in args.stock_ids:
                # For now, we keep competitors constant within an agent. Since these competitors are
                # thought up by GPT, it doesn't make sense to update them except when the model updates,
                # most variation will be random
                if stock in old_competitor_lookup:
                    old_stock_groups.append(old_competitor_lookup[stock])
                else:
                    new_wanted_stocks.append(stock)
            wanted_stocks = new_wanted_stocks
            stock_groups = old_stock_groups

    except Exception as e:
        logger.warning(f"Error including stock ids from previous run: {e}")

    llm = GPT(context=None, model=GPT4_O)
    tasks = []
    for stock in wanted_stocks:
        tasks.append(
            get_peer_group_for_stock(
                stock=stock,
                llm=llm,
                context=context,
                category=None,
            )
        )

    results = await gather_with_concurrency(tasks, n=30)
    for target_stock, competitors in zip(wanted_stocks, results):
        stock_groups.append(
            StockGroup(
                name=f"{target_stock.company_name if target_stock.company_name else target_stock.symbol} Competitors",
                stocks=competitors,
                ref_stock=target_stock,
            )
        )
    return StockGroups(
        header="Target Stocks", stock_list_header="Competitors", stock_groups=stock_groups
    )


async def main() -> None:
    input_text = "Hello :)"
    user_message = Message(message=input_text, is_user_message=True, message_time=get_now_utc())
    chat_context = ChatContext(messages=[user_message])
    plan_context = PlanRunContext(
        agent_id="123",
        plan_id="123",
        user_id="123",
        plan_run_id="123",
        chat=chat_context,
        run_tasks_without_prefect=True,
        skip_db_commit=True,
    )
    output = await get_peer_group_for_stock(
        stock=StockID(gbi_id=714, symbol="AAPL", isin="AAPLISIN", company_name="Apple Inc."),
        llm=GPT(context=None, model=GPT4_O),
        context=plan_context,
        category="Mobile Devices",
    )
    for stock in output:
        print(stock.company_name)

    peers = await get_affected_peers(
        PeersForStockInput(
            stock_ids=[
                StockID(gbi_id=714, symbol="AAPL", isin="AAPLISIN", company_name="Apple Inc.")
            ]
        ),
        context=plan_context,
    )

    for peer in peers:  # type: ignore
        citation = await peer.to_rich_output(pg=SyncBoostedPG())  # type: ignore
        print(citation)

    output2 = await get_earnings_peers_table(PeersConnections(connections=peers), plan_context)  # type: ignore
    print(output2)

    output3 = await summarize_texts(SummarizeTextInput(texts=peers), plan_context)  # type: ignore
    print(output3)


if __name__ == "__main__":
    asyncio.run(main())
