import asyncio
import datetime
from collections import defaultdict
from typing import Dict, List, Optional

from agent_service.io_types.stock import StockAlignedTextGroups, StockID
from agent_service.io_types.text import EarningsSummaryText, TextGroup
from agent_service.tool import ToolArgs, ToolCategory, ToolRegistry, tool
from agent_service.tools.dates import DateFromDateStrInput, get_date_from_date_str
from agent_service.tools.LLM_analysis import SummarizeTextInput, summarize_texts
from agent_service.tools.stocks import GetStockUniverseInput, get_stock_universe
from agent_service.types import ChatContext, Message, PlanRunContext
from agent_service.utils.date_utils import get_now_utc
from agent_service.utils.postgres import get_psql


class CollapseListOfListsOfStocksInput(ToolArgs):
    list_of_list_of_stock_ids: List[List[StockID]]


@tool(
    description=(
        "This function flattens a list of lists of stocks into a list of stocks. "
        "Use this function if you want to use all stocks returned from stock earnings impacts."
        "Do not use this function to build lists, use build_list or add_list"
    ),
    category=ToolCategory.EARNINGS,
    is_visible=False,
)
async def collapse_list_of_lists_of_stocks(
    args: CollapseListOfListsOfStocksInput, context: PlanRunContext
) -> List[StockID]:
    return [item for inner_list in args.list_of_list_of_stock_ids for item in inner_list]


class GetImpactingStocksInput(ToolArgs):
    impacted_stock_ids: List[StockID]


@tool(
    description=(
        "This returns a list of lists of stocks, each inner list corresponds to all the"
        " stocks which impact the corresponding stock. "
        "The length of the returned list of lists is the same as the input stock_ids"
    ),
    category=ToolCategory.EARNINGS,
    tool_registry=ToolRegistry,
)
async def get_impacting_stocks(
    args: GetImpactingStocksInput, context: PlanRunContext
) -> List[List[StockID]]:
    db = get_psql()
    sql = """
    SELECT impacted_gbi_id, JSON_AGG(JSON_BUILD_ARRAY(gbi_id, reason)) AS impacting_stocks
    FROM nlp_service.earnings_relations
    WHERE impacted_gbi_id = ANY(%(gbi_ids)s)
    GROUP BY impacted_gbi_id
    """
    rows = db.generic_read(sql, {"gbi_ids": args.impacted_stock_ids})
    impacting_lookup = {row["impacted_gbi_id"]: row["impacting_stocks"] for row in rows}
    output = []
    total = 0
    for impacted_id in args.impacted_stock_ids:
        output.append(
            await StockID.from_gbi_id_list(
                [pair[0] for pair in impacting_lookup[impacted_id.gbi_id]]
            )
            if impacted_id in impacting_lookup
            else []
        )
        total += len(output[-1])
    if total == 0:
        raise Exception("Did not get any impacting stocks for these stocks")
    return output


async def _get_earnings_summary_helper(
    stock_ids: List[StockID],
    start_date: Optional[datetime.date] = None,
    end_date: Optional[datetime.date] = None,
) -> Dict[StockID, List[EarningsSummaryText]]:
    db = get_psql()
    sql = """
        SELECT summary_id::TEXT, gbi_id, sources
        FROM nlp_service.earnings_call_summaries
        WHERE gbi_id = ANY(%(gbi_ids)s)
        """

    rows = db.generic_read(sql, {"gbi_ids": [stock.gbi_id for stock in stock_ids]})
    by_stock_lookup = defaultdict(list)
    for row in rows:
        by_stock_lookup[row["gbi_id"]].append(row)

    if not start_date:
        start_date = (get_now_utc() - datetime.timedelta(days=90)).date()
    if not end_date:
        # Add an extra day to be sure we don't miss anything with timezone weirdness
        end_date = get_now_utc().date() + datetime.timedelta(days=1)

    output: Dict[StockID, List[EarningsSummaryText]] = {}
    for stock_id in stock_ids:
        stock_output = []
        for row in by_stock_lookup.get(stock_id.gbi_id, []):
            publish_date = datetime.datetime.fromisoformat(
                row["sources"][0]["publishing_time"]
            ).date()
            if publish_date < start_date or publish_date > end_date:
                continue
            stock_output.append(EarningsSummaryText(id=row["summary_id"]))
        output[stock_id] = stock_output
    return output


class GetEarningsCallSummariesInput(ToolArgs):
    stock_ids: List[StockID]
    start_date: Optional[datetime.date] = None
    end_date: Optional[datetime.date] = None


@tool(
    description=(
        "This returns stock-aligned groups of earnings call summaries, each text group corresponds to all the"
        " earnings calls for the corresponding stock that were published between start_date and end_date. "
        " end_date defaults to today, start_date defaults to one quarter ago, which will return exactly"
        " the summary for the most recent earnings call and what the clients are usually interested"
        " in unless they explicitly state otherwise. "
        " You will use this function if you want to use the earnings calls on a per stock basis, for instance "
        " to filter multiple stocks based on the content of their earnings."
        "For example, if a user asks `give me a list of stocks in the S&P 500` whose last earnings call "
        " discussed mergers and acquisitions`, you would use this function to get the earnings calls`."
        " You should NOT use this function for getting data for questions about specific stocks."
    ),
    category=ToolCategory.EARNINGS,
    tool_registry=ToolRegistry,
)
async def get_stock_aligned_earnings_call_summaries(
    args: GetEarningsCallSummariesInput, context: PlanRunContext
) -> StockAlignedTextGroups:
    summary_lookup = await _get_earnings_summary_helper(
        args.stock_ids, args.start_date, args.end_date
    )
    output: Dict[StockID, TextGroup] = {}
    count = 0
    for stock_id, topic_list in summary_lookup.items():
        count += len(topic_list)
        output[stock_id] = TextGroup(val=topic_list)  # type: ignore
    if count == 0:
        raise Exception(
            "Did not get any earnings call summaries for these stocks over the specified time period"
        )
    return StockAlignedTextGroups(val=output)


@tool(
    description=(
        "This returns a list of all earnings call summaries for one or more stocks "
        " that were published between start_date and end_date. "
        " end_date defaults to today, start_date defaults to one quarter ago, which will return exactly"
        " the summary for the most recent earnings call and what the clients are usually interested"
        " in unless they explicitly state otherwise."
        " You will use this function if you want to use further summarize or display earnings calls summaries for"
        " one or a very small groups of stocks, but it is NOT appropriate for use for per stock applications"
        " like stock filtering since the alignment with stocks is not preserved. "
        " For example, if the user asked `give me a summary of all the latest earnings calls "
        " for the top-market cap tech stocks`, you would use this function."
        " You should also use it for answering a question about a specific stock related to"
        " information likely to be discussed in an earnings call."
    ),
    category=ToolCategory.EARNINGS,
    tool_registry=ToolRegistry,
)
async def get_earnings_call_summaries(
    args: GetEarningsCallSummariesInput, context: PlanRunContext
) -> List[EarningsSummaryText]:
    topic_lookup = await _get_earnings_summary_helper(
        args.stock_ids, args.start_date, args.end_date
    )
    output: List[EarningsSummaryText] = []
    for topic_list in topic_lookup.values():
        output.extend(topic_list)
    if not output:
        raise Exception(
            "Did not get any earnings call summaries for these stocks over the specified time period"
        )
    return output


async def main() -> None:
    input_text = "Need a summary of all the earnings calls from the last month of companies that might impact stocks in the TSX composite"  # noqa: E501
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
    start_date = await get_date_from_date_str(
        DateFromDateStrInput(date_str="1 month ago"), plan_context
    )  # Get the date for one month ago
    print(start_date)
    stock_ids = await get_stock_universe(
        GetStockUniverseInput(universe_name="TSX Composite"), plan_context
    )
    print(stock_ids)
    impacting_stocks_list = await get_impacting_stocks(
        GetImpactingStocksInput(impacted_stock_ids=stock_ids),  # type: ignore
        plan_context,
    )

    print(impacting_stocks_list)

    impacted_stocks = await collapse_list_of_lists_of_stocks(
        CollapseListOfListsOfStocksInput(list_of_list_of_stock_ids=impacting_stocks_list), plan_context  # type: ignore
    )
    print(len(impacted_stocks))  # type: ignore

    earnings_summaries = await get_earnings_call_summaries(
        GetEarningsCallSummariesInput(stock_ids=impacted_stocks, start_date=start_date), plan_context  # type: ignore
    )

    print(len(earnings_summaries))  # type: ignore

    summary = await summarize_texts(
        SummarizeTextInput(texts=earnings_summaries), plan_context  # type: ignore
    )  # Summarize the filtered news texts into a single summary
    print(summary)


if __name__ == "__main__":
    asyncio.run(main())
