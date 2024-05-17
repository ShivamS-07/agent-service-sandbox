import asyncio
import datetime
from collections import defaultdict
from typing import List, Optional

from agent_service.io_types import EarningsSummaryText
from agent_service.tool import ToolArgs, ToolCategory, ToolRegistry, tool
from agent_service.tools.dates import DateFromDateStrInput, get_date_from_date_str
from agent_service.tools.lists import CollapseListsInput, collapse_lists
from agent_service.tools.LLM_analysis import SummarizeTextInput, summarize_texts
from agent_service.tools.stock_universe import GetStockUniverseInput, get_stock_universe
from agent_service.types import ChatContext, Message, PlanRunContext
from agent_service.utils.date_utils import get_now_utc
from agent_service.utils.postgres import get_psql


class GetImpactingStocksInput(ToolArgs):
    impacted_stock_ids: List[int]


async def get_impacting_stocks(
    args: GetImpactingStocksInput, context: PlanRunContext
) -> List[List[int]]:

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
    for impacted_id in args.impacted_stock_ids:
        output.append(
            [pair[0] for pair in impacting_lookup[impacted_id]]
            if impacted_id in impacting_lookup
            else []
        )
    return output


class GetEarningsCallSummariesInput(ToolArgs):
    stock_ids: List[int]
    start_date: Optional[datetime.date] = None
    end_date: Optional[datetime.date] = None


@tool(
    description=(
        "This returns a list of lists of earnings call summary texts, each inner list corresponds to all the"
        " earnings calls for the corresponding stock that were published between start_date and end_date. "
        "start_date or end_date being None indicates the range is unbounded"
    ),
    category=ToolCategory.EARNINGS,
    tool_registry=ToolRegistry,
)
async def get_earnings_call_summaries(
    args: GetEarningsCallSummariesInput, context: PlanRunContext
) -> List[List[EarningsSummaryText]]:
    db = get_psql()
    sql = """
        SELECT summary_id::TEXT, gbi_id, created_timestamp
        FROM nlp_service.earnings_call_summaries
        WHERE gbi_id = ANY(%(gbi_ids)s)
        """

    rows = db.generic_read(sql, {"gbi_ids": args.stock_ids})
    by_stock_lookup = defaultdict(list)
    for row in rows:
        by_stock_lookup[row["gbi_id"]].append(row)

    output: List[List[EarningsSummaryText]] = []
    for stock_id in args.stock_ids:
        stock_output = []
        for row in by_stock_lookup.get(stock_id, []):
            if args.start_date is not None and row["created_timestamp"].date() < args.start_date:
                continue
            if args.end_date is not None and row["created_timestamp"].date() > args.end_date:
                continue
            stock_output.append(EarningsSummaryText(id=row["summary_id"]))
        output.append(stock_output)
    return output


async def main() -> None:
    input_text = "Need a summary of all the earnings calls from the last month of companies that might impact stocks in the TSX Composite"  # noqa: E501
    user_message = Message(message=input_text, is_user_message=True, message_time=get_now_utc())
    chat_context = ChatContext(messages=[user_message])
    plan_context = PlanRunContext(
        agent_id="123",
        plan_id="123",
        user_id="123",
        plan_run_id="123",
        chat=chat_context,
        run_tasks_without_prefect=True,
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

    impacted_stocks = await collapse_lists(
        CollapseListsInput(lists_of_lists=impacting_stocks_list), plan_context  # type: ignore
    )
    print(len(impacted_stocks))  # type: ignore

    earnings_summaries_list = await get_earnings_call_summaries(
        GetEarningsCallSummariesInput(stock_ids=impacted_stocks, start_date=start_date), plan_context  # type: ignore
    )

    earnings_summaries = await collapse_lists(
        CollapseListsInput(lists_of_lists=earnings_summaries_list), plan_context  # type: ignore
    )

    print(len(earnings_summaries))  # type: ignore

    summary = await summarize_texts(
        SummarizeTextInput(texts=earnings_summaries), plan_context  # type: ignore
    )  # Summarize the filtered news texts into a single summary
    print(summary)


if __name__ == "__main__":
    asyncio.run(main())
