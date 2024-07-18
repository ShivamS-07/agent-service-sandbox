import asyncio
import datetime
from collections import defaultdict
from typing import Dict, List, Optional

from agent_service.external.feature_svc_client import get_earnings_releases_in_range
from agent_service.io_type_utils import TableColumnType
from agent_service.io_types.dates import DateRange
from agent_service.io_types.stock import StockID
from agent_service.io_types.table import (
    StockTable,
    StockTableColumn,
    TableColumn,
    TableColumnMetadata,
)
from agent_service.io_types.text import StockEarningsSummaryText
from agent_service.tool import ToolArgs, ToolCategory, ToolRegistry, tool
from agent_service.tools.dates import GetDateRangeInput, get_date_range
from agent_service.tools.LLM_analysis.tools import SummarizeTextInput, summarize_texts
from agent_service.tools.stocks import GetStockUniverseInput, get_stock_universe
from agent_service.tools.tool_log import tool_log
from agent_service.types import ChatContext, Message, PlanRunContext
from agent_service.utils.date_utils import get_now_utc
from agent_service.utils.postgres import get_psql


class GetImpactingStocksInput(ToolArgs):
    impacted_stock_ids: List[StockID]


@tool(
    description=(
        "This returns a list of stocks corresponds to all the"
        " stocks which are likely to directly financially impact the provided stocks. "
    ),
    category=ToolCategory.EARNINGS,
    tool_registry=ToolRegistry,
)
async def get_impacting_stocks(
    args: GetImpactingStocksInput, context: PlanRunContext
) -> List[StockID]:
    db = get_psql()
    sql = """
    SELECT impacted_gbi_id, JSON_AGG(JSON_BUILD_ARRAY(gbi_id, reason)) AS impacting_stocks
    FROM nlp_service.earnings_relations
    WHERE impacted_gbi_id = ANY(%(gbi_ids)s)
    GROUP BY impacted_gbi_id
    """
    rows = db.generic_read(
        sql, {"gbi_ids": [stock_id.gbi_id for stock_id in args.impacted_stock_ids]}
    )
    impacting_lookup = {row["impacted_gbi_id"]: row["impacting_stocks"] for row in rows}
    output = set()
    for impacted_id in args.impacted_stock_ids:
        output.update(
            await StockID.from_gbi_id_list(
                [pair[0] for pair in impacting_lookup[impacted_id.gbi_id]]
            )
            if impacted_id.gbi_id in impacting_lookup
            else []
        )
    if len(output) == 0:
        raise Exception("Did not get any impacting stocks for these stocks")
    return list(output)


async def _get_earnings_summary_helper(
    stock_ids: List[StockID],
    start_date: Optional[datetime.date] = None,
    end_date: Optional[datetime.date] = None,
) -> Dict[StockID, List[StockEarningsSummaryText]]:
    db = get_psql()
    sql = """
        SELECT summary_id::TEXT, gbi_id, sources
        FROM nlp_service.earnings_call_summaries
        WHERE gbi_id = ANY(%(gbi_ids)s)
        AND (status_msg = 'COMPLETE' OR status_msg IS NULL)
        """

    rows = db.generic_read(sql, {"gbi_ids": [stock.gbi_id for stock in stock_ids]})
    by_stock_lookup = defaultdict(list)
    for row in rows:
        by_stock_lookup[row["gbi_id"]].append(row)

    output: Dict[StockID, List[StockEarningsSummaryText]] = {}
    for stock_id in stock_ids:
        stock_output: List[StockEarningsSummaryText] = []
        rows = by_stock_lookup.get(stock_id.gbi_id, [])
        # Sort most to least recent
        sorted_rows = sorted(
            rows,
            key=lambda row: datetime.datetime.fromisoformat(
                row["sources"][0]["publishing_time"]
            ).date(),
            reverse=True,
        )
        for row in sorted_rows:
            publish_date = datetime.datetime.fromisoformat(
                row["sources"][0]["publishing_time"]
            ).date()
            if (start_date and publish_date < start_date) or (end_date and publish_date > end_date):
                continue
            if not start_date and not end_date and len(stock_output) > 0:
                # If no start or end date were set, just return the most recent for each stock
                continue
            stock_output.append(StockEarningsSummaryText(id=row["summary_id"], stock_id=stock_id))
        output[stock_id] = stock_output
    return output


class GetEarningsCallSummariesInput(ToolArgs):
    stock_ids: List[StockID]
    date_range: Optional[DateRange] = None


@tool(
    description=(
        "This returns a list of all earnings call summaries for one or more stocks "
        " that were published within the provided date range. "
        " If no date range is provided, it defaults to the last quarter, containing the "
        " the summary for the most recent earnings call and what the clients are usually interested"
        " in unless they explicitly state otherwise."
        " You may alternatively provide a date_range created by the get_n_width_date_range_near_date tool"
    ),
    category=ToolCategory.EARNINGS,
    tool_registry=ToolRegistry,
)
async def get_earnings_call_summaries(
    args: GetEarningsCallSummariesInput, context: PlanRunContext
) -> List[StockEarningsSummaryText]:
    # if a date range obj was provided, fill in any missing dates
    if args.date_range:
        start_date = args.date_range.start_date
        end_date = args.date_range.end_date
    else:
        start_date = None
        end_date = None

    topic_lookup = await _get_earnings_summary_helper(args.stock_ids, start_date, end_date)
    output: List[StockEarningsSummaryText] = []
    for topic_list in topic_lookup.values():
        output.extend(topic_list)
    if not output:
        raise Exception(
            "Did not get any earnings call summaries for these stocks over the specified time period"
        )

    await tool_log(log=f"Found {len(output)} earnings call summaries", context=context)

    return output


class GetEarningsCallDatesInput(ToolArgs):
    stock_ids: List[StockID]
    date_range: Optional[DateRange] = None


@tool(
    description=(
        "This returns a stock table with stocks and dates for earnings calls that"
        " occurred within the specified date range. "
        " If no date range is provided, it defaults to fetching earnings calls occurring today."
        " You may alternatively provide a date_range created by the get_n_width_date_range_near_date tool."
        " Note that this does NOT return texts or summaries of the earnings calls, ONLY dates."
    ),
    category=ToolCategory.EARNINGS,
    tool_registry=ToolRegistry,
)
async def get_earnings_call_dates(
    args: GetEarningsCallSummariesInput, context: PlanRunContext
) -> StockTable:
    # if a date range obj was provided, fill in any missing dates
    if args.date_range:
        start_date = args.date_range.start_date
        end_date = args.date_range.end_date
    else:
        start_date = datetime.date.today()
        end_date = datetime.date.today()

    gbi_id_stock_map = {stock.gbi_id: stock for stock in args.stock_ids}
    gbi_dates_map = await get_earnings_releases_in_range(
        gbi_ids=list(gbi_id_stock_map.keys()),
        start_date=start_date,
        end_date=end_date,
        user_id=context.user_id,
    )
    rows = []
    for gbi_id, dates in gbi_dates_map.items():
        for date in dates:
            rows.append((gbi_id, date))

    return StockTable(
        columns=[
            StockTableColumn(data=[gbi_id_stock_map[row[0]] for row in rows]),
            TableColumn(
                data=[row[1] for row in rows],
                metadata=TableColumnMetadata(
                    label="Earnings Call Date", col_type=TableColumnType.DATE
                ),
            ),
        ]
    )


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
    date_range = await get_date_range(
        GetDateRangeInput(date_range_str="last month"), plan_context
    )  # Get the date for one month ago
    stock_ids = await get_stock_universe(
        GetStockUniverseInput(universe_name="TSX Composite"), plan_context
    )
    print(stock_ids)
    impacting_stocks = await get_impacting_stocks(
        GetImpactingStocksInput(impacted_stock_ids=stock_ids),  # type: ignore
        plan_context,
    )

    print(len(impacting_stocks))  # type: ignore

    earnings_summaries = await get_earnings_call_summaries(
        GetEarningsCallSummariesInput(stock_ids=impacting_stocks, date_range=date_range), plan_context  # type: ignore
    )

    print(len(earnings_summaries))  # type: ignore

    summary = await summarize_texts(
        SummarizeTextInput(texts=earnings_summaries), plan_context  # type: ignore
    )  # Summarize the filtered news texts into a single summary
    print(summary)


if __name__ == "__main__":
    asyncio.run(main())
