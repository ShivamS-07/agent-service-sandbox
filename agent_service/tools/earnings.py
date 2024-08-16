import asyncio
import datetime
import uuid
from collections import defaultdict
from typing import Any, Dict, List, Optional

from nlp_service_proto_v1.earnings_impacts_pb2 import EventInfo

from agent_service.external.feature_svc_client import get_earnings_releases_in_range
from agent_service.external.grpc_utils import timestamp_to_datetime
from agent_service.external.nlp_svc_client import (
    get_earnings_call_events,
    get_earnings_call_transcripts,
    get_latest_earnings_call_events,
)
from agent_service.io_type_utils import TableColumnType
from agent_service.io_types.dates import DateRange
from agent_service.io_types.stock import StockID
from agent_service.io_types.table import (
    StockTable,
    StockTableColumn,
    TableColumn,
    TableColumnMetadata,
)
from agent_service.io_types.text import (
    StockEarningsSummaryText,
    StockEarningsText,
    StockEarningsTranscriptText,
)
from agent_service.tool import ToolArgs, ToolCategory, ToolRegistry, tool
from agent_service.tools.dates import GetDateRangeInput, get_date_range
from agent_service.tools.LLM_analysis.tools import SummarizeTextInput, summarize_texts
from agent_service.tools.stocks import GetStockUniverseInput, get_stock_universe
from agent_service.tools.tool_log import tool_log
from agent_service.types import ChatContext, Message, PlanRunContext
from agent_service.utils.clickhouse import Clickhouse
from agent_service.utils.date_utils import get_now_utc, parse_date_str_in_utc
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
    user_id: str,
    stock_ids: List[StockID],
    start_date: Optional[datetime.date] = None,
    end_date: Optional[datetime.date] = None,
    allow_simple_generated_earnings: bool = False,
) -> Dict[StockID, List[StockEarningsText]]:
    db = get_psql()
    if allow_simple_generated_earnings:
        simple_generation_clause = ""
    else:
        simple_generation_clause = "AND fully_generated = TRUE"

    earning_summary_sql = f"""SELECT DISTINCT ON (gbi_id, year, quarter)
    summary_id::TEXT,
    summary,
    gbi_id,
    sources,
    year,
    quarter,
    created_timestamp
    FROM
        nlp_service.earnings_call_summaries
    WHERE
        gbi_id = ANY(%(gbi_ids)s)
        AND (status_msg = 'COMPLETE' OR status_msg IS NULL)
        {simple_generation_clause}
    ORDER BY
        gbi_id,
        year,
        quarter,
        created_timestamp DESC
    """

    rows = db.generic_read(earning_summary_sql, {"gbi_ids": [stock.gbi_id for stock in stock_ids]})
    by_stock_lookup = defaultdict(list)
    for row in rows:
        by_stock_lookup[row["gbi_id"]].append(row)

    # Get all earning events we expect to have for the even date range
    if start_date and end_date:
        earning_call_events = (
            await get_earnings_call_events(
                user_id, [stock.gbi_id for stock in stock_ids], start_date, end_date
            )
        ).earnings_event_info
    else:
        # If no date range then get the latest available earnings
        earning_call_events = (
            await get_latest_earnings_call_events(user_id, [stock.gbi_id for stock in stock_ids])
        ).earnings_event_info

    # Create a lookup for gbi_ids and the year-quarter earnings they should have along with the associated event
    all_earning_fiscal_quarters = defaultdict(set)
    earning_events_lookup: Dict[int, Any] = defaultdict(dict)
    for event in earning_call_events:
        all_earning_fiscal_quarters[event.gbi_id].add((event.year, event.quarter))
        earning_events_lookup[event.gbi_id][(event.year, event.quarter)] = event

    output: Dict[StockID, List[StockEarningsText]] = defaultdict(list)
    for stock_id in stock_ids:
        # Use this to track the year-quarter earnings with summaries for the given stock_id
        year_quarters_with_summaries = set()
        stock_output: List[StockEarningsText] = []
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
            year = row["sources"][0]["year"]
            quarter = row["sources"][0]["quarter"]
            publish_time = parse_date_str_in_utc(row["sources"][0]["publishing_time"])
            publish_date = publish_time.date()
            if (start_date and publish_date < start_date) or (end_date and publish_date > end_date):
                continue
            if not start_date and not end_date and len(stock_output) > 0:
                # If no start or end date were set, just return the most recent for each stock
                continue
            stock_output.append(
                StockEarningsSummaryText(
                    id=row["summary_id"],
                    stock_id=stock_id,
                    timestamp=publish_time,
                    year=year,
                    quarter=quarter,
                )
            )
            year_quarters_with_summaries.add((year, quarter))

        output[stock_id] = stock_output
        # Remove the year-quarters for the given stock_id for which earning summaries were generated
        all_earning_fiscal_quarters[stock_id.gbi_id] = (
            all_earning_fiscal_quarters[stock_id.gbi_id] - year_quarters_with_summaries
        )

    # Find the list of events for the set of year-quarters for each stock_id that did not have earning summaries
    events_without_summaries: List[EventInfo] = []
    event_id_to_stock_id_lookup: Dict[int, StockID] = {}
    gbi_id_stock_id_lookup = {stock_id.gbi_id: stock_id for stock_id in stock_ids}

    for gbi_id, earning_year_quarters in all_earning_fiscal_quarters.items():
        stock_id = gbi_id_stock_id_lookup[gbi_id]
        for year_quarter in earning_year_quarters:
            year = year_quarter[0]
            quarter = year_quarter[1]
            event = earning_events_lookup[gbi_id][(year, quarter)]

            event_id_to_stock_id_lookup[event.event_id] = stock_id
            events_without_summaries.append(event)

    # Get the transcripts for the events without summaries
    if len(events_without_summaries) > 0:
        finalized_output = await get_earnings_full_transcripts(
            user_id, stock_ids, events_without_summaries, event_id_to_stock_id_lookup, output
        )
    else:
        finalized_output = output

    return finalized_output


async def get_earnings_full_transcripts(
    user_id: str,
    stock_ids: List[StockID],
    events: List[EventInfo],
    event_id_to_stock_id_lookup: Dict[int, StockID],
    initial_stock_earnings_text_dict: Optional[Dict[StockID, List[StockEarningsText]]] = None,
) -> Dict[StockID, List[StockEarningsText]]:
    """
    This function will grab the full raw earnings transcript for a given earnings event by first checking if
    the event_id is present in our Clickhouse DB, if it is, it will use that data by referencing the entry's id
    in the database.

    If the event_id is not present in our DB we will hit an nlp_service endpoint to grab the transcript data and
    insert it into the db. Then referencing the new entry's id within the database.
    """
    earnings_transcript_sql = """
        SELECT id::TEXT as id, gbi_id, earnings_date, event_id
        FROM company_earnings.full_earning_transcripts
        WHERE gbi_id IN %(gbi_ids)s
    """
    if initial_stock_earnings_text_dict is None:
        stock_earnings_text_dict: Dict[StockID, List[StockEarningsText]] = defaultdict(list)
    else:
        stock_earnings_text_dict = initial_stock_earnings_text_dict

    ch = Clickhouse()
    transcript_query_result = await ch.generic_read(
        earnings_transcript_sql,
        params={
            "gbi_ids": [stock.gbi_id for stock in stock_ids],
        },
    )
    # Lookup for all available db data, the transcripts are not included to reduce the size of the call
    transcript_db_data_lookup = {row["event_id"]: row for row in transcript_query_result}
    # Track missing events not found within the db
    missing_events_in_db: List[EventInfo] = []

    for event in events:
        stock_id = event_id_to_stock_id_lookup[event.event_id]
        db_data = transcript_db_data_lookup.get(event.event_id)
        if db_data:
            stock_earnings_text_dict[stock_id].append(
                StockEarningsTranscriptText(
                    id=str(db_data["id"]),
                    stock_id=stock_id,
                    timestamp=db_data["earnings_date"],
                )
            )
        else:
            # Not in database, need to grab from nlp_service endpoint
            missing_events_in_db.append(event)

    # Hit an nlp_service endpoint to grab earnings that were not stored in the db
    transcripts_response = await get_earnings_call_transcripts(user_id, missing_events_in_db)

    # Create an event_id lookup for year, quarter (these are fiscal as they are taken direct from aiera)
    event_year_quarter_lookup = {
        event.event_id: {"year": event.year, "quarter": event.quarter}
        for event in missing_events_in_db
    }

    # Insert the missing entries and create StockEarningsText entries to map to them
    records_to_upload_to_db = []
    for transcript_data in transcripts_response.transcripts_data:
        stock_id = event_id_to_stock_id_lookup[transcript_data.event_id]
        event_id = transcript_data.event_id
        publish_time = timestamp_to_datetime(transcript_data.earnings_date)

        year = event_year_quarter_lookup[event_id]["year"]
        quarter = event_year_quarter_lookup[event_id]["quarter"]

        transcript_entry_id = uuid.uuid4()
        records_to_upload_to_db.append(
            {
                "id": transcript_entry_id,
                "gbi_id": stock_id.gbi_id,
                "event_id": event_id,
                "earnings_date": publish_time,
                "fiscal_year": year,
                "fiscal_quarter": quarter,
                "transcript": transcript_data.transcript,
            }
        )
        stock_earnings_text_dict[stock_id].append(
            StockEarningsTranscriptText(
                id=str(transcript_entry_id),
                stock_id=stock_id,
                timestamp=publish_time,
            )
        )

    if records_to_upload_to_db:
        await ch.multi_row_insert(
            table_name="company_earnings.full_earning_transcripts", rows=records_to_upload_to_db
        )
    return stock_earnings_text_dict


class GetEarningsCallDataInput(ToolArgs):
    stock_ids: List[StockID]
    date_range: Optional[DateRange] = None


@tool(
    description=(
        "This returns a list of all earnings call transcripts for one or more stocks "
        "that were published within the provided date range. This should only be used when "
        "a user explicitly asks for the full earnings transcript. Otherwise earnings data should "
        "be retrieved through the get_earnings_call_summaries tool. "
        "If no date range is provided, it defaults to the last quarter, containing the "
        "the full earnings call transcript for the most recent earnings call and what the clients "
        "are usually interested in unless they explicitly state otherwise. "
        "You may alternatively provide a date_range created by the get_n_width_date_range_near_date tool"
    ),
    category=ToolCategory.EARNINGS,
    tool_registry=ToolRegistry,
)
async def get_earnings_call_full_transcripts(
    args: GetEarningsCallDataInput, context: PlanRunContext
) -> List[StockEarningsText]:
    if args.date_range:
        start_date = args.date_range.start_date
        end_date = args.date_range.end_date
    else:
        start_date = None
        end_date = None

    # Get all earning events we expect to have for the even date range
    if start_date and end_date:
        earning_call_events = (
            await get_earnings_call_events(
                context.user_id, [stock.gbi_id for stock in args.stock_ids], start_date, end_date
            )
        ).earnings_event_info
    else:
        # If no date range then get the latest available earnings
        earning_call_events = (
            await get_latest_earnings_call_events(
                context.user_id, [stock.gbi_id for stock in args.stock_ids]
            )
        ).earnings_event_info

    gbi_id_stock_id_lookup = {stock_id.gbi_id: stock_id for stock_id in args.stock_ids}
    event_id_to_stock_id_lookup = {
        event.event_id: gbi_id_stock_id_lookup[event.gbi_id] for event in earning_call_events
    }
    topic_lookup = await get_earnings_full_transcripts(
        context.user_id, args.stock_ids, list(earning_call_events), event_id_to_stock_id_lookup
    )

    output: List[StockEarningsText] = []
    for topic_list in topic_lookup.values():
        output.extend(topic_list)
    if not output:
        raise Exception(
            "Did not get any earnings call transcripts for these stocks over the specified time period"
        )
    await tool_log(log=f"Found {len(output)} earnings call transcripts", context=context)
    return output


@tool(
    description=(
        "This returns a list of all earnings call summaries for one or more stocks "
        "that were published within the provided date range. "
        "If the client simply mentions `earnings calls`, you will get data using this tool unless "
        "the term `transcript` is specifically used, at which point you would use the transcript tool. "
        "Again, this tool is the default tool for getting earnings call information! "
        "If no date range is provided, it defaults to the last quarter, containing the "
        "the summary for the most recent earnings call and what the clients are usually interested "
        "in unless they explicitly state otherwise. "
    ),
    category=ToolCategory.EARNINGS,
    tool_registry=ToolRegistry,
)
async def get_earnings_call_summaries(
    args: GetEarningsCallDataInput, context: PlanRunContext
) -> List[StockEarningsText]:
    # if a date range obj was provided, fill in any missing dates
    if args.date_range:
        start_date = args.date_range.start_date
        end_date = args.date_range.end_date
    else:
        start_date = None
        end_date = None

    topic_lookup = await _get_earnings_summary_helper(
        context.user_id, args.stock_ids, start_date, end_date, True
    )
    output: List[StockEarningsText] = []
    for topic_list in topic_lookup.values():
        output.extend(topic_list)
    if not output:
        raise Exception(
            "Did not get any earnings call summaries for these stocks over the specified time period"
        )

    num_earning_call_summaries = len(
        [text for text in output if isinstance(text, StockEarningsSummaryText)]
    )
    num_earning_call_transcripts = len(
        [text for text in output if isinstance(text, StockEarningsTranscriptText)]
    )

    await tool_log(
        log=f"Found {num_earning_call_summaries} earnings call summaries", context=context
    )
    await tool_log(
        log=(
            f"Summaries unavailable for some companies, found {num_earning_call_transcripts} "
            "earning call transcripts instead"
        ),
        context=context,
    )

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
    args: GetEarningsCallDataInput, context: PlanRunContext
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
        GetEarningsCallDataInput(stock_ids=impacting_stocks, date_range=date_range), plan_context  # type: ignore
    )

    print(len(earnings_summaries))  # type: ignore

    summary = await summarize_texts(
        SummarizeTextInput(texts=earnings_summaries), plan_context  # type: ignore
    )  # Summarize the filtered news texts into a single summary
    print(summary)


if __name__ == "__main__":
    asyncio.run(main())
