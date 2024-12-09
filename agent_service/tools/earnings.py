import asyncio
import datetime
from collections import defaultdict
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

from nlp_service_proto_v1.earnings_impacts_pb2 import EventInfo

from agent_service.external.feature_svc_client import get_earnings_releases_in_range
from agent_service.external.grpc_utils import timestamp_to_datetime
from agent_service.external.nlp_svc_client import (
    get_earnings_call_events,
    get_earnings_call_summaries_with_real_time_gen,
    get_earnings_call_transcripts,
    get_latest_earnings_call_events,
)
from agent_service.io_type_utils import TableColumnType
from agent_service.io_types.dates import DateRange
from agent_service.io_types.stock import StockID
from agent_service.io_types.table import (
    DateTableColumn,
    DatetimeTableColumn,
    StockTable,
    StockTableColumn,
    TableColumnMetadata,
)
from agent_service.io_types.text import (
    StockEarningsSummaryText,
    StockEarningsText,
    StockEarningsTranscriptText,
)
from agent_service.planner.errors import EmptyOutputError
from agent_service.tool import ToolArgs, ToolCategory, default_tool_registry, tool
from agent_service.tools.dates import GetDateRangeInput, get_date_range
from agent_service.tools.LLM_analysis.tools import SummarizeTextInput, summarize_texts
from agent_service.tools.stocks import GetStockUniverseInput, get_stock_universe
from agent_service.tools.tool_log import tool_log
from agent_service.types import ChatContext, Message, PlanRunContext
from agent_service.utils.async_db import get_async_db
from agent_service.utils.clickhouse import Clickhouse
from agent_service.utils.date_utils import get_now_utc, parse_date_str_in_utc
from agent_service.utils.postgres import get_psql
from agent_service.utils.prefect import get_prefect_logger

MAX_ALLOWED_EARNINGS_FOR_REAL_TIME_GEN = 0
CLICKHOUSE_WAIT_TIME = 10


class GetImpactingStocksInput(ToolArgs):
    impacted_stock_ids: List[StockID]


@tool(
    description=(
        "This returns a list of stocks corresponds to all the"
        " stocks which are likely to directly financially impact the provided stocks. "
    ),
    category=ToolCategory.EARNINGS,
    tool_registry=default_tool_registry(),
    enabled=False,
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
        raise EmptyOutputError("Did not get any impacting stocks for these stocks")
    return list(output)


async def _get_earnings_summary_helper(
    context: PlanRunContext,
    stock_ids: List[StockID],
    start_date: Optional[datetime.date] = None,
    end_date: Optional[datetime.date] = None,
    allow_simple_generated_earnings: bool = False,
) -> Dict[StockID, List[StockEarningsText]]:
    user_id = context.user_id

    async_db = get_async_db()

    gbi_id_to_stock_id_lookup = {x.gbi_id: x for x in stock_ids}
    gbi_ids = [stock.gbi_id for stock in stock_ids]
    company_id_sql = """
            SELECT gbi_id , spiq_company_id
            FROM spiq_security_mapping
            WHERE gbi_id = ANY(%(gbi_ids)s);
            """
    params = {"gbi_ids": gbi_ids}
    rows = await async_db.generic_read(company_id_sql, params)
    gbi_to_comp_id_lookup = {row["gbi_id"]: row["spiq_company_id"] for row in rows}

    if allow_simple_generated_earnings:
        simple_generation_clause = ""
    else:
        simple_generation_clause = "AND fully_generated = TRUE"

    no_empty_summary_clause = "AND summary != '{}'"
    earning_summary_sql = f"""WITH earning_summaries AS (
            SELECT
                ecs.summary_id::TEXT, ecs.gbi_id, ssm.spiq_company_id, ecs.summary, ecs.year, ecs.quarter,
                ecs.sources, ecs.is_complete, ecs.status_msg, ecs.created_timestamp
            FROM nlp_service.earnings_call_summaries ecs
            JOIN spiq_security_mapping ssm ON ecs.gbi_id = ssm.gbi_id
            WHERE ssm.spiq_company_id = ANY(%(company_ids)s)
                {no_empty_summary_clause}
                {simple_generation_clause}
            ORDER BY
                ecs.gbi_id, ecs.year DESC, ecs.quarter DESC, ecs.created_timestamp DESC
            )
            SELECT DISTINCT ON (spiq_company_id, year, quarter)
                es.summary_id, es.spiq_company_id, es.gbi_id, es.summary,
                es.year, es.quarter, es.sources, es.is_complete,
                es.status_msg, es.created_timestamp
            FROM earning_summaries es
            ORDER BY
                es.spiq_company_id, es.year DESC, es.quarter DESC, es.created_timestamp DESC;
    """

    company_ids = [gbi_to_comp_id_lookup[gbi_id] for gbi_id in gbi_ids]
    params = {"company_ids": company_ids}
    rows = await async_db.generic_read(earning_summary_sql, params)

    comp_id_to_gbi_mapping: Dict[int, List[int]] = defaultdict(list)
    for gbi_id, comp_id in gbi_to_comp_id_lookup.items():
        comp_id_to_gbi_mapping[comp_id].append(gbi_id)

    # Get all earning events we expect to have for the even date range
    if start_date and end_date:
        earning_call_events = list(
            (
                await get_earnings_call_events(
                    user_id, [stock.gbi_id for stock in stock_ids], start_date, end_date
                )
            ).earnings_event_info
        )
    elif context.as_of_date:
        # pretend we are in the past

        # note there is some slightly different logic further down with start/end are set,
        # we may need to use dif var name here to not trigger that if it is problematic
        end_date_past = context.as_of_date.date()

        # look back slightly more than a year
        start_date_past = end_date_past - datetime.timedelta(days=400)
        earning_call_events_ltm = (
            await get_earnings_call_events(
                user_id, [stock.gbi_id for stock in stock_ids], start_date_past, end_date_past
            )
        ).earnings_event_info

        # sort by date
        sorted_call_events = sorted(
            earning_call_events_ltm,
            key=lambda row: (timestamp_to_datetime(row.earnings_date), row.year, row.quarter),
            reverse=True,
        )

        # keep only the most recent earnings for each gbi_id
        earnings_map = {}
        for r in sorted_call_events:
            if r.gbi_id not in earnings_map:
                earnings_map[r.gbi_id] = r
        earning_call_events = list(earnings_map.values())
    else:
        # If no date range then get the latest available earnings
        earning_call_events = list(
            (
                await get_latest_earnings_call_events(
                    user_id, [stock.gbi_id for stock in stock_ids]
                )
            ).earnings_event_info
        )
    # Create a lookup for gbi_ids and the year-quarter earnings they should have along with the associated event
    all_earning_fiscal_quarters = defaultdict(set)
    earning_events_lookup: Dict[int, Any] = defaultdict(dict)
    for event in earning_call_events:
        all_earning_fiscal_quarters[event.gbi_id].add((event.year, event.quarter))
        earning_events_lookup[event.gbi_id][(event.year, event.quarter)] = event

    by_stock_lookup = defaultdict(list)
    for row in rows:
        for gbi_id in comp_id_to_gbi_mapping[row["spiq_company_id"]]:
            if (row["year"], row["quarter"]) in earning_events_lookup.get(gbi_id, {}).keys():
                by_stock_lookup[gbi_id].append(row)

    comp_id_to_earnings: Dict[int, List[StockEarningsText]] = defaultdict(list)
    for stock_id in stock_ids:
        # Use this to track the year-quarter earnings with summaries for the given stock_id
        year_quarters_with_summaries = set()
        stock_output: List[StockEarningsText] = []
        rows = by_stock_lookup.get(stock_id.gbi_id, [])
        # Sort most to least recent
        sorted_rows_for_stock = sorted(
            rows,
            key=lambda row: datetime.datetime.fromisoformat(
                row["sources"][0]["publishing_time"]
            ).date(),
            reverse=True,
        )
        for row in sorted_rows_for_stock:
            year = row["sources"][0]["year"]
            quarter = row["sources"][0]["quarter"]
            publish_time = parse_date_str_in_utc(row["sources"][0]["publishing_time"])
            publish_date = publish_time.date()

            if context.as_of_date and not start_date:
                # pretend we are in the past
                if publish_date > context.as_of_date.date():
                    continue
                if len(stock_output) > 0:
                    continue
            else:
                if (start_date and publish_date < start_date) or (
                    end_date and publish_date > end_date
                ):
                    continue
                if not start_date and not end_date and len(stock_output) > 0:
                    # If no start or end date were set, just return the most recent for each stock
                    continue
            sest = StockEarningsSummaryText(
                id=row["summary_id"],
                stock_id=stock_id,
                timestamp=publish_time,
                year=year,
                quarter=quarter,
            )
            stock_output.append(sest)
            year_quarters_with_summaries.add((year, quarter))

        comp_id_to_earnings[gbi_to_comp_id_lookup[stock_id.gbi_id]] = stock_output
        # Remove the year-quarters for the given stock_id for which earning summaries were generated
        all_earning_fiscal_quarters[stock_id.gbi_id] = (
            all_earning_fiscal_quarters[stock_id.gbi_id] - year_quarters_with_summaries
        )

    output: Dict[StockID, List[StockEarningsText]] = defaultdict(list)
    for gbi_id, comp_id in gbi_to_comp_id_lookup.items():
        output[gbi_id_to_stock_id_lookup[gbi_id]] = comp_id_to_earnings[comp_id]

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
    if len(events_without_summaries) == 0:
        finalized_output = output
    elif len(events_without_summaries) < MAX_ALLOWED_EARNINGS_FOR_REAL_TIME_GEN:
        await tool_log(
            log=(
                f"Earning summaries not available for {len(events_without_summaries)} earning calls, "
                "generating them now, this may take a couple minutes..."
            ),
            context=context,
        )
        finalized_output = await generate_missing_summaries(
            user_id, events_without_summaries, gbi_id_stock_id_lookup, output
        )
    else:
        await tool_log(
            log=(
                f"Some earning summaries are not processed ({len(events_without_summaries)}), "
                "using earning call transcripts inplace of the missing summaries for now..."
            ),
            context=context,
        )
        finalized_output = await get_earnings_full_transcripts(
            user_id, stock_ids, events_without_summaries, event_id_to_stock_id_lookup, output
        )
    num_companies_without_events = 0
    for stock_id, events in finalized_output.items():
        if len(events) == 0:
            num_companies_without_events += 1
            print(stock_id.gbi_id, stock_id.company_name)
    if num_companies_without_events > 0:
        await tool_log(
            log=(f"Earnings unavailable for {num_companies_without_events} companies"),
            context=context,
        )
    return finalized_output


async def _get_earning_transcript_lookup_from_ch(
    events: List[EventInfo], stock_ids: List[StockID]
) -> Dict[int, Dict[Tuple[int, int], Any]]:

    transcript_db_data_lookup: Dict[int, Dict[Tuple[int, int], Any]] = defaultdict(dict)
    if len(events) == 0:
        return transcript_db_data_lookup

    oldest_event = min(events, key=lambda x: (x.year, x.quarter))
    latest_event = max(events, key=lambda x: (x.year, x.quarter))

    earnings_transcript_sql = """
        SELECT id::TEXT as id, gbi_id, earnings_date, event_id, fiscal_year, fiscal_quarter
        FROM company_earnings.full_earning_transcripts
        WHERE gbi_id IN %(gbi_ids)s
        AND fiscal_year >= %(oldest_event_fiscal_year)s
        AND fiscal_year <= %(latest_event_fiscal_year)s
    """

    ch = Clickhouse()
    transcript_query_result = await ch.generic_read(
        earnings_transcript_sql,
        params={
            "gbi_ids": [stock.gbi_id for stock in stock_ids],
            "oldest_event_fiscal_year": oldest_event.year,
            "latest_event_fiscal_year": latest_event.year,
        },
    )
    # Lookup for all available db data, the transcripts are not included to reduce the size of the call
    for row in transcript_query_result:
        transcript_db_data_lookup[row["gbi_id"]][(row["fiscal_year"], row["fiscal_quarter"])] = row
    return transcript_db_data_lookup


async def generate_missing_summaries(
    user_id: str,
    events: List[EventInfo],
    gbi_id_stock_id_lookup: Dict[int, StockID],
    initial_stock_earnings_text_dict: Optional[Dict[StockID, List[StockEarningsText]]] = None,
) -> Dict[StockID, List[StockEarningsText]]:
    if initial_stock_earnings_text_dict is None:
        stock_earnings_text_dict: Dict[StockID, List[StockEarningsText]] = defaultdict(list)
    else:
        stock_earnings_text_dict = deepcopy(initial_stock_earnings_text_dict)

    resp = await get_earnings_call_summaries_with_real_time_gen(user_id, events)
    for summary_event in resp.earnings_summaries:
        stock_id = gbi_id_stock_id_lookup[summary_event.gbi_id]
        stock_text = StockEarningsSummaryText(
            id=summary_event.summary_id,
            stock_id=stock_id,
            timestamp=timestamp_to_datetime(summary_event.earnings_date),
            year=summary_event.year,
            quarter=summary_event.quarter,
        )
        stock_earnings_text_dict[stock_id].append(stock_text)
    return stock_earnings_text_dict


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
    logger = get_prefect_logger(__name__)

    if initial_stock_earnings_text_dict is None:
        stock_earnings_text_dict: Dict[StockID, List[StockEarningsText]] = defaultdict(list)
    else:
        stock_earnings_text_dict = deepcopy(initial_stock_earnings_text_dict)

    transcript_db_data_lookup = await _get_earning_transcript_lookup_from_ch(events, stock_ids)
    # Track missing events not found within the db
    missing_events_in_db: List[EventInfo] = []

    for event in events:
        stock_id = event_id_to_stock_id_lookup[event.event_id]
        transcript_data_for_gbi_id = transcript_db_data_lookup.get(stock_id.gbi_id, {})
        db_data = transcript_data_for_gbi_id.get((event.year, event.quarter))
        if db_data:
            if db_data["fiscal_year"] != 0:
                year = db_data["fiscal_year"]
                quarter = db_data["fiscal_quarter"]
            else:
                year = None
                quarter = None

            stock_earnings_text_dict[stock_id].append(
                StockEarningsTranscriptText(
                    id=str(db_data["id"]),
                    stock_id=stock_id,
                    timestamp=db_data["earnings_date"],
                    year=year,
                    quarter=quarter,
                )
            )
        else:
            # Not in database, need to grab from nlp_service endpoint
            missing_events_in_db.append(event)

    if len(missing_events_in_db) > 0:
        # Hit an nlp_service endpoint to grab earnings that were not stored in the db
        transcript_resp = await get_earnings_call_transcripts(user_id, missing_events_in_db)

        if len(transcript_resp.transcripts_data) != len(missing_events_in_db):
            logger.error("Unable to retrieve all requested earning transcripts")

        await asyncio.sleep(CLICKHOUSE_WAIT_TIME)
        transcript_db_data_lookup = await _get_earning_transcript_lookup_from_ch(
            missing_events_in_db, stock_ids
        )
        # Look for the previously missing events in the updated db data
        for event in missing_events_in_db:
            stock_id = event_id_to_stock_id_lookup[event.event_id]
            transcript_data_for_gbi_id = transcript_db_data_lookup.get(stock_id.gbi_id, {})
            db_data = transcript_data_for_gbi_id.get((event.year, event.quarter))
            if db_data:
                if db_data["fiscal_year"] != 0:
                    year = db_data["fiscal_year"]
                    quarter = db_data["fiscal_quarter"]
                else:
                    year = None
                    quarter = None

                stock_earnings_text_dict[stock_id].append(
                    StockEarningsTranscriptText(
                        id=str(db_data["id"]),
                        stock_id=stock_id,
                        timestamp=db_data["earnings_date"],
                        year=year,
                        quarter=quarter,
                    )
                )

    return stock_earnings_text_dict


class GetEarningsCallDataInput(ToolArgs):
    stock_ids: List[StockID]
    date_range: Optional[DateRange] = None


@tool(
    description="""This tool returns a list of all earnings call **transcripts** for one or more stocks \
that were published within the provided date range. This should only be used when \
a user explicitly asks for the full earnings transcript. Otherwise earnings data should \
be retrieved through the `get_earnings_call_summaries` tool. \
The user must explicitly ask for the full earning transcript, without summary to use this tool. \
Do not use this tool if the client mentions `summarizing`, in that case, \
use the `get_earnings_call_summaries` tool. \
You will be fired if you use this tool and then try to summarize it because \
that would indicate that the user wanted to summarize the output of this text. \
You should first use the tool `get_date_range` to create a DateRange object if there is a specific \
date range mentioned in the client's messages. If no date range is provided or can be inferred from \
the client's messages, you should set `date_range` to None, and it defaults to the last quarter under \
the hood, which you can assume is what the clients are usually interested in unless they explicitly \
state otherwise.\
You should not pass a date_range containing dates after todays date into this function. \
documents can only be found for dates in the past up to the present, including todays date. \
I repeat you will be FIRED if you try to find documents from the future!!! YOU MUST NEVER DO THAT!!!"
""",
    category=ToolCategory.EARNINGS,
    tool_registry=default_tool_registry(),
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

    # Currently we drop earning events from FMP as they don't have a simple id lookup (event_id = -1)
    event_id_to_stock_id_lookup = {
        event.event_id: gbi_id_stock_id_lookup[event.gbi_id]
        for event in earning_call_events
        if event.event_id != -1
    }
    earning_call_events_with_ids = [event for event in earning_call_events if event.event_id != -1]

    if len(earning_call_events_with_ids) < len(earning_call_events):
        await tool_log(
            f"Could not retrieve {len(earning_call_events) - len(earning_call_events_with_ids)} earning transcript(s)",
            context=context,
        )
    transcript_lookup = await get_earnings_full_transcripts(
        context.user_id,
        args.stock_ids,
        list(earning_call_events_with_ids),
        event_id_to_stock_id_lookup,
    )

    output: List[StockEarningsText] = []
    for transcript_list in transcript_lookup.values():
        output.extend(transcript_list)
    if not output:
        await tool_log(
            log="Did not get any earnings call transcripts for these stocks over the specified time period",
            context=context,
        )
    await tool_log(log=f"Found {len(output)} earnings call transcripts", context=context)
    return output


@tool(
    description="""This tool returns a list of all earnings call **summaries** for one or more stocks \
that were published within the provided date range. \
If the client simply mentions `earnings calls` or mentions `summarizing` earnings call information, \
you will get data using this tool unless \
the client asks for the full transcript, at which point you would use the transcript tool. \'
Again, this tool is the default tool for getting earnings call information!
If the client wants to summarize earnings call information, you should use this tool, you should
only use the transcript tool if the clients wants the entire transcript text.
Never, ever use get_default_text_data_for_stock as a substitute for this tool if the client says they want
to look 'earnings' data, you must use either this tool, or the transcript tool if the word transcript
is also used.
You should first use the tool `get_date_range` to create a DateRange object if there is a specific \
date range mentioned in the client's messages. If no date range is provided or can be inferred from \
the client's messages, you should set `date_range` to None, and it defaults to the last quarter under \
the hood, containing the summary for the most recent earnings call in which the clients are \
usually interested in unless they explicitly state otherwise. \
The output of this tool is not ordered by time. If the client needs to compare two earnings calls, you
must call this tool twice with two 3 month time ranges, do NOT do this in a single call
You should not pass a date_range containing dates after todays date into this function. \
documents can only be found for dates in the past up to the present, including todays date. \
I repeat you will be FIRED if you try to find documents from the future!!! YOU MUST NEVER DO THAT!!!
""",
    category=ToolCategory.EARNINGS,
    tool_registry=default_tool_registry(),
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
        context,
        args.stock_ids,
        start_date,
        end_date,
        True,
    )
    output: List[StockEarningsText] = []
    for topic_list in topic_lookup.values():
        output.extend(topic_list)
    if not output:
        await tool_log(
            log="Did not get any earnings call summaries for these stocks over the specified time period",
            context=context,
        )
        return []

    num_earning_call_summaries = len(
        [text for text in output if isinstance(text, StockEarningsSummaryText)]
    )
    num_earning_call_transcripts = len(
        [text for text in output if isinstance(text, StockEarningsTranscriptText)]
    )

    await tool_log(
        log=f"Found {num_earning_call_summaries} earnings call summaries", context=context
    )
    if num_earning_call_transcripts > 0:
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
    date_range: DateRange


@tool(
    description="""This tool returns a table for the dates of earnings calls for the provided stocks that \
occurred within the specified date range. For each earnings call it will also return a second column containing \
the datetime, which will also include the specific time of the earnings call, however this is not guaranteed, \
if a earnings call does not have this data, it will instead have None. You MUST always use the tool \
`get_date_range` first to create a DateRange object for the date range from the context of the client's messages \
before using this tool. For the upcoming earnings calls, the date range should be from today to today + 90 days. \
For the most recent quarter's earnings calls, the date range should be from today - 90 days to today. \
If the user asks to be notified when an earnings call happens, the date range should be today. You
must always include a date range. If there is no clear indication of the date range from the client's
messages, you should default to the upcoming earnings calls so the date range should be from today to
today + 90 days.
This tool only contains columns related to stock earnings date and times, it does NOT contain any other typical
information about stocks (e.g. performance, sectors, market cap, etc.). If you need to do further manipulation
of these stocks based on their statistics, you MUST extract the stock list and get additional data from the
get_statistics tool or other relevant tools.
""",
    category=ToolCategory.EARNINGS,
    tool_registry=default_tool_registry(),
)
async def get_earnings_call_dates(
    args: GetEarningsCallDatesInput, context: PlanRunContext
) -> StockTable:
    start_date = args.date_range.start_date
    end_date = args.date_range.end_date

    gbi_id_stock_map = {stock.gbi_id: stock for stock in args.stock_ids}
    gbi_dates_map = await get_earnings_releases_in_range(
        gbi_ids=list(gbi_id_stock_map.keys()),
        start_date=start_date,
        end_date=end_date,
        user_id=context.user_id,
    )

    rows = []
    for gbi_id, dates_data in gbi_dates_map.items():
        for date, date_with_time in dates_data:
            rows.append((gbi_id, date, date_with_time))

    default_date = datetime.date(datetime.MINYEAR, 1, 1)
    default_datetime = datetime.datetime(datetime.MINYEAR, 1, 1)
    rows.sort(
        key=lambda x: (x[1] if x[1] else default_date, x[2] if x[2] else default_datetime),
        reverse=True,
    )

    return StockTable(
        columns=[
            StockTableColumn(data=[gbi_id_stock_map[row[0]] for row in rows]),
            DateTableColumn(
                data=[row[1] for row in rows],
                metadata=TableColumnMetadata(
                    label="Earnings Call Date", col_type=TableColumnType.DATE
                ),
            ),
            DatetimeTableColumn(
                data=[row[2] for row in rows],
                metadata=TableColumnMetadata(
                    label="Earnings Call Date and Time", col_type=TableColumnType.DATETIME
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
