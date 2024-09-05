import datetime
import json
from collections import defaultdict
from heapq import heappop, heappush
from typing import Dict, List, Optional, Tuple

from rapidfuzz.fuzz import ratio

from agent_service.GPT.constants import GPT4_O_MINI
from agent_service.GPT.requests import GPT
from agent_service.io_type_utils import ComplexIOBase, io_type
from agent_service.io_types.dates import DateRange
from agent_service.io_types.output import Output
from agent_service.io_types.stock import StockID
from agent_service.io_types.text import (
    StockOtherSecFilingText,
    StockSecFilingText,
    Text,
)
from agent_service.planner.errors import EmptyOutputError
from agent_service.tool import ToolArgs, ToolCategory, ToolRegistry, tool
from agent_service.tools.tool_log import tool_log
from agent_service.types import PlanRunContext
from agent_service.utils.boosted_pg import BoostedPG
from agent_service.utils.date_utils import parse_date_str_in_utc, timezoneify
from agent_service.utils.prefect import get_prefect_logger
from agent_service.utils.prompt_utils import Prompt
from agent_service.utils.sec.constants import (
    FILE_10K,
    FILE_10Q,
    FILED_TIMESTAMP,
    FORM_TYPE,
)
from agent_service.utils.sec.sec_api import SecFiling
from agent_service.utils.sec.supported_types import SUPPORTED_TYPE_MAPPING
from agent_service.utils.string_utils import repair_json_if_needed


async def get_sec_filings_helper(
    stock_ids: List[StockID], start_date: Optional[datetime.date], end_date: Optional[datetime.date]
) -> Dict[StockID, List[StockSecFilingText]]:
    stock_filing_map = defaultdict(list)

    gbi_id_to_stock_id = {stock.gbi_id: stock for stock in stock_ids}

    filing_gbi_pairs, filing_to_db_id = await SecFiling.get_filings(
        gbi_ids=list(gbi_id_to_stock_id.keys()),
        form_types=[FILE_10K, FILE_10Q],
        start_date=start_date,
        end_date=end_date,
    )

    for filing_str, gbi_id in filing_gbi_pairs:
        stock_id = gbi_id_to_stock_id[gbi_id]
        db_id = filing_to_db_id.get(filing_str, None)
        filing_json = json.loads(filing_str)
        timestamp = parse_date_str_in_utc(filing_json[FILED_TIMESTAMP])
        stock_filing_map[stock_id].append(
            StockSecFilingText(
                id=filing_str,
                stock_id=stock_id,
                db_id=db_id,
                timestamp=timestamp,
                form_type=filing_json.get(FORM_TYPE),
            )
        )

    return stock_filing_map


class GetSecFilingsInput(ToolArgs):
    stock_ids: List[StockID]
    date_range: Optional[DateRange] = None


@tool(
    description="Given a list of stock ID's, return a list of 10-K/10-Q SEC filings for the stocks. "
    "Specifically, this includes the management and risk factors sections of the 10-K and 10-Q documents "
    "which provides detailed, up-to-date information about company operations and financial status "
    "for the previous quarter (10-Q) or year (10-K) for any US-listed companies. "
    "This tool should be used by default when users just ask for SEC filings, or when they explicitly ask "
    "for `10-K` or `10-Q` filings."
    "It is especially useful for finding current information about less well-known "
    "companies which may have little or no news for a given period. "
    "Any documents published within the date_range are included. Date_range will "
    "default to the last quarter, which includes the latest SEC filing.",
    category=ToolCategory.SEC_FILINGS,
    tool_registry=ToolRegistry,
    store_output=False,
    enabled=True,
)
async def get_10k_10q_sec_filings(
    args: GetSecFilingsInput, context: PlanRunContext
) -> List[StockSecFilingText]:
    if args.date_range:
        start_date = args.date_range.start_date
        end_date = args.date_range.end_date
    else:
        start_date = None
        end_date = None

    await tool_log(
        "Retrieving 10-K, 10-Q SEC filings for"
        + (
            f" {args.stock_ids[0].symbol}"
            if len(args.stock_ids) == 1
            else f" {len(args.stock_ids)} stocks"
        )
        + (
            f" from {start_date.isoformat()} to {end_date.isoformat()}"
            if start_date and end_date
            else " in the last 100 days"
        ),
        context=context,
    )
    stock_filing_map = await get_sec_filings_helper(args.stock_ids, start_date, end_date)
    all_filings: List[StockSecFilingText] = []
    for filings in stock_filing_map.values():
        all_filings.extend(filings)

    if len(all_filings) == 0:
        raise EmptyOutputError("No filings were retrieved for these stocks over this time period")

    filing_list = [
        f"{filing.to_citation_title()} ({filing.form_type}) "
        + (f"- {filing.timestamp.strftime('%Y-%m-%d')}" if filing.timestamp else "")
        for filing in all_filings
    ]
    await tool_log(
        log=f"Found {len(all_filings)} filing(s).",
        context=context,
        associated_data=filing_list,
    )

    return all_filings


async def get_other_sec_filings_helper(
    stock_ids: List[StockID],
    form_types: List[str],
    start_date: Optional[datetime.date],
    end_date: Optional[datetime.date],
) -> Dict[StockID, List[StockOtherSecFilingText]]:
    gbi_id_to_stock_id = {stock.gbi_id: stock for stock in stock_ids}
    filing_gbi_pairs, filing_to_db_id = await SecFiling.get_filings(
        gbi_ids=list(gbi_id_to_stock_id.keys()),
        form_types=form_types,
        start_date=start_date,
        end_date=end_date,
    )

    stock_filing_map = defaultdict(list)
    for filing_str, gbi_id in filing_gbi_pairs:
        stock_id = gbi_id_to_stock_id[gbi_id]
        db_id = filing_to_db_id.get(filing_str, None)
        filing_json = json.loads(filing_str)
        timestamp = timezoneify(datetime.datetime.fromisoformat(filing_json[FILED_TIMESTAMP]))
        stock_filing_map[stock_id].append(
            StockOtherSecFilingText(
                id=filing_str,
                stock_id=stock_id,
                db_id=db_id,
                timestamp=timestamp,
                form_type=filing_json.get(FORM_TYPE),
            )
        )

    if not stock_filing_map:
        raise EmptyOutputError("No filings were found.")

    return stock_filing_map


class GetOtherSecFilingsInput(ToolArgs):
    stock_ids: List[StockID]
    form_types: List[str]
    start_date: Optional[datetime.date] = None
    end_date: Optional[datetime.date] = None
    date_range: Optional[DateRange] = None


@tool(
    description="This function was renamed to get_sec_filings_with_type",
    category=ToolCategory.SEC_FILINGS,
    tool_registry=ToolRegistry,
    store_output=False,
    enabled=False,
)
async def get_non_10k_10q_sec_filings(
    args: GetOtherSecFilingsInput, context: PlanRunContext
) -> List[StockSecFilingText]:
    form_types = [SecFilingType(name=form_type) for form_type in args.form_types]
    return await get_sec_filings_with_type(
        GetSecFilingsWithTypeInput(
            stock_ids=args.stock_ids,
            form_types=form_types,
            date_range=args.date_range,
        ),
        context=context,
    )  # type: ignore


FIND_SEC_FILING_TYPES_SYS_PROMPT = Prompt(
    name="FIND_SEC_FILING_TYPES_SYS_PROMPT",
    template="""
    You will be given a search term or question or query that the user would like to find within SEC filings.
    Your task is to identify the relevant SEC filing types that may contain the answer.
    You will be provided with a JSON object of all the SEC filing types and a short description of the document.
    Output a JSON list of the relevant SEC filing types and a short reasoning.
    Return in the format: {{"reason": "", "sec_filing_name": ""}}
    DO NOT RETURN ANY ADDITIONAL EXPLANATION OR JUSTIFICATION
    """,
)

FIND_SEC_FILING_TYPES_MAIN_PROMPT = Prompt(
    name="FIND_SEC_FILING_TYPES_MAIN_PROMPT",
    template="""
    Here are the SEC filing types in JSON format:
    ---
    {supported_sec_filing_types}
    ---
    Here is the search term or question that the user is looking for: {search_term}
    """,
)


@io_type
class SecFilingType(ComplexIOBase):
    name: str

    async def to_rich_output(self, pg: BoostedPG, title: str = "") -> Output:
        t: Text = Text(
            val=self.name,
        )
        return await t.to_rich_output(pg=pg, title=title)


class SecFilingsTypeLookupInput(ToolArgs):
    search_term: str
    form_types: Optional[List[str]] = None


@tool(
    description="Given a search term or question that relates to information that can"
    " be found in SEC filings, return a list of SEC filing types that"
    " we should retrieve to answer the question or find the requested information."
    " Specifically, the output from this tool should be fed into the `get_all_sec_filings` tool"
    " in its `form_types` parameter, as it will help that tool retrieve the correct"
    " SEC filings."
    " This tool should only be called if the user is looking for something specific within SEC filings"
    " and does not exactly tell us which SEC filings to retrieve."
    " As an example, if a user asked `Describe the corporate structure for Apple`,"
    " you would include `Corporate Structure` in the search_term."
    " If the user specifically requests certain SEC filings to be included in the query, this"
    " list of SEC filing types must be passed to `form_types` to force the tool to"
    " also include those filing types in its output."
    " As an example, if a user asked `Please find IPO details for Apple in S-1`,"
    " you must include `S-1` in the form_types parameter",
    category=ToolCategory.SEC_FILINGS,
    tool_registry=ToolRegistry,
    store_output=False,
    enabled=True,
)
async def sec_filings_type_lookup(
    args: SecFilingsTypeLookupInput, context: PlanRunContext
) -> List[SecFilingType]:
    logger = get_prefect_logger(__name__)

    sec_filing_types = set()
    if args.form_types and len(args.form_types) > 0:
        for form_type in args.form_types:
            if form_type in SUPPORTED_TYPE_MAPPING:
                sec_filing_types.add(form_type)
                continue
            # fuzzy matching of form type using a max heap, inefficient but should almost never run
            ratios_heap: List[Tuple[float, str]] = []
            for supported_type in SUPPORTED_TYPE_MAPPING.keys():
                heappush(
                    ratios_heap,
                    (-1 * ratio(form_type.upper(), supported_type.upper()), supported_type),
                )
            sec_filing_types.add(heappop(ratios_heap)[1])

    if args.search_term and args.search_term != "":
        llm = GPT(context=None, model=GPT4_O_MINI)

        sec_filing_types_resp = await llm.do_chat_w_sys_prompt(
            sys_prompt=FIND_SEC_FILING_TYPES_SYS_PROMPT.format(),
            main_prompt=FIND_SEC_FILING_TYPES_MAIN_PROMPT.format(
                supported_sec_filing_types=json.dumps(SUPPORTED_TYPE_MAPPING, indent=4),
                search_term=args.search_term,
            ),
        )
        logger.info(f"{args.search_term}: {sec_filing_types_resp}")
        sec_filing_types_dict = json.loads(repair_json_if_needed(sec_filing_types_resp))

        for sec_filing_type in sec_filing_types_dict:
            sec_filing_types.add(sec_filing_type["sec_filing_name"])

    return (
        [SecFilingType(name=filing_type) for filing_type in sec_filing_types]
        if len(sec_filing_types) > 0
        else [SecFilingType(name=FILE_10K), SecFilingType(name=FILE_10K)]
    )


class GetSecFilingsWithTypeInput(ToolArgs):
    stock_ids: List[StockID]
    form_types: List[SecFilingType]
    date_range: Optional[DateRange] = None


@tool(
    description="Given a list of stock ID's, return a list of requested SEC filings for the stocks."
    " This tool should be used after `sec_filings_type_lookup` which will determine the SEC filing types"
    " we need to retrieve. This tool should never be called by itself."
    " This tool should not be called if we only want to retrieve general information from SEC filings,"
    " instead the `get_10k_10q_sec_filings` tool should be used."
    " Any documents published between start_date and end_date will be included, if the end_date is"
    " excluded it is assumed to include documents up to today, if start_date is not"
    " included, the start date is a quarter ago, which includes only the latest SEC filing.",
    category=ToolCategory.SEC_FILINGS,
    tool_registry=ToolRegistry,
    store_output=False,
    enabled=True,
)
async def get_sec_filings_with_type(
    args: GetSecFilingsWithTypeInput, context: PlanRunContext
) -> List[StockSecFilingText]:
    if args.date_range:
        start_date = args.date_range.start_date
        end_date = args.date_range.end_date
    else:
        start_date = None
        end_date = None

    if args.form_types and len(args.form_types) > 0:
        form_types = [
            form_type.name
            for form_type in args.form_types
            if form_type.name != FILE_10K or form_type.name != FILE_10Q
        ]
        sec_10k_10q_filings: List[StockSecFilingText] = []
        if len(form_types) != len(args.form_types):
            sec_10k_10q_filings = await get_10k_10q_sec_filings(
                GetSecFilingsInput(
                    stock_ids=args.stock_ids,
                    date_range=args.date_range,
                ),
                context=context,
            )  # type: ignore

        form_types_str = ", ".join(form_types)
        await tool_log(
            (
                "Retrieving" f" {form_types_str}"
                if len(form_types) < 5
                else f"{len(form_types)} types of"
            )
            + (
                " SEC filings for" f" {args.stock_ids[0].symbol}"
                if len(args.stock_ids) == 1
                else f"{len(args.stock_ids)} stocks"
            )
            + (
                f" from {start_date.isoformat()} to {end_date.isoformat()}"
                if start_date and end_date
                else " in the last 100 days"
            ),
            context=context,
        )
        stock_filing_map = await get_other_sec_filings_helper(
            args.stock_ids, form_types, start_date, end_date
        )
        sec_filings: List[StockSecFilingText] = []
        for filings in stock_filing_map.values():
            sec_filings.extend(filings)

        if len(sec_filings) == 0 and len(sec_10k_10q_filings) == 0:
            raise EmptyOutputError(
                "No filings were retrieved for these stocks over this time period"
            )
        sec_filings.extend(sec_10k_10q_filings)

        filing_list = [
            f"{filing.to_citation_title()} ({filing.form_type}) "
            + (f"- {filing.timestamp.strftime('%Y-%m-%d')}" if filing.timestamp else "")
            for filing in sec_filings
        ]
        await tool_log(
            log=f"Found {len(sec_filings)} filing(s).",
            context=context,
            associated_data=filing_list,
        )

        return sec_filings

    return await get_10k_10q_sec_filings(
        GetSecFilingsInput(
            stock_ids=args.stock_ids,
            date_range=args.date_range,
        ),
        context=context,
    )  # type: ignore
