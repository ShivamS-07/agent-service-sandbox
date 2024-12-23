import datetime
import json
from collections import defaultdict
from heapq import heappop, heappush
from typing import Dict, List, Optional, Tuple, Union

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
from agent_service.tool import ToolArgs, ToolCategory, default_tool_registry, tool
from agent_service.tools.tool_log import tool_log
from agent_service.types import PlanRunContext
from agent_service.utils.boosted_pg import BoostedPG
from agent_service.utils.date_utils import parse_date_str_in_utc
from agent_service.utils.feature_flags import get_ld_flag
from agent_service.utils.prefect import get_prefect_logger
from agent_service.utils.prompt_utils import Prompt
from agent_service.utils.sec.constants import (
    FILE_10K,
    FILE_10Q,
    FILE_20F,
    FILE_40F,
    FILED_TIMESTAMP,
    FORM_TYPE,
)
from agent_service.utils.sec.sec_api import SecFiling
from agent_service.utils.sec.supported_types import SUPPORTED_TYPE_MAPPING
from agent_service.utils.string_utils import repair_json_if_needed

MATCH_RATIO = 70  # minimum ratio for fuzzy matching to map filing type (higher is more strict)


def use_full_text_10k(context: PlanRunContext) -> bool:
    return get_ld_flag(
        flag_name="use-full-text-10k",
        default=False,
        user_context=context.user_id,
    )


async def get_sec_filings_helper(
    stock_ids: List[StockID],
    context: PlanRunContext,
    form_types: Optional[List[str]],
    start_date: Optional[datetime.date],
    end_date: Optional[datetime.date],
    use_full_text: Optional[bool] = False,
) -> Dict[StockID, List[Union[StockSecFilingText | StockOtherSecFilingText]]]:
    if not form_types:
        form_types = [FILE_10Q, FILE_10K]

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
        timestamp = parse_date_str_in_utc(filing_json[FILED_TIMESTAMP])
        form_type = filing_json.get(FORM_TYPE)
        if (
            form_type in [FILE_10Q, FILE_10K]
            and not use_full_text
            and not use_full_text_10k(context)
        ):
            stock_filing_map[stock_id].append(
                StockSecFilingText(
                    id=filing_str,
                    stock_id=stock_id,
                    db_id=db_id,
                    timestamp=timestamp,
                    form_type=filing_json.get(FORM_TYPE),
                )
            )
        else:
            stock_filing_map[stock_id].append(
                StockOtherSecFilingText(
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
    must_include_10q: Optional[bool] = True
    must_include_10k: Optional[bool] = True
    use_full_text: Optional[bool] = False


@tool(
    description="Given a list of stock ID's, return a list of 10-K/10-Q SEC filings for the stocks. "
    "Specifically, this includes the management and risk factors sections of the 10-K and 10-Q documents "
    "which provides detailed, up-to-date information about company operations and financial status "
    "for the previous quarter (10-Q) or year (10-K) for any US-listed companies. "
    "This tool should be used by default when users just ask for SEC filings, or when they explicitly ask "
    "for `10-K` or `10-Q` filings."
    "By default, this tool only retrieves both quarterly 10-Q and 10-K reports. If the user explicitly asks"
    "for ONLY 10-Q, set the `must_include_10k` to False. Similarly, if the user explicitly asks for"
    "ONLY 10-K, set the `must_include_10q` to False."
    "It is especially useful for finding current information about less well-known "
    "companies which may have little or no news for a given period. "
    "Any documents published within the date_range are included. Date_range will "
    "default to the last year. If the user is only looking for 10-Q, the date range will be the last quarter."
    "The output of this tool is not ordered by time. If the client needs to compare two filings, you "
    "must call this tool twice with two 3 month time ranges, do NOT do this in a single call! "
    " You should not pass a date_range containing dates after todays date into this function."
    " documents can only be found for dates in the past up to the present, including todays date."
    " I repeat you will be FIRED if you try to find documents from the future!!! YOU MUST NEVER DO THAT!!!"
    "If the client asks to use the FULL 10-K or FULL 10-Q, you must set the `use_full_text` parameter to True."
    "Only set it to true if the client explicitly asks for the full filing text.",
    category=ToolCategory.SEC_FILINGS,
    tool_registry=default_tool_registry(),
    store_output=False,
    enabled=True,
)
async def get_10k_10q_sec_filings(
    args: GetSecFilingsInput, context: PlanRunContext
) -> List[StockSecFilingText]:
    form_types: List[SecFilingType | str] = [FILE_10Q, FILE_10K, FILE_20F, FILE_40F]
    if not args.must_include_10q:
        form_types.remove(FILE_10Q)
    if not args.must_include_10k:
        form_types.remove(FILE_10K)
        form_types.remove(FILE_20F)
        form_types.remove(FILE_40F)

    return await get_sec_filings_with_type(
        GetSecFilingsWithTypeInput(
            stock_ids=args.stock_ids,
            form_types=form_types,
            date_range=args.date_range,
            use_full_text=args.use_full_text,
        ),
        context=context,
    )  # type: ignore


class GetOtherSecFilingsInput(ToolArgs):
    stock_ids: List[StockID]
    form_types: List[str]
    start_date: Optional[datetime.date] = None
    end_date: Optional[datetime.date] = None
    date_range: Optional[DateRange] = None


@tool(
    description="This function was renamed to get_sec_filings_with_type",
    category=ToolCategory.SEC_FILINGS,
    tool_registry=default_tool_registry(),
    store_output=False,
    enabled=False,
)
async def get_non_10k_10q_sec_filings(
    args: GetOtherSecFilingsInput, context: PlanRunContext
) -> List[StockSecFilingText]:
    form_types: List[SecFilingType | str] = [
        SecFilingType(name=form_type) for form_type in args.form_types
    ]
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
    " Never, ever use `get_default_text_data_for_stocks` as a substitute for this tool if the "
    " client says they want to look SEC filings or 10-K/Q data for companies, you must always use this"
    " tool even if multiple data sources are mentioned"
    " As an example, if a user asked `Describe the corporate structure for Apple`,"
    " you would include `Corporate Structure` in the search_term."
    " If the user specifically requests certain SEC filings to be included in the query, this"
    " list of SEC filing types must be passed to `form_types` to force the tool to"
    " also include those filing types in its output."
    " As an example, if a user asked `Please find IPO details for Apple in S-1`,"
    " you must include `S-1` in the form_types parameter",
    category=ToolCategory.SEC_FILINGS,
    tool_registry=default_tool_registry(),
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
            best_match = heappop(ratios_heap)
            if best_match[0] <= -1 * MATCH_RATIO:
                sec_filing_types.add(best_match[1])

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

    if len(sec_filing_types) > 0:
        output_types = [SecFilingType(name=filing_type) for filing_type in sec_filing_types]
        sec_types_str = ", ".join(sec_type.name for sec_type in output_types[0:3])
        if len(output_types) > 3:
            sec_types_str += f" and {len(output_types) - 3} other"
        await tool_log(
            log=f"Using {sec_types_str} filing type(s).",
            context=context,
            associated_data=output_types,
        )
        return output_types

    await tool_log(
        log="Using 10-K and 10-Q SEC filings.",
        context=context,
        associated_data=[SecFilingType(name=FILE_10K), SecFilingType(name=FILE_10K)],
    )
    return [SecFilingType(name=FILE_10K), SecFilingType(name=FILE_10K)]


class GetSecFilingsWithTypeInput(ToolArgs):
    stock_ids: List[StockID]
    form_types: List[Union[SecFilingType, str]]
    date_range: Optional[DateRange] = None
    use_full_text: Optional[bool] = False


@tool(
    description="Given a list of stock ID's, return a list of requested SEC filings for the stocks."
    " This tool should be used after `sec_filings_type_lookup` which will determine the SEC filing types"
    " we need to retrieve. This tool should never be called unless it is after `sec_filings_type_lookup`."
    " This tool should not be called if we only want to retrieve general information from SEC filings,"
    " instead the `get_10k_10q_sec_filings` tool should be used."
    " The output of this tool should never be outputted directly, instead you should use the `summarize_texts`"
    " or `answer_question_with_text_data` tool on the text output of this tool. Do not output it directly."
    " If a client asks for specific filings other than 10-k/q, you MUST use this tool, the SEC filings "
    " that this tool accesses are NOT included in the output of `get_default_text_data_for_stocks`!"
    " Any documents published between start_date and end_date will be included, if the end_date is"
    " excluded it is assumed to include documents up to today, if start_date is not"
    " included, the start date is a quarter ago, which includes only the latest SEC filing."
    " You should not pass a date_range containing dates after todays date into this function."
    " documents can only be found for dates in the past up to the present, including todays date."
    " I repeat you will be FIRED if you try to find documents from the future!!! YOU MUST NEVER DO THAT!!!"
    "If the client asks to use the FULL 10-K or FULL 10-Q, you must set the `use_full_text` parameter to True."
    "Only set it to true if the client explicitly asks for the full filing text.",
    category=ToolCategory.SEC_FILINGS,
    tool_registry=default_tool_registry(),
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

    num_input_stocks = len(args.stock_ids)
    stocks_str = f"{len(args.stock_ids)} stocks"
    if num_input_stocks == 0:
        return []
    elif num_input_stocks == 1:
        input_stock = args.stock_ids[0]
        stocks_str = input_stock.symbol if input_stock.symbol else input_stock.company_name

    daterange_str = ""
    use_latest_str = "the most recent "
    if start_date and end_date:
        daterange_str = f"from {start_date.isoformat()} to {end_date.isoformat()}"
        use_latest_str = ""

    form_types = [FILE_10Q, FILE_10K, FILE_20F, FILE_40F]
    if args.form_types and len(args.form_types) > 0:
        form_types = []
        for form_type in args.form_types:
            form_types.append(form_type if isinstance(form_type, str) else form_type.name)

    await tool_log(
        f"Retrieving {use_latest_str} {", ".join(form_types)} SEC filings for {stocks_str} {daterange_str}.",
        context=context,
    )

    stock_filing_map = await get_sec_filings_helper(
        args.stock_ids,
        context,
        form_types,
        start_date,
        end_date,
        args.use_full_text,
    )
    sec_filings: List[Union[StockSecFilingText | StockOtherSecFilingText]] = []
    for filings in stock_filing_map.values():
        sec_filings.extend(filings)

    if len(sec_filings) == 0:
        await tool_log(
            log="No filings were retrieved for these stocks over this time period",
            context=context,
        )
        return sec_filings

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
