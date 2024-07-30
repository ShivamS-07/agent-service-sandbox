import datetime
import json
from collections import defaultdict
from typing import Dict, List, Optional

from agent_service.io_types.dates import DateRange
from agent_service.io_types.stock import StockID
from agent_service.io_types.text import StockOtherSecFilingText, StockSecFilingText
from agent_service.tool import ToolArgs, ToolCategory, ToolRegistry, tool
from agent_service.types import PlanRunContext
from agent_service.utils.date_utils import timezoneify
from agent_service.utils.sec.constants import FILE_10K, FILE_10Q
from agent_service.utils.sec.sec_api import SecFiling


async def get_sec_filings_helper(
    stock_ids: List[StockID], start_date: Optional[datetime.date], end_date: Optional[datetime.date]
) -> Dict[StockID, List[StockSecFilingText]]:
    stock_filing_map = defaultdict(list)

    gbi_id_to_stock_id = {stock.gbi_id: stock for stock in stock_ids}

    filing_gbi_pairs, filing_to_db_id = SecFiling.get_filings(
        gbi_ids=list(gbi_id_to_stock_id.keys()),
        form_types=[FILE_10K, FILE_10Q],
        start_date=start_date,
        end_date=end_date,
    )

    for filing_str, gbi_id in filing_gbi_pairs:
        stock_id = gbi_id_to_stock_id[gbi_id]
        db_id = filing_to_db_id.get(filing_str, None)
        timestamp = timezoneify(datetime.datetime.fromisoformat(json.loads(filing_str)["filedAt"]))
        stock_filing_map[stock_id].append(
            StockSecFilingText(id=filing_str, stock_id=stock_id, db_id=db_id, timestamp=timestamp)
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

    stock_filing_map = await get_sec_filings_helper(args.stock_ids, start_date, end_date)
    all_filings = []
    for filings in stock_filing_map.values():
        all_filings.extend(filings)

    if len(all_filings) == 0:
        raise Exception("No filings were retrieved for these stocks over this time period")
    return all_filings


async def get_other_sec_filings_helper(
    stock_ids: List[StockID],
    form_types: List[str],
    start_date: Optional[datetime.date],
    end_date: Optional[datetime.date],
) -> Dict[StockID, List[StockOtherSecFilingText]]:
    gbi_id_to_stock_id = {stock.gbi_id: stock for stock in stock_ids}
    filing_gbi_pairs, filing_to_db_id = SecFiling.get_filings(
        gbi_ids=list(gbi_id_to_stock_id.keys()),
        form_types=form_types,
        start_date=start_date,
        end_date=end_date,
    )

    stock_filing_map = defaultdict(list)
    for filing_str, gbi_id in filing_gbi_pairs:
        stock_id = gbi_id_to_stock_id[gbi_id]
        db_id = filing_to_db_id.get(filing_str, None)
        timestamp = timezoneify(datetime.datetime.fromisoformat(json.loads(filing_str)["filedAt"]))
        stock_filing_map[stock_id].append(
            StockOtherSecFilingText(
                id=filing_str, stock_id=stock_id, db_id=db_id, timestamp=timestamp
            )
        )

    if not stock_filing_map:
        raise Exception("No filings were found.")

    return stock_filing_map


class GetOtherSecFilingsInput(ToolArgs):
    stock_ids: List[StockID]
    form_types: List[str]
    start_date: Optional[datetime.date] = None
    end_date: Optional[datetime.date] = None
    date_range: Optional[DateRange] = None


@tool(
    description="Given a list of stock ID's, return a list of requested SEC filings for the stocks."
    " Specifically, this includes all the supported types of filings that are NOT 10-K or 10-Q."
    " You must ONLY use this tool when user specifies a type or some types of filing documents."
    " As an example, if a user asked `Please give me a brief summary for Apple's latest 8-K filing`"
    " , you would use the function to get the information because it's in the list."
    " I will say it again, if the type is `10-K` or `10-Q`, you should also NOT use this tool."
    " Any documents published between start_date and end_date will be included, if the end_date is"
    " excluded it is assumed to include documents up to today, if start_date is not"
    " included, the start date is a quarter ago, which includes only the latest SEC filing.",
    category=ToolCategory.SEC_FILINGS,
    tool_registry=ToolRegistry,
    store_output=False,
)
async def get_non_10k_10q_sec_filings(
    args: GetOtherSecFilingsInput, context: PlanRunContext
) -> List[StockOtherSecFilingText]:
    if args.date_range:
        start_date = args.date_range.start_date
        end_date = args.date_range.end_date
    else:
        start_date = None
        end_date = None

    stock_filing_map = await get_other_sec_filings_helper(
        args.stock_ids, args.form_types, start_date, end_date
    )
    all_filings = []
    for filings in stock_filing_map.values():
        all_filings.extend(filings)

    if len(all_filings) == 0:
        raise Exception("No filings were retrieved for these stocks over this time period")
    return all_filings
