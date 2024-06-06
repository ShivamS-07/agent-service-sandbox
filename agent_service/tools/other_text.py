import datetime
import json
from typing import Dict, List, Optional

from agent_service.external.sec_utils import FILINGS, SecFiling, SecMapping
from agent_service.io_types.stock import StockID
from agent_service.io_types.text import SecFilingText, Text
from agent_service.tool import ToolArgs, ToolCategory, ToolRegistry, tool
from agent_service.tools.earnings import (  # get_stock_aligned_earnings_call_summaries,
    GetEarningsCallSummariesInput,
    get_earnings_call_summaries,
)
from agent_service.tools.news import (  # get_stock_aligned_news_developments,
    GetNewsDevelopmentsAboutCompaniesInput,
    get_all_news_developments_about_companies,
)
from agent_service.tools.stock_metadata import (  # get_company_descriptions_stock_aligned,
    GetStockDescriptionInput,
    get_company_descriptions,
)
from agent_service.tools.tool_log import tool_log
from agent_service.types import PlanRunContext
from agent_service.utils.date_utils import get_now_utc
from agent_service.utils.postgres import get_psql
from agent_service.utils.prefect import get_prefect_logger


async def get_sec_filings_helper(
    stock_ids: List[StockID], start_date: Optional[datetime.date], end_date: Optional[datetime.date]
) -> Dict[StockID, List[SecFilingText]]:
    stock_filing_map = {}
    gbi_id_metadata_map = get_psql().get_sec_metadata_from_gbi(
        gbi_ids=[stock.gbi_id for stock in stock_ids]
    )
    for stock_id in stock_ids:
        cik = await SecMapping.map_gbi_id_to_cik(stock_id.gbi_id, gbi_id_metadata_map)
        if cik is None:
            continue

        query = SecFiling.build_query_for_filings(cik, start_date, end_date)
        resp = SecFiling.query_api.get_filings(query)
        if (not resp) or (FILINGS not in resp) or (not resp[FILINGS]):
            continue
        stock_filing_map[stock_id] = [
            SecFilingText(id=json.dumps(filing), gbi_id=stock_id.gbi_id) for filing in resp[FILINGS]
        ]
    return stock_filing_map


class GetSecFilingsInput(ToolArgs):
    stock_ids: List[StockID]
    start_date: Optional[datetime.date] = None
    end_date: Optional[datetime.date] = None


@tool(
    description="Given a list of stock ID's, return a list of sec filings for the stocks. "
    "Specifically, this includes the management and risk factors sections of the 10-K and 10-Q documents "
    "which provides detailed, up-to-date information about company operations and financial status "
    "for the previous quarter (10-Q) or year (10-K) for any US-listed companies. "
    "It is especially useful for finding current information about less well-known "
    "companies which may have little or no news for a given period. "
    "Any documents published between start_date and end_date will be included, if the end_date is "
    "excluded it is assume to include documents up to today, if start_date is not "
    "included, the start date is a quarter ago, which includes only the latest SEC filing.",
    category=ToolCategory.TEXT,
    tool_registry=ToolRegistry,
)
async def get_sec_filings(args: GetSecFilingsInput, context: PlanRunContext) -> List[SecFilingText]:
    stock_filing_map = await get_sec_filings_helper(args.stock_ids, args.start_date, args.end_date)
    all_filings = []
    for filings in stock_filing_map.values():
        all_filings.extend(filings)

    if len(all_filings) == 0:
        raise Exception("No filings were retrieved for these stocks over this time period")
    return all_filings


class GetAllTextDataForStocksInput(ToolArgs):
    # name of the universe to lookup
    stock_ids: List[StockID]
    start_date: Optional[datetime.date] = None
    end_date: Optional[datetime.date] = None


@tool(
    description=(
        "This function takes a lists of stocks and returns a list of text objects with consists of"
        " all the text information that is available for these stocks. This data includes:\n"
        "1. Company descriptions\n2. News developments\n3. Earnings Call Summaries\n4. SEC Filings\n"
        "This function should be your default tool for getting data when the best source is not clear"
        "If a client asks for something related to qualitative information that could be contained in any of"
        " these sources, you must this function to get the data, though if they specifically mention a subset "
        " of these sources or it is clear which individual source would have the information required, "
        "you must gather data using the individual function. For example, if a client asked "
        "`Summarize the challenges McDonald's is facing in Asian markets`, relevant information"
        " might be spread across multiple sources and you can use this function, but if the client "
        "mentions they are specific interested in information from SEC filings, you should pull only that data."
        "If end_date is omitted, data up to the current day will be included, if the start date is "
        "omitted, all data for the last quarter will be included."
    ),
    category=ToolCategory.TEXT,
    tool_registry=ToolRegistry,
    is_visible=False,
)
async def get_all_text_data_for_stocks(
    args: GetAllTextDataForStocksInput, context: PlanRunContext
) -> List[Text]:
    stock_ids = args.stock_ids
    start_date = args.start_date
    end_date = args.end_date
    logger = get_prefect_logger(__name__)
    if not start_date:
        start_date = (get_now_utc() - datetime.timedelta(days=90)).date()
    if not end_date:
        # Add an extra day to be sure we don't miss anything with timezone weirdness
        end_date = get_now_utc().date() + datetime.timedelta(days=1)

    all_data: List[Text] = []
    await tool_log(log="Getting company descriptions", context=context)
    try:
        description_data = await get_company_descriptions(
            GetStockDescriptionInput(stock_ids=stock_ids), context=context
        )
        all_data.extend(description_data)  # type: ignore
    except Exception as e:
        logger.exception(f"failed to get company description(s) due to error: {e}")
    await tool_log(log="Getting news developments", context=context)
    try:
        news_data = await get_all_news_developments_about_companies(
            GetNewsDevelopmentsAboutCompaniesInput(
                stock_ids=stock_ids, start_date=start_date, end_date=end_date
            ),
            context=context,
        )
        all_data.extend(news_data)  # type: ignore
    except Exception as e:
        logger.exception(f"failed to get company news due to error: {e}")
    await tool_log(log="Getting earnings summaries", context=context)
    try:
        earnings_data = await get_earnings_call_summaries(
            GetEarningsCallSummariesInput(
                stock_ids=stock_ids, start_date=start_date, end_date=end_date
            ),
            context=context,
        )
        all_data.extend(earnings_data)  # type: ignore
    except Exception as e:
        logger.exception(f"failed to get earnings summaries due to error: {e}")
    await tool_log(log="Getting SEC filings", context=context)
    try:
        sec_filings = await get_sec_filings(
            GetSecFilingsInput(stock_ids=stock_ids, start_date=start_date, end_date=end_date),
            context=context,
        )
        all_data.extend(sec_filings)  # type: ignore
    except Exception as e:
        logger.exception(f"failed to get SEC filings(s) due to error: {e}")
    await tool_log(log="Combining all text data", context=context)
    if len(all_data) == 0:
        raise Exception("Found no data for the provided stocks")
    return all_data
