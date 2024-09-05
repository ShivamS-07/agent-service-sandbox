import datetime
from typing import List, Optional

from agent_service.io_types.dates import DateRange
from agent_service.io_types.stock import StockID
from agent_service.io_types.text import StockText
from agent_service.tool import ToolArgs, ToolCategory, ToolRegistry, tool
from agent_service.tools.custom_documents import (
    GetCustomDocsInput,
    get_user_custom_documents,
)
from agent_service.tools.earnings import (
    GetEarningsCallDataInput,
    get_earnings_call_summaries,
)
from agent_service.tools.news import (
    GetNewsDevelopmentsAboutCompaniesInput,
    get_all_news_developments_about_companies,
)
from agent_service.tools.sec import GetSecFilingsInput, get_10k_10q_sec_filings
from agent_service.tools.stock_metadata import (  # get_company_descriptions_stock_aligned,
    GetStockDescriptionInput,
    get_company_descriptions,
)
from agent_service.tools.tool_log import tool_log
from agent_service.types import PlanRunContext
from agent_service.utils.prefect import get_prefect_logger


class GetAllTextDataForStocksInput(ToolArgs):
    # name of the universe to lookup
    stock_ids: List[StockID]
    start_date: Optional[datetime.date] = None
    end_date: Optional[datetime.date] = None
    date_range: Optional[DateRange] = None


@tool(
    description=(
        "This function takes a lists of stocks and returns a list of text objects with consists of"
        " all the text information that is available for these stocks. This data includes:\n"
        "1. Company descriptions\n2. News developments\n3. Earnings Call Summaries\n4. 10k/q SEC Filings\n"
        "This function should be your default tool for getting data when the best source is not clear. "
        "However, you should never use this tool if the client specifically asks for one of the above "
        "kinds of text, use the corresponding tool for that kind of text. You must absolutely never use "
        "this tool when the clients asks for specific kinds of SEC filings that are not 10k/q, this tool"
        "will never return anything but 10k/q SEC filings."
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
    store_output=False,
)
async def get_all_text_data_for_stocks(
    args: GetAllTextDataForStocksInput, context: PlanRunContext
) -> List[StockText]:
    stock_ids = args.stock_ids

    logger = get_prefect_logger(__name__)
    if not args.date_range:
        today = context.as_of_date.date() if context.as_of_date else datetime.date.today()
        start_date = today - datetime.timedelta(days=90)
        # Add an extra day to be sure we don't miss anything with timezone weirdness
        end_date = today + datetime.timedelta(days=1)
        args.date_range = DateRange(start_date=start_date, end_date=end_date)

    all_data: List[StockText] = []

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
            GetNewsDevelopmentsAboutCompaniesInput(stock_ids=stock_ids, date_range=args.date_range),
            context=context,
        )
        all_data.extend(news_data)  # type: ignore
    except Exception as e:
        logger.exception(f"failed to get company news due to error: {e}")

    await tool_log(log="Getting earnings summaries", context=context)
    try:
        earnings_data = await get_earnings_call_summaries(
            GetEarningsCallDataInput(stock_ids=stock_ids, date_range=args.date_range),
            context=context,
        )
        all_data.extend(earnings_data)  # type: ignore
    except Exception as e:
        logger.exception(f"failed to get earnings summaries due to error: {e}")

    await tool_log(log="Getting SEC filings", context=context)
    try:
        sec_filings = await get_10k_10q_sec_filings(
            GetSecFilingsInput(stock_ids=stock_ids, date_range=args.date_range),
            context=context,
        )
        all_data.extend(sec_filings)  # type: ignore
    except Exception as e:
        logger.exception(f"failed to get SEC filings(s) due to error: {e}")

    await tool_log(log="Getting user uploaded documents", context=context)
    try:
        custom_docs = await get_user_custom_documents(
            GetCustomDocsInput(
                stock_ids=stock_ids,
                date_range=args.date_range,
                limit=None,
            ),
            context=context,
        )
        all_data.extend(custom_docs)  # type: ignore
    except Exception as e:
        logger.exception(f"failed to get user custom documents due to error: {e}")

    await tool_log(log="Combining all text data", context=context)
    if len(all_data) == 0:
        raise Exception("Found no data for the provided stocks")
    return all_data
