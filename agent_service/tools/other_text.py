import asyncio
import datetime
from typing import List, Optional

from agent_service.io_types.dates import DateRange
from agent_service.io_types.stock import StockID
from agent_service.io_types.text import StockEarningsText, StockText
from agent_service.tool import ToolArgs, ToolCategory, default_tool_registry, tool
from agent_service.tools.custom_documents import (
    GetCustomDocsInput,
    get_user_custom_documents,
)
from agent_service.tools.earnings import (
    GetEarningsCallDataInput,
    get_earnings_call_summaries,
)
from agent_service.tools.news import (
    GetLatestNewsForCompaniesInput,
    GetNewsDevelopmentsAboutCompaniesInput,
    get_all_news_developments_about_companies,
    get_latest_news_for_companies,
)
from agent_service.tools.sec import GetSecFilingsInput, get_10k_10q_sec_filings
from agent_service.tools.stock_metadata import (  # get_company_descriptions_stock_aligned,
    GetStockDescriptionInput,
    get_company_descriptions,
)
from agent_service.tools.tool_log import tool_log
from agent_service.tools.web_search.general_websearch import (
    NEWS_URLS_TO_SCRAPE,
    REDUCED_NEWS_URLS_TO_SCRAPE,
    REDUCED_URLS_TO_SCRAPE,
    URLS_TO_SCRAPE,
)
from agent_service.types import PlanRunContext
from agent_service.utils.feature_flags import get_ld_flag, get_user_context
from agent_service.utils.logs import async_perf_logger
from agent_service.utils.prefect import get_prefect_logger


class GetAllTextDataForStocksInput(ToolArgs):
    # name of the universe to lookup
    stock_ids: List[StockID]
    start_date: Optional[datetime.date] = None
    end_date: Optional[datetime.date] = None
    date_range: Optional[DateRange] = None


@tool(
    description=(
        "This function takes a lists of stocks and returns a list of text objects which consists of"
        "the most important text information that is available for these stocks including company descriptions, "
        "news developments, earnings summaries, SEC filings, and user uploaded documents. "
        "This is a substitute for calling the specific text retrieval tools, that are available for each "
        "of these text types. So if you called this function, you MUST not call the other text retrieval tools and "
        "vice versa. If you do, your plan will be very slow and you will be punished. "
        "This function should be used for getting data when the user just asks to look "
        "at text data generally for a company or companies but does not make any mention any specific type "
        "(i.e. there is no mention of news, earnings, or SEC filings) and the best type is not otherwise obvious. "
        "If a client asks for something related to qualitative information that could be contained in any of "
        "of various sources, you must use this function to get the data. For example, if a client asked "
        "`Summarize the challenges McDonald's is facing in Asian markets`, relevant information"
        "might be spread across multiple sources and you can use this tool. However, if they say "
        "`Summarize the challenges McDonald's is facing in Asian markets based on their SEC filings`"
        "i.e. they specifically mentioned an text type, you must NOT use this tool, and instead gather"
        "data using the individual tool associated with that data source, i.e. get_10k_10q_sec_filings. "
        "Again, you should never use this tool if the client specifically asks for particular "
        "kinds of text, use the corresponding tool for that kind of text and then add the text lists "
        "using `add_lists`."
        "Specifically, NEVER use this tool if the user explicily mentions news!"
        "NEVER use this tool if the user mentions SEC filings! "
        "NEVER use this tool if the user mentions earnings calls! "
        "And you must absolutely never use this tool when the clients asks for specific kinds "
        "of SEC filings that are not 10-K/Q (e.g. 8-K), this tool will never return such filings "
        "When the user asks for specific filings, use the get_sec_filings_with_type tool. "
        "The fact that a client mentions multiple text sources must NOT be used an excuse to call this "
        "tool, any mention of any of the specific sources must mean you do NOT use this tool."
        "Seriously, you will be punished if you use this tool when the user asks for specific kinds of documents, "
        "If end_date is omitted, data up to the current day will be included, if the start date is "
        "omitted, all data for the last quarter will be included."
        " You should not pass a date_range containing dates after todays date into this function."
        " documents can only be found for dates in the past up to the present, including todays date."
        " I repeat you will be punished if you try to find documents from the future!!! YOU MUST NEVER DO THAT!!!"
    ),
    category=ToolCategory.TEXT_RETRIEVAL,
    tool_registry=default_tool_registry(),
    is_visible=False,
    store_output=False,
)
@async_perf_logger
async def get_default_text_data_for_stocks(
    args: GetAllTextDataForStocksInput, context: PlanRunContext
) -> List[StockText]:
    stock_ids = args.stock_ids
    logger = get_prefect_logger(__name__)
    all_data: List[StockText] = []
    using_default_date_range = False

    if not args.date_range:
        using_default_date_range = True
        today = context.as_of_date.date() if context.as_of_date else datetime.date.today()
        start_date = today - datetime.timedelta(days=90)
        # Add an extra day to be sure we don't miss anything with timezone weirdness
        end_date = today + datetime.timedelta(days=1)
        args.date_range = DateRange(start_date=start_date, end_date=end_date)

    async def _get_web_results() -> None:
        await tool_log(
            log="Getting company descriptions and news developments from the web",
            context=context,
        )
        try:
            date_range = None
            if args.start_date and args.end_date:
                date_range = DateRange(start_date=args.start_date, end_date=args.end_date)
            news_data = await get_latest_news_for_companies(
                GetLatestNewsForCompaniesInput(
                    stock_ids=stock_ids,
                    topic="",
                    get_developments=False,
                    num_google_urls=(
                        URLS_TO_SCRAPE if len(args.stock_ids) <= 10 else REDUCED_URLS_TO_SCRAPE
                    ),
                    num_news_urls=(
                        NEWS_URLS_TO_SCRAPE
                        if len(args.stock_ids) <= 10
                        else REDUCED_NEWS_URLS_TO_SCRAPE
                    ),
                    date_range=date_range,
                ),
                context=context,
            )
            all_data.extend(news_data)  # type: ignore
        except Exception as e:
            logger.exception(f"failed to get web news due to error: {e}")

    async def _get_company_descriptions() -> None:
        try:
            description_data = await get_company_descriptions(
                GetStockDescriptionInput(stock_ids=stock_ids), context=context
            )
            all_data.extend(description_data)  # type: ignore
        except Exception as e:
            logger.exception(f"failed to get company description(s) due to error: {e}")

    async def _get_all_news_developments_about_companies() -> None:
        try:
            news_data = await get_all_news_developments_about_companies(
                GetNewsDevelopmentsAboutCompaniesInput(
                    stock_ids=stock_ids,
                    date_range=args.date_range,
                ),
                context=context,
            )
            all_data.extend(news_data)  # type: ignore
        except Exception as e:
            logger.exception(f"failed to get company news due to error: {e}")

    async def _get_earnings_call_summaries() -> None:
        try:
            earnings_data: List[StockEarningsText] = []
            if using_default_date_range:
                # Grab the latest earnings provided they are not older than a year
                earnings_data_unfiltered: List[
                    StockEarningsText
                ] = await get_earnings_call_summaries(
                    GetEarningsCallDataInput(stock_ids=stock_ids),
                    context=context,
                )  # type: ignore
                date_cutoff = datetime.date.today() - datetime.timedelta(days=365)

                logger.info(
                    "Filtering earnings texts to those that happened within a year "
                    f"(earning calls after {date_cutoff.strftime("%Y-%m-%d")})"
                )

                for data in earnings_data_unfiltered:
                    if data.timestamp and data.timestamp.date() > date_cutoff:
                        earnings_data.append(data)

            else:
                earnings_data = await get_earnings_call_summaries(
                    GetEarningsCallDataInput(stock_ids=stock_ids, date_range=args.date_range),
                    context=context,
                )  # type: ignore
            all_data.extend(earnings_data)  # type: ignore
        except Exception as e:
            logger.exception(f"failed to get earnings summaries due to error: {e}")

    async def _get_10k_10q_sec_filings() -> None:
        try:
            sec_filings = await get_10k_10q_sec_filings(
                GetSecFilingsInput(stock_ids=stock_ids, date_range=args.date_range),
                context=context,
            )
            all_data.extend(sec_filings)  # type: ignore
        except Exception as e:
            logger.exception(f"failed to get SEC filings(s) due to error: {e}")

    async def _get_user_custom_documents() -> None:
        try:
            custom_docs = await get_user_custom_documents(
                GetCustomDocsInput(
                    stock_ids=stock_ids,
                    date_range=args.date_range,
                ),
                context=context,
            )
            all_data.extend(custom_docs)  # type: ignore
        except Exception as e:
            logger.exception(f"failed to get user custom documents due to error: {e}")

    tasks = [
        _get_company_descriptions(),
        _get_all_news_developments_about_companies(),
        _get_earnings_call_summaries(),
        _get_10k_10q_sec_filings(),
        _get_user_custom_documents(),
    ]

    if (
        args.date_range.end_date >= datetime.date.today() - datetime.timedelta(days=1)
        and get_ld_flag(
            flag_name="web-search-tool",
            default=False,
            user_context=get_user_context(user_id=context.user_id),
        )
        and context.user_settings.include_web_results
    ):
        tasks.append(_get_web_results())

    await asyncio.gather(*tasks)

    await tool_log(log=f"Combining all text data, size: {len(all_data)}", context=context)
    if len(all_data) == 0:
        await tool_log(log="Found no data for the provided stocks", context=context)
    return all_data


@tool(
    description="",
    category=ToolCategory.TEXT_RETRIEVAL,
    tool_registry=default_tool_registry(),
    enabled=False,
    store_output=False,
)
async def get_all_text_data_for_stocks(
    args: GetAllTextDataForStocksInput, context: PlanRunContext
) -> List[StockText]:
    return await get_default_text_data_for_stocks(args, context)  # type: ignore


class GetAllStocksFromTextInput(ToolArgs):
    stock_texts: List[StockText]


@tool(
    description=("This function takes a lists of stock texts and returns a list of stock ids."),
    category=ToolCategory.STOCK,
    tool_registry=default_tool_registry(),
    enabled=True,
    store_output=False,
)
async def get_all_stocks_from_text(
    args: GetAllStocksFromTextInput, context: PlanRunContext
) -> List[StockID]:
    groups: dict[int, StockID] = {}
    for stock_text in args.stock_texts:
        if stock_text is None or stock_text.stock_id is None:
            continue

        gbi_id = stock_text.stock_id.gbi_id
        stock_id = stock_text.stock_id
        if gbi_id not in groups:
            groups[gbi_id] = stock_id
        else:
            groups[gbi_id]._extend_history_from(stock_id)
    return list(groups.values())
