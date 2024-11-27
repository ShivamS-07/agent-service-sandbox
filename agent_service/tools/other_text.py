import asyncio
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
        "This function takes a lists of stocks and returns a list of text objects with consists of"
        "the most important text information that is available for these stocks. "
        "This is a substitute for calling specific text retrieval tools, you must, ever never call both this "
        "AND get_news/earnings/sec filings tools."
        "Instead, this function should be your default tool for getting data when the user just asks to look "
        "at text data generally for a company or companies but does not make any mention any specific type "
        "(i.e. there is no mention of news, earnings, or SEC filings) and the best type is not otherwise obvious. "
        "If a client asks for something related to qualitative information that could be contained in any of "
        "of various sources, you must this function to get the data. For example, if a client asked "
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
        "Seriously, you will be fired if you use this tool when the user asks for specific kinds of documents, "
        "If end_date is omitted, data up to the current day will be included, if the start date is "
        "omitted, all data for the last quarter will be included."
        " You should not pass a date_range containing dates after todays date into this function."
        " documents can only be found for dates in the past up to the present, including todays date."
        " I repeat you will be FIRED if you try to find documents from the future!!! YOU MUST NEVER DO THAT!!!"
    ),
    category=ToolCategory.TEXT_RETRIEVAL,
    tool_registry=ToolRegistry,
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

    if not args.date_range:
        # perform web search IF there is no date_range
        if (
            get_ld_flag(
                flag_name="web-search-tool",
                default=False,
                user_context=get_user_context(user_id=context.user_id),
            )
            and context.user_settings.include_web_results
        ):
            await tool_log(
                log="Getting company descriptions and news developments from the web",
                context=context,
            )
            try:
                news_data = await get_latest_news_for_companies(
                    GetLatestNewsForCompaniesInput(
                        stock_ids=stock_ids,
                        get_developments=False,
                    ),
                    context=context,
                )
                all_data.extend(news_data)  # type: ignore
            except Exception as e:
                logger.exception(f"failed to get web news due to error: {e}")

        today = context.as_of_date.date() if context.as_of_date else datetime.date.today()
        start_date = today - datetime.timedelta(days=90)
        # Add an extra day to be sure we don't miss anything with timezone weirdness
        end_date = today + datetime.timedelta(days=1)
        args.date_range = DateRange(start_date=start_date, end_date=end_date)

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
            earnings_data = await get_earnings_call_summaries(
                GetEarningsCallDataInput(stock_ids=stock_ids, date_range=args.date_range),
                context=context,
            )
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

    await asyncio.gather(
        _get_company_descriptions(),
        _get_all_news_developments_about_companies(),
        _get_earnings_call_summaries(),
        _get_10k_10q_sec_filings(),
        _get_user_custom_documents(),
    )

    await tool_log(log=f"Combining all text data, size: {len(all_data)}", context=context)
    if len(all_data) == 0:
        await tool_log(log="Found no data for the provided stocks", context=context)
    return all_data


@tool(
    description="",
    category=ToolCategory.TEXT_RETRIEVAL,
    tool_registry=ToolRegistry,
    enabled=False,
    store_output=False,
)
async def get_all_text_data_for_stocks(
    args: GetAllTextDataForStocksInput, context: PlanRunContext
) -> List[StockText]:
    return await get_default_text_data_for_stocks(args, context)  # type: ignore
