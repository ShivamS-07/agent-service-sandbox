import asyncio
import logging
from typing import List, Optional

from agent_service.io_types.dates import DateRange
from agent_service.io_types.stock import StockID
from agent_service.io_types.text import WebStockText, WebText
from agent_service.planner.errors import EmptyOutputError
from agent_service.tool import (
    ToolArgMetadata,
    ToolArgs,
    ToolCategory,
    default_tool_registry,
    tool,
)
from agent_service.tools.tool_log import tool_log
from agent_service.tools.web_search.brightdata_websearch import (
    get_urls_async,
    get_web_texts_async,
)
from agent_service.types import AgentUserSettings, PlanRunContext
from agent_service.utils.async_utils import gather_with_concurrency
from agent_service.utils.feature_flags import get_ld_flag

URLS_TO_SCRAPE = 4
NEWS_URLS_TO_SCRAPE = 8

REDUCED_URLS_TO_SCRAPE = 2
REDUCED_NEWS_URLS_TO_SCRAPE = 4

logger = logging.getLogger(__name__)


def prepend_url_with_https(url: str) -> str:
    if not url.startswith("https://") and not url.startswith("http://"):
        return "https://" + url
    return url


def enabler_function(user_id: Optional[str], user_settings: Optional[AgentUserSettings]) -> bool:
    result = (
        get_ld_flag("web-search-tool", default=False, user_context=user_id)
        and user_settings
        and user_settings.include_web_results
    )
    return bool(result)


class SingleStockWebSearchInput(ToolArgs):
    stock_id: StockID
    query: str
    num_google_urls: int = URLS_TO_SCRAPE
    num_news_urls: int = NEWS_URLS_TO_SCRAPE
    date_range: Optional[DateRange] = None


# TODO: calling MANY of these separately may be inefficient, may be opportunity for speedups here
@tool(
    description=(
        "This function takes in a StockID and a single query which contain search details ALONGSIDE the stock name and "
        "returns text entries of the top search results when the query is made on the web. "
        "Be SURE that the stock name is included in the query or we won't know what stock is being referred to! "
        "Unless not specified within a sample plan, always call some text processing tool sometime after this tool. "
        "Again, it is VERY important that a "
        "text processing tool is called before the end of a plan containing this tool! DO not EVER directly output "
        "the returned text from this tool! AGAIN, DO NOT DIRECTLY OUTPUT THE RESULTS OF THIS TOOL!!!"
    ),
    category=ToolCategory.WEB,
    tool_registry=default_tool_registry(),
    enabled=False,
)
async def single_stock_web_search(
    args: SingleStockWebSearchInput, context: PlanRunContext
) -> List[WebStockText]:
    if not context.user_settings.include_web_results:
        await tool_log("Skipping web search due to user setting", context=context)
        return []
    query = args.query
    stock_id = args.stock_id
    tasks = [
        get_urls_async(
            [query],
            args.num_google_urls,
            context=context,
            log_event_dict={"gbi_id": stock_id.gbi_id},
            date_range=args.date_range,
        ),
        get_urls_async(
            [query],
            args.num_news_urls,
            context=context,
            log_event_dict={"gbi_id": stock_id.gbi_id},
            get_news=True,
            date_range=args.date_range,
        ),
    ]
    results = await gather_with_concurrency(tasks)

    urls: List[str] = []
    for result in results:
        urls.extend(result)

    search_results = await get_web_texts_async(
        urls=urls,
        plan_context=context,
        log_event_dict={"gbi_id": stock_id.gbi_id},
        date_range=args.date_range,
    )
    web_stock_text_results = [
        WebStockText(
            stock_id=stock_id,
            id=web_text.id,
            url=web_text.url,
            title=web_text.title,
            timestamp=web_text.timestamp,
        )
        for web_text in search_results
    ]
    return web_stock_text_results


class GeneralStockWebSearchInput(ToolArgs):
    stock_ids: List[StockID]
    topic: str
    num_google_urls: int = URLS_TO_SCRAPE
    num_news_urls: int = NEWS_URLS_TO_SCRAPE
    date_range: Optional[DateRange] = None

    arg_metadata = {
        "num_google_urls": ToolArgMetadata(hidden_from_planner=True),
        "num_news_urls": ToolArgMetadata(hidden_from_planner=True),
    }


@tool(
    description=(
        "This function takes in a list of StockIDs and a single topic which is then appended to the end of each StockID"
        " in the list of StockIDs. Each topic + stockID is then searched for on the web and we end up with the "
        "results. The topic should be text which helps to guide the search towards the client's original prompt, be "
        "sure the topic phrase makes sense to appear after a company name. Examples of this company + topic combo "
        "include Samsung latest news, Apple mobile phone release or Huawei marketplace in the USA. "
        "If specified, the date range will be used to filter out web pages published outside of the date range. "
        "Make sure to use this date range if the user asks for anything date specific. Default to including if not sure. "
        "Unless not specified within a sample plan, always call the summarize_texts tool sometime after this tool. "
        "Again, it is VERY important that the "
        "summarize_texts tool is called before the end of a plan containing this tool! DO not EVER directly output "
        "the returned text from this tool! AGAIN, DO NOT DIRECTLY OUTPUT THE RESULTS OF THIS TOOL!!!"
    ),
    category=ToolCategory.WEB,
    tool_registry=default_tool_registry(),
    enabled_checker_func=enabler_function,
)
async def general_stock_web_search(
    args: GeneralStockWebSearchInput, context: PlanRunContext
) -> List[WebStockText]:
    if not context.user_settings.include_web_results:
        await tool_log("Skipping web search due to user setting", context=context)
        return []

    if len(args.stock_ids) == 0:
        logger.error("No stocks were inputted for latest news search")
        return []

    tasks = []
    for stock_id in args.stock_ids:
        tasks.append(
            single_stock_web_search(
                SingleStockWebSearchInput(
                    stock_id=stock_id,
                    query=f"{stock_id.company_name} {args.topic}",
                    num_google_urls=args.num_google_urls,
                    num_news_urls=args.num_news_urls,
                    date_range=args.date_range,
                ),
                context=context,
            )
        )

    # 100 threads overloads the requests, 25 is a good amount
    results = await gather_with_concurrency(tasks, n=25)
    texts: List[WebStockText] = []
    for result in results:
        texts.extend(result)

    if len(texts) == 0:
        logger.error("Found no web articles for the provided stocks and topic")

    await tool_log(f"Found {len(texts)} results using web search", context=context)
    return texts


class GeneralWebSearchInput(ToolArgs):
    queries: List[str]
    urls: Optional[List[str]] = []
    num_google_urls: int = URLS_TO_SCRAPE
    num_news_urls: int = NEWS_URLS_TO_SCRAPE
    date_range: Optional[DateRange] = None

    arg_metadata = {
        "num_google_urls": ToolArgMetadata(hidden_from_planner=True),
        "num_news_urls": ToolArgMetadata(hidden_from_planner=True),
    }


@tool(
    description=(
        "This function takes in a list of string queries and returns text entries "
        "of the top search results when such queries are made on the web. If a URL is provided, you may use this tool "
        "and treat those URLS as supporting sources. "
        "Place the URLs in a list as an input to this tool and be sure to use the provided URLs as sources. "
        "If specified, the date range will be used to filter out web pages published outside of the date range. "
        "Make sure to use this date range if the user asks for anything date specific. Default to including if not sure. "
        "Unless not specified within a sample plan, always call a text processing tool sometime after this tool. "
        "Again, it is VERY important that a "
        "text processing tool is called before the end of a plan containing this tool! DO not EVER directly output "
        "the returned text from this tool! AGAIN, DO NOT DIRECTLY OUTPUT THE RESULTS OF THIS TOOL!!!"
    ),
    category=ToolCategory.WEB,
    tool_registry=default_tool_registry(),
    enabled_checker_func=enabler_function,
)
async def general_web_search(args: GeneralWebSearchInput, context: PlanRunContext) -> List[WebText]:
    if not context.user_settings.include_web_results:
        await tool_log("Skipping web search due to user setting", context=context)
        return []

    tasks = [
        get_urls_async(
            args.queries, args.num_google_urls, context=context, date_range=args.date_range
        ),
        get_urls_async(
            args.queries,
            args.num_news_urls,
            context=context,
            get_news=True,
            date_range=args.date_range,
        ),
    ]
    results = await gather_with_concurrency(tasks)

    urls: List[str] = [prepend_url_with_https(url) for url in (args.urls or [])]
    for result in results:
        urls.extend(result)

    # currently, this returns a list of WebTexts
    search_results = await get_web_texts_async(
        urls=urls, plan_context=context, date_range=args.date_range
    )
    await tool_log(f"Found {len(search_results)} results using web search", context=context)
    return search_results


class SiteSpecificWebSearchInput(ToolArgs):
    urls: List[str]


@tool(
    description=(
        "This function takes in a list of urls and returns text entries from those urls. "
        "Be sure to include the FULL URL of that is provided, this includes the domain as well as the routing! "
        "Be sure to use all possible valid URLS that are provided from the prompt, if there are multiple provided "
        "and you only use one, you will be FIRED!!! "
        "If the client requests to get information from a specific URLs they input, USE THIS TOOL!!! "
        "Again, it is VERY IMPORTANT that this tool is the web related tool called when URLS are used, other web "
        "tools like general_web_search may be called alongside, but this tool MUST be called with the provided URLs. "
        "You MUST call a text processing tool sometime after this tool. "
        "Again, it is VERY important that a text processing tool is called before the end of a "
        "plan containing this tool! DO not EVER directly output the returned text from this tool! "
        "AGAIN, DO NOT DIRECTLY OUTPUT THE RESULTS OF THIS TOOL!!!"
    ),
    category=ToolCategory.WEB,
    tool_registry=default_tool_registry(),
    enabled_checker_func=enabler_function,
)
async def site_specific_websearch(
    args: SiteSpecificWebSearchInput, context: PlanRunContext
) -> List[WebText]:
    if not context.user_settings.include_web_results:
        await tool_log("Skipping web search due to user setting", context=context)
        return []
    # Extra time since the user specified this exact site.
    urls = [prepend_url_with_https(url) for url in args.urls]
    search_results = await get_web_texts_async(
        urls=urls, plan_context=context, should_print_errors=True, timeout=120
    )
    if len(search_results) == 0:
        raise EmptyOutputError(message="All provided URLs had an error")
    else:
        await tool_log(f"Found {len(search_results)} result(s) using web search", context=context)

    return search_results


async def main() -> None:
    plan_context = PlanRunContext.get_dummy()

    query_1 = "nintendo switch 2 news"
    query_2 = "Australia betting news"
    query_3 = "top legal tech stocks"
    query_4 = "countries which have prominent effects on the stock market"

    queries = GeneralWebSearchInput(queries=[query_1, query_2, query_3, query_4])
    result = await general_web_search(queries, plan_context)

    """
    query = SingleStockWebSearchInput(
        stock_id=StockID(gbi_id=714, symbol="AAPL", isin="US0378331005", company_name=""),
        query="news on apple",
    )

    result = await single_stock_web_search(query, plan_context)
    """
    """
    url_1 = "http://jobs.apple.com/en-ca/search?location=new-york-city-NYC"
    urls = SiteSpecificWebSearchInput(urls=[url_1])
    result = await site_specific_websearch(urls, plan_context)
    """

    print(result)


if __name__ == "__main__":
    asyncio.run(main())
