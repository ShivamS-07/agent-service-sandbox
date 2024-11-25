import asyncio
import logging
from typing import List, Optional

from agent_service.io_types.stock import StockID
from agent_service.io_types.text import WebStockText, WebText
from agent_service.planner.errors import EmptyOutputError
from agent_service.tool import ToolArgs, ToolCategory, ToolRegistry, tool
from agent_service.tools.product_comparison.brightdata_websearch import (
    get_news_urls_async,
    get_urls_async,
    get_web_texts_async,
)
from agent_service.tools.tool_log import tool_log
from agent_service.types import PlanRunContext
from agent_service.utils.async_utils import gather_with_concurrency
from agent_service.utils.feature_flags import get_ld_flag

URLS_TO_SCRAPE = 5
NEWS_URLS_TO_SCRAPE = 10

logger = logging.getLogger(__name__)


def prepend_url_with_https(url: str) -> str:
    if not url.startswith("https://") and not url.startswith("http://"):
        return "https://" + url
    return url


def enabler_function(user_id: Optional[str]) -> bool:
    result = get_ld_flag("web-search-tool", default=False, user_context=user_id)
    logger.info(f"Web search tool being used: {result}")
    return result


class SingleStockWebSearchInput(ToolArgs):
    stock_id: StockID
    query: str


@tool(
    description=(
        "This function takes in a StockID and a single query which contain search details and "
        "returns text entries of the top search results when the query is made on the web. "
        "Unless not specified within a sample plan, always call the summarize_texts tool sometime after this tool. "
        "Again, it is VERY important that the "
        "summarize_texts tool is called before the end of a plan containing this tool! DO not EVER directly output "
        "the returned text from this tool! AGAIN, DO NOT DIRECTLY OUTPUT THE RESULTS OF THIS TOOL!!!"
    ),
    category=ToolCategory.WEB,
    tool_registry=ToolRegistry,
    enabled_checker_func=enabler_function,
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
        get_urls_async([query], URLS_TO_SCRAPE, context=context),
        get_news_urls_async([query], NEWS_URLS_TO_SCRAPE, context=context),
    ]
    results = await gather_with_concurrency(tasks)

    urls: List[str] = []
    for result in results:
        urls.extend(result)

    search_results = await get_web_texts_async(urls=urls, plan_context=context)
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


class GeneralWebSearchInput(ToolArgs):
    queries: List[str]
    urls: Optional[List[str]] = []


@tool(
    description=(
        "This function takes in a list of string queries and returns text entries "
        "of the top search results when such queries are made on the web. If a URL is provided, you may use this tool "
        "and treat those URLS as supporting sources. "
        "Place the URLs in a list as an input to this tool and be sure to use the provided URLs as sources. "
        "Unless not specified within a sample plan, always call the summarize_texts tool sometime after this tool. "
        "Again, it is VERY important that the "
        "summarize_texts tool is called before the end of a plan containing this tool! DO not EVER directly output "
        "the returned text from this tool! AGAIN, DO NOT DIRECTLY OUTPUT THE RESULTS OF THIS TOOL!!!"
    ),
    category=ToolCategory.WEB,
    tool_registry=ToolRegistry,
    enabled_checker_func=enabler_function,
)
async def general_web_search(args: GeneralWebSearchInput, context: PlanRunContext) -> List[WebText]:
    if not context.user_settings.include_web_results:
        await tool_log("Skipping web search due to user setting", context=context)
        return []
    tasks = [
        get_urls_async(args.queries, URLS_TO_SCRAPE, context=context),
        get_news_urls_async(args.queries, NEWS_URLS_TO_SCRAPE, context=context),
    ]
    results = await gather_with_concurrency(tasks)

    urls: List[str] = [prepend_url_with_https(url) for url in (args.urls or [])]
    for result in results:
        urls.extend(result)

    # currently, this returns a list of WebTexts
    search_results = await get_web_texts_async(urls=urls, plan_context=context)
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
        "You MUST call the summarize_texts tool sometime after this tool. "
        "Again, it is VERY important that the summarize_texts tool is called before the end of a "
        "plan containing this tool! DO not EVER directly output the returned text from this tool! "
        "AGAIN, DO NOT DIRECTLY OUTPUT THE RESULTS OF THIS TOOL!!!"
    ),
    category=ToolCategory.WEB,
    tool_registry=ToolRegistry,
    enabled_checker_func=enabler_function,
)
async def site_specific_websearch(
    args: SiteSpecificWebSearchInput, context: PlanRunContext
) -> List[WebText]:
    if not context.user_settings.include_web_results:
        await tool_log("Skipping web search due to user setting", context=context)
        return []
    urls = [prepend_url_with_https(url) for url in args.urls]
    search_results = await get_web_texts_async(
        urls=urls, plan_context=context, should_print_errors=True
    )
    if len(search_results) == 0:
        raise EmptyOutputError(message="All provided URLs had an error")

    return search_results


async def main() -> None:
    plan_context = PlanRunContext.get_dummy()

    """
    query_1 = "nintendo switch 2 news"
    query_2 = "Australia betting news"
    query_3 = "top legal tech stocks"
    query_4 = "countries which have prominent effects on the stock market"

    queries = GeneralWebSearchInput(queries=[query_1, query_2])
    result = await general_web_search(queries, plan_context)
    """
    """
    query = SingleStockWebSearchInput(
        stock_id=StockID(gbi_id=714, symbol="AAPL", isin="US0378331005", company_name=""),
        query="news on apple",
    )

    result = await single_stock_web_search(query, plan_context)
    """
    url_1 = "http://jobs.apple.com/en-ca/search?location=new-york-city-NYC"
    urls = SiteSpecificWebSearchInput(urls=[url_1])
    result = await site_specific_websearch(urls, plan_context)

    print(result)


if __name__ == "__main__":
    asyncio.run(main())
