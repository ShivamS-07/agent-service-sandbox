import asyncio
import logging
from typing import List, Optional

from agent_service.io_type_utils import ComplexIOBase, io_type
from agent_service.io_types.text import WebText
from agent_service.tool import ToolArgs, ToolCategory, ToolRegistry, tool
from agent_service.tools.product_comparison.brightdata_websearch import (
    get_urls_async,
    get_web_texts_async,
)
from agent_service.types import PlanRunContext
from agent_service.utils.feature_flags import get_ld_flag

URLS_TO_SCRAPE = 10


@io_type
class WebQuery(ComplexIOBase):
    query: str
    browser: Optional[str] = None
    widget_focus: Optional[str] = None
    top_x_results: Optional[int] = URLS_TO_SCRAPE


class GeneralWebSearchInput(ToolArgs):
    queries: List[str]


logger = logging.getLogger(__name__)


def enabler_function(user_id: Optional[str]) -> bool:
    result = get_ld_flag("web-search-tool", default=False, user_context=user_id)
    logger.info(f"Web search tool being used: {result}")
    return result


@tool(
    description=(
        "This function takes in a list WebQuery objects which contain search details and returns text entries "
        "of the top search results when such queries are made on the web. Unless not specified within a sample plan,"
        "always call the summarize_texts tool sometime after this tool. Again, it is VERY important that the "
        "summarize_texts tool is called before the end of a plan containing this tool! DO not EVER directly output "
        "the returned text from this tool! AGAIN, DO NOT DIRECTLY OUTPUT THE RESULTS OF THIS TOOL!!!"
    ),
    category=ToolCategory.WEB,
    tool_registry=ToolRegistry,
    enabled_checker_func=enabler_function,
)
async def general_web_search(args: GeneralWebSearchInput, context: PlanRunContext) -> List[WebText]:
    queries = []
    for query in args.queries:
        if isinstance(query, WebQuery):
            queries.append(query.query)
        else:
            queries.append(query)

    urls = await get_urls_async(queries, URLS_TO_SCRAPE)

    # currently, this returns a list of WebTexts
    search_results = await get_web_texts_async(urls)
    return search_results


async def main() -> None:
    plan_context = PlanRunContext.get_dummy()
    query_1 = "nintendo switch 2 news"
    query_2 = "Australia betting news"
    query_3 = "top legal tech stocks"
    query_4 = "countries which have prominent effects on the stock market"

    queries = GeneralWebSearchInput(queries=[query_1, query_2, query_3, query_4])
    result = await general_web_search(queries, plan_context)
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
