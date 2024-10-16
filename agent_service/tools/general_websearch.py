import asyncio
import logging
from typing import List, Optional

from agent_service.io_type_utils import ComplexIOBase, io_type
from agent_service.io_types.text import WebText
from agent_service.tool import ToolArgs, ToolCategory, ToolRegistry, tool
from agent_service.tools.product_comparison.brightdata_websearch import (
    brd_websearch,
    get_web_texts_async,
)
from agent_service.types import PlanRunContext
from agent_service.utils.feature_flags import get_ld_flag, get_user_context


@io_type
class WebQuery(ComplexIOBase):
    query: str
    browser: Optional[str] = None
    widget_focus: Optional[str] = None
    top_x_results: Optional[int] = 10


class GeneralWebSearchInput(ToolArgs):
    queries: List[str]


logger = logging.getLogger(__name__)


def enabler_function(user_id: str) -> bool:
    ld_user = get_user_context(user_id)
    result = get_ld_flag("web-search-tool", default=False, user_context=ld_user)
    logger.info(f"Web search tool being used: {result}")
    return result


@tool(
    description=(
        "This function takes in a list WebQuery objects which contain search details and returns text entries "
        "of the top search results when such queries are made on the web."
    ),
    category=ToolCategory.WEB,
    tool_registry=ToolRegistry,
    enabled_checker_func=enabler_function,
)
async def general_web_search(args: GeneralWebSearchInput, context: PlanRunContext) -> List[WebText]:
    queries = args.queries
    search_results = []
    for query in queries:
        if isinstance(query, WebQuery):
            urls = brd_websearch(query.query, 10)
        else:
            urls = brd_websearch(query, 10)

        # currently, this returns a list of WebTexts
        responses = await get_web_texts_async(urls)
        search_results.extend(responses)

    return search_results


async def main() -> None:
    plan_context = PlanRunContext.get_dummy()
    query_1 = (
        "current size of Australia sports betting market Australia sports betting market growth over "
        "past 5 years projected future growth rate of Australia sports betting market"
    )

    queries = GeneralWebSearchInput(queries=[query_1])
    result = await general_web_search(queries, plan_context)
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
