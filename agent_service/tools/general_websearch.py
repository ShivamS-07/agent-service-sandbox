import asyncio
import io
import logging
from typing import List, Optional

import requests
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader

from agent_service.io_type_utils import ComplexIOBase, io_type
from agent_service.io_types.text import Text
from agent_service.tool import ToolArgs, ToolCategory, ToolRegistry, tool
from agent_service.tools.product_comparison.brightdata_websearch import (
    brd_request,
    brd_websearch,
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
    queries: List[str] | List[WebQuery]


@io_type
class WebResultText(Text):
    pass


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
async def general_web_search(
    args: GeneralWebSearchInput, context: PlanRunContext
) -> List[WebResultText]:
    queries = args.queries
    search_results = []
    for query in queries:
        if isinstance(query, WebQuery):
            urls = brd_websearch(query.query, 10)
        else:
            urls = brd_websearch(query, 10)

        # parse
        parsed_results = []
        for url in urls:
            try:
                response = brd_request(url)
                content_type = response.headers.get("Content-Type")
                if "application/pdf" in content_type:
                    pdf_binary_data = response.content
                    pdf_file = io.BytesIO(pdf_binary_data)
                    pdf_reader = PdfReader(pdf_file)
                    text = ""
                    for page in pdf_reader.pages:
                        text += page.extract_text()
                    title = url
                elif "text/html" in content_type:
                    html_content = response.text
                    soup = BeautifulSoup(html_content, "html.parser")
                    text = soup.get_text()
                    title = soup.title.string if soup.title else url
                else:
                    logger.info(f"Unsupported content type: {content_type}")
                    continue
            except requests.exceptions.HTTPError as e:
                logger.info(
                    f"Failed to retrieve {url}. HTTPError: {e.response.status_code} - {e.response.reason}"
                )
                continue
            except requests.exceptions.RequestException as e:
                if e.response:
                    logger.info(f"URLError: {e.response.reason}")
                continue
            except TimeoutError as e:
                logger.info(f"TimeoutError: {e}")
                continue
            except Exception as e:
                logger.info(f"Failed to retrieve {url}. Error: {e}")
                continue

            parsed_results.append({"title": title, "text": text, "url": url})
            search_results.append(WebResultText(val=text))

    return search_results


async def main() -> None:
    plan_context = PlanRunContext.get_dummy()
    query_1 = "Nvidia latest news"

    queries = GeneralWebSearchInput(queries=[WebQuery(query=query_1)])
    result = await general_web_search(queries, plan_context)
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
