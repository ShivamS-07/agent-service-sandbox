import asyncio
import io
import logging
import ssl
from typing import Any, Dict, List, Optional

import aiohttp
import requests
from bs4 import BeautifulSoup
from gbi_common_py_utils.utils.ssm import get_param
from PyPDF2 import PdfReader

from agent_service.io_types.text import WebText
from agent_service.tools.product_comparison.constants import (
    BRIGHTDATA_REQ_PASSWORD,
    BRIGHTDATA_REQ_USERNAME,
    BRIGHTDATA_SERP_PASSWORD,
    BRIGHTDATA_SERP_USERNAME,
    HEADER,
)

logger = logging.getLogger(__name__)

# should probably move these values elsewhere
host = "brd.superproxy.io"
port = 22225

username = get_param(BRIGHTDATA_SERP_USERNAME)
password = get_param(BRIGHTDATA_SERP_PASSWORD)
proxy_url = f"http://{username}:{password}@{host}:{port}"
proxies = {"http": proxy_url, "https": proxy_url}


def brd_websearch(query: str, num_results: int) -> List[str]:
    query = query.replace(" ", "+")
    url = f"https://www.google.com/search?q={query}&brd_json=1&num={num_results}"
    response = requests.get(
        url, proxies=proxies, verify="agent_service/tools/product_comparison/brd_certificate.crt"
    )

    try:
        data = response.json()
        items = [item for item in data.get("organic", [])]
        urls = [item["link"] for item in items]
        return urls
    except requests.JSONDecodeError:
        return []


req_user = get_param(BRIGHTDATA_REQ_USERNAME)
req_pass = get_param(BRIGHTDATA_REQ_PASSWORD)
req_proxy_url = f"http://{req_user}:{req_pass}@{host}:{port}"
req_proxies = {"http": req_proxy_url, "https": req_proxy_url}


def brd_request(url: str, headers: Dict[str, str] = HEADER, timeout: int = 5) -> Any:
    response = requests.get(
        url,
        proxies=req_proxies,
        timeout=timeout,
        headers=headers,
        verify="agent_service/tools/product_comparison/brd_certificate.crt",
    )

    if response.status_code == 400:
        logger.info(f"Failed to retrieve {url} using brightdata. Trying regular request")
        response = requests.get(url, timeout=timeout, headers=headers)

    return response


# use aiohttp to asynchronously make URL requests and scrape using BeautifulSoup
async def req_and_scrape(
    session: Any, url: str, headers: Dict[str, str], proxy: str, timeout: int, ssl_context: Any
) -> Optional[WebText]:
    async with session.get(
        url, headers=headers, proxy=proxy, timeout=timeout, ssl=ssl_context
    ) as response:
        try:
            content_type = response.headers.get("Content-Type")
            if "application/pdf" in content_type:
                pdf_binary_data = await response.read()
                pdf_file = io.BytesIO(pdf_binary_data)
                pdf_reader = PdfReader(pdf_file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text()
                title = url
            elif "text/html" in content_type:
                html_content = await response.text()
                soup = BeautifulSoup(html_content, "html.parser")
                text = soup.get_text()
                title = soup.title.string if soup.title else url
            else:
                logger.info(f"Unsupported content type: {content_type}")
                return None

        except requests.exceptions.HTTPError as e:
            logger.info(
                f"Failed to retrieve {url}. HTTPError: {e.response.status} - {e.response.reason}"
            )
            return None
        except requests.exceptions.RequestException as e:
            if e.response:
                logger.info(f"URLError: {e.response.reason}")
            return None
        except TimeoutError as e:
            logger.info(f"TimeoutError: {e}")
            return None
        except Exception as e:
            logger.info(f"Failed to retrieve {url}. Error: {e}")
            return None

        # Return the response object
        return WebText(val=text, url=url, title=title)


# takes in URLs, returns a list of WebTexts
async def get_web_texts_async(
    urls: List[str], headers: Dict[str, str] = HEADER, timeout: int = 60
) -> Any:
    ssl_context = ssl.create_default_context()
    ssl_context.load_verify_locations(
        cafile="agent_service/tools/product_comparison/brd_certificate.crt"
    )

    # Create a ClientSession with proxy support
    async with aiohttp.ClientSession(max_line_size=8190 * 5, max_field_size=8190 * 5) as session:
        tasks = []
        for url in urls:
            tasks.append(
                req_and_scrape(session, url, headers, req_proxies["http"], timeout, ssl_context)
            )

        results = await asyncio.gather(*tasks)
        return [result for result in results if result is not None]


async def main() -> None:
    urls = [
        "https://www.apple.com/ca/iphone-16-pro/",
        "https://www.gsmarena.com/apple_iphone_16-13317.php",
        "https://www.macrumors.com/roundup/iphone-16/",
        "https://www.tomsguide.com/phones/iphones/apple-iphone-16-review",
        "https://www.phonearena.com/phones/Apple-iPhone-16_id12240",
    ]

    responses = await get_web_texts_async(urls)
    print(responses)


if __name__ == "__main__":
    asyncio.run(main())
