import asyncio
import io
import logging
import re
import ssl
from typing import Any, Dict, List, Optional

import aioboto3
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
    S3_BUCKET_BOOSTED_WEBSEARCH,
    URL_CACHE_TTL_HOURS,
)
from agent_service.utils.logs import async_perf_logger
from agent_service.utils.postgres import get_psql
from agent_service.utils.prefect import get_prefect_logger

logger = logging.getLogger(__name__)

# should probably move these values elsewhere
host = "brd.superproxy.io"
port = 22225

username = get_param(BRIGHTDATA_SERP_USERNAME)
password = get_param(BRIGHTDATA_SERP_PASSWORD)
proxy_url = f"http://{username}:{password}@{host}:{port}"
proxies = {"http": proxy_url, "https": proxy_url}


async def async_fetch_json(
    session: aiohttp.ClientSession,
    url: str,
    headers: Dict[str, str],
    proxy: str,
    timeout: int,
    ssl_context: Any,
) -> Dict[str, Any]:
    async with session.get(
        url, headers=headers, proxy=proxy, timeout=timeout, ssl=ssl_context  # type: ignore
    ) as response:
        return await response.json()


async def get_urls_async(
    queries: List[str], num_results: int, headers: Dict[str, str] = HEADER, timeout: int = 60
) -> List[Any]:
    ssl_context = ssl.create_default_context()
    ssl_context.load_verify_locations(
        cafile="agent_service/tools/product_comparison/brd_certificate.crt"
    )

    queries = [query.replace(" ", "+") for query in queries]
    urls = [
        f"https://www.google.com/search?q={query}&brd_json=1&num={num_results}" for query in queries
    ]
    async with aiohttp.ClientSession(max_line_size=8190 * 5, max_field_size=8190 * 5) as session:
        tasks = []
        for url in urls:
            tasks.append(
                async_fetch_json(session, url, headers, req_proxies["http"], timeout, ssl_context)
            )

        results = await asyncio.gather(*tasks)
        results = [result for result in results if result is not None]

    result_urls = set()
    for result in results:
        try:
            # this line parses the returned result from Brightdata's SERP service to get the URLS to search
            items = [item for item in result.get("organic", [])]  # type: ignore
            result_urls.update([item["link"] for item in items])
        except requests.JSONDecodeError:
            logger.info(f"JSON decoding error for {url}")

    return list(result_urls)


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


def remove_excess_formatting(text: str) -> str:
    # Remove tabs and replace three or more newlines with two newlines
    text = text.replace("\t", "")
    return re.sub(r"\n{3,}", "\n\n", text)


@async_perf_logger
async def req_and_scrape(
    session: aiohttp.ClientSession,
    s3_client: Any,
    url: str,
    headers: Dict[str, str],
    proxy: str,
    timeout: int,
    ssl_context: Any,
) -> Optional[WebText]:
    # use aiohttp to asynchronously make URL requests and scrape using BeautifulSoup
    try:
        async with session.get(
            url, headers=headers, proxy=proxy, timeout=timeout, ssl=ssl_context  # type: ignore
        ) as response:
            title: Optional[str] = None
            content_type = response.headers.get("Content-Type", "")
            if "application/pdf" in content_type:
                pdf_binary_data = await response.read()
                pdf_file = io.BytesIO(pdf_binary_data)
                pdf_reader = PdfReader(pdf_file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text()

                if pdf_reader.metadata:
                    title = pdf_reader.metadata.title
            elif "text/html" in content_type:
                html_content = await response.text()
                soup = BeautifulSoup(html_content, "html.parser")

                if soup.title:
                    title = soup.title.string

                # find all main content divs
                main_contents = soup.find_all(name="div", attrs={"class": "content"})
                if main_contents:
                    # remove script, style, aside, footer tags
                    cleaned_texts = []
                    for main_content in main_contents:
                        for tag in main_content(["script", "style", "aside", "footer"]):
                            tag.decompose()
                        cleaned_text = main_content.get_text()
                        cleaned_texts.append(cleaned_text)
                    text = "\n\n".join(cleaned_texts)
                else:
                    # fallback to extracting all text
                    text = soup.get_text()
            else:
                logger.info(f"Unsupported content type: {content_type}")
                return None

    except aiohttp.ClientResponseError as e:
        logger.info(f"HTTP error for {url}: {e.status} - {e.message}")
        return None

    except aiohttp.ClientConnectionError as e:
        logger.info(f"Connection error for {url}: {e}")
        return None

    except Exception as e:
        logger.info(f"An unexpected error occurred: {e}")
        return None

    obj = WebText(url=url, title=title)  # we don't save `text` in the obj and pass around
    clean_text = remove_excess_formatting(text)

    # upload to s3
    await s3_client.upload_fileobj(
        Fileobj=io.BytesIO(clean_text.encode("utf-8")),
        Bucket=S3_BUCKET_BOOSTED_WEBSEARCH,
        Key=obj.id,
    )

    # Return the response object
    return obj


# takes in URLs, returns a list of WebTexts
@async_perf_logger
async def get_web_texts_async(
    urls: List[str], headers: Dict[str, str] = HEADER, timeout: int = 300
) -> Any:
    logger = get_prefect_logger(__name__)

    ssl_context = ssl.create_default_context()
    ssl_context.load_verify_locations(
        cafile="agent_service/tools/product_comparison/brd_certificate.crt"
    )

    sql1 = """
        SELECT DISTINCT ON (url) id::VARCHAR, url, title
        FROM agent.websearch_metadata
        WHERE url = ANY(%(urls)s) AND inserted_at > NOW() - INTERVAL '%(num_hours)s hours'
        ORDER BY url, inserted_at DESC
    """
    pg = get_psql()
    rows = pg.generic_read(sql1, params={"urls": urls, "num_hours": URL_CACHE_TTL_HOURS})

    logger.info(f"Cache hit! Found {len(rows)} cached URLs out of {len(urls)}!")

    url_to_obj = {
        row["url"]: WebText(id=row["id"], url=row["url"], title=row["title"]) for row in rows
    }
    results: list[WebText] = list(url_to_obj.values())

    uncached_urls = [url for url in urls if url not in url_to_obj]
    if uncached_urls:
        logger.info(f"Scrapping {len(uncached_urls)} uncached URLs out of {len(urls)}")

        # Create a ClientSession with proxy support
        async with aiohttp.ClientSession(
            max_line_size=8190 * 5, max_field_size=8190 * 5
        ) as session, aioboto3.Session().client("s3") as s3_client:
            tasks = [
                req_and_scrape(
                    session, s3_client, url, headers, req_proxies["http"], timeout, ssl_context
                )
                for url in uncached_urls
            ]
            uncached_res = await asyncio.gather(*tasks)
            uncached_res = [res for res in uncached_res if res is not None]

            if uncached_res:
                logger.info(
                    f"Succesfully scrapped {len(uncached_res)} results "
                    f"out of {len(uncached_urls)} URLs. Inserting metadata into DB"
                )

                results.extend(uncached_res)

                # Once the contents are on S3, insert the metadata into the database
                sql2 = """
                    INSERT INTO agent.websearch_metadata (id, url, title)
                    VALUES (
                        %(id)s, %(url)s, %(title)s
                    )
                """
                with pg.transaction_cursor() as cursor:
                    cursor.executemany(sql2, [res.to_db_dict() for res in uncached_res])

    return results


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
