import asyncio
import io
import json
import logging
import re
import ssl
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
import boto3
import dateparser
import requests
from bs4 import BeautifulSoup
from gbi_common_py_utils.utils.ssm import get_param
from mypy_boto3_s3.client import S3Client

from agent_service.io_types.text import WebText
from agent_service.tools.tool_log import tool_log
from agent_service.tools.web_search.constants import (
    BRIGHTDATA_REQ_PASSWORD,
    BRIGHTDATA_REQ_USERNAME,
    BRIGHTDATA_SERP_PASSWORD,
    BRIGHTDATA_SERP_USERNAME,
    DEFAULT_SERP_PERIOD,
    HEADER,
    JSON_MODIFIED_KEY,
    JSON_PUBLISHED_KEY,
    META_LAST_MODIFIED_KEY,
    META_PUBLISHED_KEY,
    S3_BUCKET_BOOSTED_WEBSEARCH,
    SERP_CACHE_TTL_HOURS,
    SERP_NEWS_TYPE,
    SERP_WEB_TYPE,
    URL_CACHE_TTL_HOURS,
)
from agent_service.types import PlanRunContext
from agent_service.utils.async_db import get_async_db
from agent_service.utils.async_utils import gather_with_concurrency
from agent_service.utils.date_utils import get_now_utc
from agent_service.utils.event_logging import log_event
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

CERTIFICATE_LOCATION = "agent_service/tools/web_search/brd_certificate.crt"

WEB_SEARCH_FETCH_URLS_TIMEOUT = 25
WEB_SEARCH_PULL_SITES_TIMEOUT = 40


async def async_fetch_json(
    session: aiohttp.ClientSession,
    url: str,
    headers: Dict[str, str],
    proxy: str,
    timeout: int,
    ssl_context: Any,
    event_data: Dict[str, Any],
) -> Dict[str, Any]:
    start_time = datetime.now(timezone.utc).isoformat()
    async with session.get(
        url,
        headers=headers,
        proxy=proxy,
        timeout=aiohttp.ClientTimeout(total=timeout),
        ssl=ssl_context,  # type: ignore
    ) as response:
        result = await response.json()
        end_time = datetime.now(timezone.utc).isoformat()
        additional_event_data = {"start_time_utc": start_time, "end_time_utc": end_time}

        if event_data:
            log_event(
                event_name="brd_request",
                event_data={**event_data, **additional_event_data},
            )

        return result


# we need to slice to remove responses from other google widgets like "top stories"
def parse_web_result(search_result: Any, num_results: int) -> List[str]:
    # Google Web search results have the link under "organic"
    items = [item for item in search_result.get("organic", [])[:num_results]]  # type: ignore
    return [item["link"] for item in items]


def parse_news_result(search_result: Any, num_results: int) -> List[str]:
    # Google News results have the link under "news"
    items = [item for item in search_result.get("news", [])[:num_results]]
    return [item["link"] for item in items]


async def get_urls_async(
    queries: List[str],
    num_results: int,
    context: Optional[PlanRunContext] = None,
    headers: Dict[str, str] = HEADER,
    timeout: int = WEB_SEARCH_FETCH_URLS_TIMEOUT,
    log_event_dict: Optional[dict] = None,
    get_news: bool = False,
) -> List[str]:
    if get_news:
        query_type = SERP_NEWS_TYPE
        url_template = "https://www.google.com/search?q={query}&brd_json=1&num={num_results}&tbm=nws&as_qdr=m&hl=en-US"
        parse_urls = parse_news_result
        padded_num_results = num_results
    else:
        query_type = SERP_WEB_TYPE
        url_template = (
            "https://www.google.com/search?q={query}&brd_json=1&num={num_results}&hl=en-US"
        )
        parse_urls = parse_web_result
        padded_num_results = num_results + 3

    ssl_context = ssl.create_default_context()
    ssl_context.load_verify_locations(cafile=CERTIFICATE_LOCATION)
    queries = [query.replace(" ", "+") for query in queries]
    result_urls = set()
    async_db = get_async_db()
    rows = await async_db.get_cached_urls(
        queries, DEFAULT_SERP_PERIOD, query_type, SERP_CACHE_TTL_HOURS
    )

    if len(rows) > 0:
        logger.info(
            f"Cache hit! Found {len(rows)} cached {query_type} search queries out of {len(queries)}!"
        )
    else:
        logger.info(
            f"Cache Miss! Found no cached {query_type} search queries out of {len(queries)}!"
        )

    # add cached results to the total returned results
    cached_query_results = {row["query"]: row["urls"] for row in rows}
    for urls in list(cached_query_results.values()):
        result_urls.update(urls)
    uncached_queries = [query for query in queries if query not in cached_query_results]

    google_urls = [
        url_template.format(query=query, num_results=padded_num_results)
        for query in uncached_queries
    ]

    async with aiohttp.ClientSession(max_line_size=8190 * 5, max_field_size=8190 * 5) as session:
        tasks = []
        for url in google_urls:
            if context:
                event_data = {
                    "agent_id": context.agent_id,
                    "user_id": context.user_id,
                    "plan_id": context.plan_id,
                    "plan_run_id": context.plan_run_id,
                    "request_type": "serp_api",
                    "tool_calling": context.tool_name,
                    "url": url,
                    **(log_event_dict or {}),
                }
            else:
                event_data = {}
            tasks.append(
                async_fetch_json(
                    session, url, headers, proxies["http"], timeout, ssl_context, event_data
                )
            )

        results = await asyncio.gather(*tasks, return_exceptions=True)

    # cache results which were found
    results_to_cache = []
    for result, url, query in zip(results, google_urls, uncached_queries):
        if result is None or isinstance(result, BaseException):
            logger.info(f"skipping bad result for {url=}")
            continue
        try:
            list_of_urls = parse_urls(result, num_results)
            result_urls.update(list_of_urls)
            if list_of_urls and query:
                results_to_cache.append(
                    {
                        "query": query,
                        "urls": list_of_urls,
                        "time_period": DEFAULT_SERP_PERIOD,
                        "query_type": query_type,
                    }
                )
        except requests.JSONDecodeError:
            logger.info(f"JSON decoding error for {url}")

    # Insert results into the cache
    if results_to_cache:
        await async_db.batch_write_cached_urls(results_to_cache)

    return list(result_urls)


req_user = get_param(BRIGHTDATA_REQ_USERNAME)
req_pass = get_param(BRIGHTDATA_REQ_PASSWORD)
req_proxy_url = f"http://{req_user}:{req_pass}@{host}:{port}"
req_proxies = {"http": req_proxy_url, "https": req_proxy_url}


def remove_excess_formatting(text: str) -> str:
    # Remove tabs and replace three or more newlines with two newlines
    text = text.replace("\t", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n[ \t]*", "\n", text)
    return re.sub(r"\n{3,}", "\n\n", text)


@dataclass
class ScrapeResult:
    text: str
    title: Optional[str] = ""
    published_timestamp: Optional[datetime] = None
    modified_timestamp: Optional[datetime] = None


async def scrape_from_pdf(pdf_binary_data: bytes) -> ScrapeResult:
    from PyPDF2 import PdfReader

    pdf_file = io.BytesIO(pdf_binary_data)
    pdf_reader = PdfReader(pdf_file)
    text = ""
    title = ""

    for page in pdf_reader.pages:
        text += page.extract_text()

    if pdf_reader.metadata:
        title = pdf_reader.metadata.title or ""

    return ScrapeResult(title=title, text=text)


def _try_extract_published_timestamp(
    soup: BeautifulSoup,
) -> Tuple[Optional[datetime], Optional[datetime]]:
    # Helper to parse date strings into timezone-aware datetime
    def parse_date(date_string: str) -> Optional[datetime]:
        if not date_string:
            return None
        return dateparser.parse(
            date_string,
            settings={
                "RETURN_AS_TIMEZONE_AWARE": True,  # Ensure timezone awareness
            },
        )

    # Messy GPT-generated code below, TODO clean it up a bit

    published = None
    modified = None

    # 1. Check <meta> tags
    meta_tag = soup.find("meta", attrs={"property": META_PUBLISHED_KEY})
    if meta_tag and meta_tag.get("content"):
        published = parse_date(str(meta_tag["content"]))

    meta_tag = soup.find("meta", attrs={"name": META_LAST_MODIFIED_KEY})
    if meta_tag and meta_tag.get("content"):
        modified = parse_date(str(meta_tag["content"]))

    if published and modified:
        return (published, modified)
    # 2. Check JSON-LD (<script type="application/ld+json">)
    json_ld_tag = soup.find("script", type="application/ld+json")
    if not json_ld_tag:
        return (published, modified)

    try:
        json_data = json.loads(json_ld_tag.string)
    except (json.JSONDecodeError, AttributeError):
        return (published, modified)
    # Handle both single-object and array JSON-LD structures
    if not isinstance(json_data, list):
        json_data = [json_data]

    for item in json_data:
        pub_date = item.get(JSON_PUBLISHED_KEY)
        mod_date = item.get(JSON_MODIFIED_KEY)
        if pub_date and not published:
            published = parse_date(pub_date)
        if mod_date and not modified:
            modified = parse_date(mod_date)

    return (published, modified)


async def scrape_from_http(html_content: str) -> ScrapeResult:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html_content, "html.parser")
    published = None
    modified = None
    try:
        published, modified = _try_extract_published_timestamp(soup)
    except Exception:
        logger.exception("Failed to extract published date with an error!")
    title = ""
    if soup.title:
        title = soup.title.string

    # use the main tag, if not possible, use the whole document
    soup = soup.find("main") or soup

    # Find tags to decompose since they likely contain no relevant text
    tags_to_decompose = soup.find_all(["header", "footer", "button", "nav", "aside"]) or []
    tags_to_decompose.extend(
        soup.find_all(
            True,
            class_=re.compile(r"authentica|button|cookie|metadata|contact|^(\w+-)*form(-\w+)*$"),
        )
        or []
    )
    tags_to_decompose.extend(
        soup.find_all(
            True,
            id=re.compile(
                r"authentica|button|header|footer|cookie|metadata|contact|^(\w+-)*form(-\w+)*$"
            ),
        )
        or []
    )
    for tag in tags_to_decompose:
        tag.decompose()

    text = soup.get_text(" ")
    return ScrapeResult(
        title=title, text=text, published_timestamp=published, modified_timestamp=modified
    )


# retry system!
@async_perf_logger
async def req_and_scrape(
    session: aiohttp.ClientSession,
    s3_client: S3Client,
    url: str,
    headers: Dict[str, str],
    proxy: str,
    timeout: int,
    ssl_context: Any,
    plan_context: PlanRunContext,
    should_print_errors: Optional[bool] = False,
    log_event_dict: Optional[dict] = None,
) -> Optional[WebText]:
    need_retry = False
    scrape_result = ScrapeResult(text="")

    # first try, no proxy, short timeout
    try:
        async with session.get(
            url,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=timeout),
            ssl=ssl_context,  # type: ignore
        ) as response:
            if response.status != 200:
                raise Exception(f"Request Error {response.status}: {url}")

            content_type = response.headers.get("Content-Type", "")
            if "application/pdf" in content_type:
                pdf_binary_data = await response.read()
                scrape_result = await scrape_from_pdf(pdf_binary_data)
            elif "text/html" in content_type:
                html_content = await response.text()
                scrape_result = await scrape_from_http(html_content)
            else:
                logger.info(f"Unsupported content type: {content_type}")
                return None

    except asyncio.TimeoutError as e:
        logger.error(f"TIMEOUT for {url}: {e}, retrying with proxy")
        need_retry = True

    except Exception as e:
        logger.error(f"An error occurred for {url}: {e}, retrying with proxy")
        need_retry = True

    # use brightdata requests on the second time around
    if need_retry:
        start_time = datetime.now(timezone.utc).isoformat()
        event_data = {
            "agent_id": plan_context.agent_id,
            "user_id": plan_context.user_id,
            "plan_id": plan_context.plan_id,
            "plan_run_id": plan_context.plan_run_id,
            "request_type": "web_unlocker",
            "tool_calling": plan_context.tool_name,
            "url": url,
            "start_time_utc": start_time,
            **(log_event_dict or {}),
        }
        try:
            async with session.get(
                url,
                headers=headers,
                proxy=proxy,
                timeout=aiohttp.ClientTimeout(total=timeout),
                ssl=ssl_context,  # type: ignore
            ) as response:
                end_time = datetime.now(timezone.utc).isoformat()
                event_data["end_time_utc"] = end_time

                if response.status != 200:
                    if should_print_errors:
                        await tool_log(
                            f"Failed to scrape from: {url}",
                            context=plan_context,
                        )
                    logger.error(f"Request Error {response.status}: {url}")
                    log_event(
                        event_name="brd_request",
                        event_data={
                            **event_data,
                            **{"error": f"Website response status: {response}"},
                        },
                    )
                    return None

                content_type = response.headers.get("Content-Type", "")
                if "application/pdf" in content_type:
                    pdf_binary_data = await response.read()
                    scrape_result = await scrape_from_pdf(pdf_binary_data)

                    log_event(
                        event_name="brd_request",
                        event_data={
                            **event_data,
                            **{"content_type": content_type},
                        },
                    )
                elif "text/html" in content_type:
                    html_content = await response.text()
                    scrape_result = await scrape_from_http(html_content)

                    log_event(
                        event_name="brd_request",
                        event_data={
                            **event_data,
                            **{"content_type": content_type},
                        },
                    )
                else:
                    logger.info(f"Unsupported content type: {content_type}")
                    log_event(
                        event_name="brd_request",
                        event_data={
                            **event_data,
                            **{"error": f"Unsupported content type: {content_type}"},
                        },
                    )
                    return None

        except asyncio.TimeoutError as e:
            logger.error(f"TIMEOUT for {url}: {e} on retry")
            end_time = datetime.now(timezone.utc).isoformat()
            log_event(
                event_name="brd_request",
                event_data={
                    **event_data,
                    **{"end_time_utc": end_time, "error": "TIMEOUT" + traceback.format_exc()},
                },
            )
            return None

        except aiohttp.ClientResponseError as e:
            if should_print_errors:
                await tool_log(
                    f"Failed to scrape from: {url}",
                    context=plan_context,
                )
            logger.error(f"HTTP error for {url}: {e.status} - {e.message}")
            end_time = datetime.now(timezone.utc).isoformat()
            log_event(
                event_name="brd_request",
                event_data={
                    **event_data,
                    **{"end_time_utc": end_time, "error": traceback.format_exc()},
                },
            )
            return None

        except aiohttp.ClientConnectionError as e:
            if should_print_errors:
                await tool_log(
                    f"Failed to scrape from: {url}",
                    context=plan_context,
                )
            logger.error(f"Connection error for {url}: {e}")
            end_time = datetime.now(timezone.utc).isoformat()
            log_event(
                event_name="brd_request",
                event_data={
                    **event_data,
                    **{"end_time_utc": end_time, "error": traceback.format_exc()},
                },
            )
            return None

        except aiohttp.InvalidURL:
            if should_print_errors:
                await tool_log(
                    f"Failed to scrape from: {url}",
                    context=plan_context,
                )
            logger.error(f"Invalid URL: {url}")
            end_time = datetime.now(timezone.utc).isoformat()
            log_event(
                event_name="brd_request",
                event_data={
                    **event_data,
                    **{"end_time_utc": end_time, "error": traceback.format_exc()},
                },
            )
            return None

        except Exception as e:
            if should_print_errors:
                await tool_log(
                    f"Failed to scrape from: {url}",
                    context=plan_context,
                )
            logger.error(f"An unexpected error occurred for {url}: {e}")
            end_time = datetime.now(timezone.utc).isoformat()
            log_event(
                event_name="brd_request",
                event_data={
                    **event_data,
                    **{"end_time_utc": end_time, "error": traceback.format_exc()},
                },
            )
            return None

    obj = WebText(
        url=url,
        title=scrape_result.title,
        # Scraped timestamp
        timestamp=get_now_utc(),
        # Published timestamp
        published_timestamp=scrape_result.published_timestamp,
        last_modified_timestamp=scrape_result.modified_timestamp,
    )  # we don't save `text` in the obj and pass around

    clean_text = remove_excess_formatting(scrape_result.text)

    await asyncio.to_thread(
        s3_client.upload_fileobj,
        io.BytesIO(clean_text.encode("utf-8")),
        S3_BUCKET_BOOSTED_WEBSEARCH,
        str(obj.id),
    )

    # Return the response object
    return obj


# takes in URLs, returns a list of WebTexts
@async_perf_logger
async def get_web_texts_async(
    urls: List[str],
    plan_context: PlanRunContext,
    headers: Dict[str, str] = HEADER,
    timeout: int = WEB_SEARCH_PULL_SITES_TIMEOUT,
    should_print_errors: bool = False,
    log_event_dict: Optional[dict] = None,
) -> list[WebText]:
    logger = get_prefect_logger(__name__)

    ssl_context = ssl.create_default_context()
    ssl_context.load_verify_locations(cafile=CERTIFICATE_LOCATION)

    sql1 = f"""
        SELECT DISTINCT ON (url) id::VARCHAR, url, title, inserted_at,
          published_timestamp, last_updated_timestamp
        FROM agent.websearch_metadata
        WHERE url = ANY(%(urls)s) AND inserted_at > %(now_utc)s - INTERVAL '{URL_CACHE_TTL_HOURS} hours'
        ORDER BY url, inserted_at DESC
    """
    pg = get_psql()
    params = {"urls": urls, "now_utc": get_now_utc()}

    rows = pg.generic_read(sql1, params=params)

    if len(rows) > 0:
        logger.info(f"Cache hit! Found {len(rows)} cached URLs out of {len(urls)} URLs!")
    else:
        logger.info(f"Cache miss! Found no cached URLs out of {len(urls)} URLs!")

    url_to_obj = {
        row["url"]: WebText(
            id=row["id"],
            url=row["url"],
            title=row["title"],
            timestamp=row["inserted_at"],
            published_timestamp=row["published_timestamp"],
            last_modified_timestamp=row["last_updated_timestamp"],
        )
        for row in rows
    }
    results: list[WebText] = list(url_to_obj.values())
    uncached_urls = [url for url in urls if url not in url_to_obj]
    if uncached_urls:
        logger.info(f"Scrapping {len(uncached_urls)} uncached URLs out of {len(urls)}")
        async with aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(ttl_dns_cache=30),
            max_line_size=8190 * 5,
            max_field_size=8190 * 5,
        ) as session:
            s3_client = boto3.client("s3")
            tasks = [
                req_and_scrape(
                    session,
                    s3_client,
                    url,
                    headers,
                    req_proxies["http"],
                    timeout,
                    ssl_context,
                    plan_context,
                    should_print_errors,
                    log_event_dict,
                )
                for url in uncached_urls
            ]
            uncached_res = await gather_with_concurrency(tasks, n=25, return_exceptions=True)
            uncached_res = [
                res
                for res in uncached_res
                if res is not None and not isinstance(res, BaseException)
            ]

            if uncached_res:
                logger.info(
                    f"Succesfully scrapped {len(uncached_res)} results "
                    f"out of {len(uncached_urls)} URLs. Inserting metadata into DB"
                )

                results.extend(uncached_res)

                # Once the contents are on S3, insert the metadata into the database
                sql2 = """
                    INSERT INTO agent.websearch_metadata
                      (id, url, title, published_timestamp, last_updated_timestamp)
                    VALUES (
                        %(id)s, %(url)s, %(title)s,
                        %(published_timestamp)s, %(last_updated_timestamp)s
                    )
                """
                with pg.transaction_cursor() as cursor:
                    cursor.executemany(sql2, [res.to_db_dict() for res in uncached_res])

    return results


async def main() -> None:
    urls = [
        "jobs.apple.com/en-ca/search?location=new-york-city-NYC",
    ]
    context = PlanRunContext.get_dummy()

    responses = await get_web_texts_async(urls, context)

    """    
    query_1 = "nintendo switch 2"
    # query_2 = "Australia betting"

    # responses = await get_urls_async([query_1], 10)
    responses = await get_urls_async([query_1], 10, get_news=True)
    """

    print(responses)


if __name__ == "__main__":
    asyncio.run(main())
