import logging
from typing import Any, Dict, List

import requests
from gbi_common_py_utils.utils.ssm import get_param

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
