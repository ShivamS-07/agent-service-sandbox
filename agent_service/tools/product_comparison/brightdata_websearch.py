from typing import List

import requests
from gbi_common_py_utils.utils.ssm import get_param

from agent_service.tools.product_comparison.constants import (
    BRIGHTDATA_PASSWORD,
    BRIGHTDATA_USERNAME,
)

# should probably move these values elsewhere
host = "brd.superproxy.io"
port = 22225

username = get_param(BRIGHTDATA_USERNAME)
password = get_param(BRIGHTDATA_PASSWORD)

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
