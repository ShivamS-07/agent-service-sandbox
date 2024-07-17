import logging
import os
from datetime import datetime
from functools import lru_cache
from typing import Any, Dict, List, Optional

import aiohttp
from gbi_common_py_utils.utils.environment import (
    DEV_TAG,
    LOCAL_TAG,
    PROD_TAG,
    get_environment_tag,
)
from pydantic import BaseModel

logger = logging.getLogger(__name__)

DEFAULT_URLS = {
    LOCAL_TAG: "http://dal-dev.boosted.ai",
    DEV_TAG: "http://dal-dev.boosted.ai",
    PROD_TAG: "http://dal-prod.boosted.ai",
}
DAL_SERVICE_CLIENT = None


class SecurityRow(BaseModel):
    gbi_id: int
    weight: Optional[float]
    date: Optional[datetime]


class ParsePortfolioWorkspaceResponse(BaseModel):
    securities: List[SecurityRow]
    parse_failures: List[str]
    parse_warnings: List[str]


@lru_cache(maxsize=1)
def get_base_url() -> str:
    url = os.environ.get("DAL_URL")
    if url is not None:
        logger.warning(f"Found DAL service url override: {url}")
        return url

    env = get_environment_tag()
    if env not in DEFAULT_URLS:
        raise ValueError(f"No URL is set for environment {env}")
    return DEFAULT_URLS[env]


class DALServiceClient:
    def __init__(self) -> None:
        self.url_base = get_base_url()
        self.session = aiohttp.ClientSession()

    async def _post(self, endpoint: str, body: dict) -> aiohttp.ClientResponse:
        return await self.session.post(self.url_base + endpoint, json=body)

    async def parse_file(self, b64data: str, content_type: str) -> ParsePortfolioWorkspaceResponse:
        response: aiohttp.ClientResponse = await self._post(
            "/api/v0/portfolio-workspace/parse_file",
            {
                "file_bytes_base64": b64data,
                "content_type": content_type,
            },
        )

        response_content: Dict[str, Any] = await response.json()

        if response.status != 200:
            logger.warn(
                f"Failed to parse file from DAL: {response_content.get('message')} {response_content.get('detail')}"
            )

        return ParsePortfolioWorkspaceResponse(
            securities=[
                SecurityRow(
                    gbi_id=security_row["gbi_id"],
                    weight=security_row.get("weight"),
                    date=(
                        datetime.strptime(security_row["date"], "%Y-%m-%d")
                        if "date" in security_row
                        else None
                    ),
                )
                for security_row in response_content.get("securities", [])
            ],
            parse_failures=response_content.get("parse_failures", []),
            parse_warnings=response_content.get("parse_warnings", []),
        )


def get_dal_client() -> DALServiceClient:
    global DAL_SERVICE_CLIENT
    if DAL_SERVICE_CLIENT is None:
        DAL_SERVICE_CLIENT = DALServiceClient()
    return DAL_SERVICE_CLIENT
