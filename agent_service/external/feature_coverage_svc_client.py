import asyncio
import logging
import os
from functools import lru_cache
from typing import Any, Dict, List, Optional

import aiohttp
import backoff
from gbi_common_py_utils.utils.environment import (
    DEV_TAG,
    LOCAL_TAG,
    get_environment_tag,
)
from pydantic import BaseModel

from agent_service.external.utils import get_http_session
from agent_service.utils.logs import async_perf_logger

logger = logging.getLogger(__name__)

# Note: Please use port forwarding to run this locally
# kubectl port-forward -n {namespace} svc/feature-coverage-service 8000:8000
# namespace - insights-backend-dev/insights-backend-prod
DEFAULT_URLS = {
    LOCAL_TAG: "http://localhost:8000",
    DEV_TAG: "http://localhost:8000",
}

FEATURE_COVERAGE_SERVICE_CLIENT = None

# Retry configuration
MAX_RETRIES = 3
RETRY_INTERVAL = 1


class FeatureCoverageServiceError(Exception):
    """Base exception for Feature Coverage Service errors"""

    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        response_content: Optional[Dict[Any, Any]] = None,
        request_details: Optional[Dict[str, Any]] = None,
    ):
        self.message = message
        self.status_code = status_code
        self.response_content = response_content
        self.request_details = request_details
        super().__init__(self.message)

    def __str__(self) -> str:
        error_parts = [f"Feature Coverage Service Error: {self.message}"]
        if self.status_code:
            error_parts.append(f"Status Code: {self.status_code}")
        if self.request_details:
            error_parts.append(f"Request Details: {self.request_details}")
        if self.response_content:
            error_parts.append(f"Response Content: {self.response_content}")
        return " | ".join(error_parts)


class FeatureCoverageItem(BaseModel):
    feature_id: str
    coverage: float


class FeatureCoverageResponse(BaseModel):
    features: List[FeatureCoverageItem]


@lru_cache(maxsize=1)
def get_base_url() -> str:
    url = os.environ.get("FEATURE_COVERAGE_SERVICE_URL")
    if url is not None:
        logger.warning(f"Found Feature Coverage service url override: {url}")
        return url

    env = get_environment_tag()
    if env not in DEFAULT_URLS:
        raise ValueError(f"No URL is set for environment {env}")
    return DEFAULT_URLS[env]


class FeatureCoverageServiceClient:
    def __init__(self) -> None:
        self.url_base = get_base_url()
        self.session = get_http_session()

    @backoff.on_exception(
        backoff.constant,
        (aiohttp.ClientError, asyncio.TimeoutError),
        interval=RETRY_INTERVAL,
        max_tries=MAX_RETRIES,
    )
    async def _post(self, endpoint: str, body: dict) -> aiohttp.ClientResponse:
        return await self.session.post(self.url_base + endpoint, json=body)

    @async_perf_logger
    async def get_latest_universe_coverage(
        self, universe_id: str, feature_ids: List[str]
    ) -> FeatureCoverageResponse:
        request_body = {
            "universeId": universe_id,
            "featureIds": feature_ids,
        }

        response: aiohttp.ClientResponse = await self._post(
            "/api/latest-universe-coverage-agent", request_body
        )

        if response.content_type != "application/json":
            raw_text = await response.text()
            logger.warning(
                f"Feature Coverage Service get_latest_universe_coverage unexpected response {raw_text}"
            )
            raise FeatureCoverageServiceError(
                f"Response type was not JSON, instead it was {response.content_type}"
            )

        response_content: Dict[str, Any] = await response.json()

        if response.status != 200:
            error_message = (
                f"Failed to get universe coverage: "
                f"{response_content.get('message')} {response_content.get('detail')}"
            )
            logger.warning(error_message)
            raise FeatureCoverageServiceError(
                error_message, status_code=response.status, response_content=response_content
            )

        return FeatureCoverageResponse(
            features=[
                FeatureCoverageItem(feature_id=item["featureId"], coverage=item["coverage"])
                for item in response_content["features"]
            ]
        )


def get_feature_coverage_client() -> FeatureCoverageServiceClient:
    global FEATURE_COVERAGE_SERVICE_CLIENT
    if FEATURE_COVERAGE_SERVICE_CLIENT is None:
        FEATURE_COVERAGE_SERVICE_CLIENT = FeatureCoverageServiceClient()
    return FEATURE_COVERAGE_SERVICE_CLIENT
