import json
import logging
import os
from contextlib import contextmanager
from functools import lru_cache
from typing import Any, Dict, Generator, Optional, Tuple

from discover_service_proto_v1.disco_service_grpc import DiscoverServiceStub
from discover_service_proto_v1.discover_pb2 import (
    GetTemporaryDiscoverBlockDataRequest,
    GetTemporaryDiscoverBlockDataResponse,
)
from discover_service_proto_v1.other_messages_pb2 import (
    DiscoverDeltaHorizonEnum,
    DiscoverRecommendationCategory,
    DiscoverRecommendationHorizon,
)
from gbi_common_py_utils.utils.environment import (
    DEV_TAG,
    LOCAL_TAG,
    PROD_TAG,
    get_environment_tag,
)
from grpclib.client import Channel

from agent_service.external.grpc_utils import get_default_grpc_metadata, grpc_retry
from agent_service.utils.logs import async_perf_logger

logger = logging.getLogger(__name__)

DEFAULT_URLS = {
    LOCAL_TAG: ("discover-service-dev.boosted.ai", 50051),
    DEV_TAG: ("discover-service-dev.boosted.ai", 50051),
    PROD_TAG: ("discover-service-prod.boosted.ai", 50051),
}


@lru_cache(maxsize=1)
def get_url_and_port() -> Tuple[str, int]:
    url = os.environ.get("DISCOVER_SERVICE_URL")
    if url is not None:
        logger.warning(f"Found Discover service url override: {url}")
        split_url, port = url.split(":")
        return split_url, int(port)

    env = get_environment_tag()
    if env not in DEFAULT_URLS:
        raise ValueError(f"No URL is set for environment {env}")
    return DEFAULT_URLS[env]


@contextmanager
def _get_service_stub() -> Generator[DiscoverServiceStub, None, None]:
    try:
        url, port = get_url_and_port()
        channel = Channel(url, port)
        yield DiscoverServiceStub(channel)
    finally:
        channel.close()


def horizon_from_str(horizon: Optional[str] = None) -> DiscoverRecommendationHorizon:
    if horizon == "3M":
        return DiscoverRecommendationHorizon.DISCOVER_RECOMMENDATION_HORIZON_3M
    elif horizon == "1Y":
        return DiscoverRecommendationHorizon.DISCOVER_RECOMMENDATION_HORIZON_1Y

    return DiscoverRecommendationHorizon.DISCOVER_RECOMMENDATION_HORIZON_1M


def delta_horizon_from_str(delta_horizon: Optional[str] = None) -> DiscoverDeltaHorizonEnum:
    if delta_horizon == "1M":
        return DiscoverDeltaHorizonEnum.DISCOVER_DELTA_HORIZON_1M
    elif delta_horizon == "3M":
        return DiscoverDeltaHorizonEnum.DISCOVER_DELTA_HORIZON_3M
    elif delta_horizon == "6M":
        return DiscoverDeltaHorizonEnum.DISCOVER_DELTA_HORIZON_6M
    elif delta_horizon == "9M":
        return DiscoverDeltaHorizonEnum.DISCOVER_DELTA_HORIZON_9M
    elif delta_horizon == "1Y":
        return DiscoverDeltaHorizonEnum.DISCOVER_DELTA_HORIZON_1Y

    return DiscoverDeltaHorizonEnum.DISCOVER_DELTA_HORIZON_1W


def get_score_from_recommendation(rec: DiscoverRecommendationCategory) -> Optional[float]:
    """
    Given a recommendation category, return a float from 0 to 1.
    """
    if rec == DiscoverRecommendationCategory.DISCOVER_RECOMMENDATION_CATEGORY_STRONG_BUY:
        return 1.0
    elif rec == DiscoverRecommendationCategory.DISCOVER_RECOMMENDATION_CATEGORY_BUY:
        return 0.75
    elif rec == DiscoverRecommendationCategory.DISCOVER_RECOMMENDATION_CATEGORY_HOLD:
        return 0.5
    elif rec == DiscoverRecommendationCategory.DISCOVER_RECOMMENDATION_CATEGORY_SELL:
        return 0.25
    elif rec == DiscoverRecommendationCategory.DISCOVER_RECOMMENDATION_CATEGORY_STRONG_SELL:
        return 0.0
    return None


@grpc_retry
@async_perf_logger
async def get_temporary_discover_block_data(
    user_id: str, settings_blob: Dict[str, Any], horizon: str, delta_horizon: str
) -> GetTemporaryDiscoverBlockDataResponse:
    with _get_service_stub() as stub:
        response: GetTemporaryDiscoverBlockDataResponse = await stub.GetTemporaryDiscoverBlockData(
            GetTemporaryDiscoverBlockDataRequest(
                settings_blob=json.dumps(settings_blob),
                horizon=horizon_from_str(horizon=horizon),
                delta_horizon=delta_horizon_from_str(delta_horizon=delta_horizon),
            ),
            metadata=get_default_grpc_metadata(user_id=user_id),
        )
        if response.status.code != 0:
            raise ValueError(
                f"Failed to get temporary discover block data: {response.status.code} {response.status.message}"  # noqa
            )
        return response
