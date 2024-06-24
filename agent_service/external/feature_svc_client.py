import datetime
import logging
import os
from contextlib import contextmanager
from functools import lru_cache
from typing import Generator, List, Optional, Tuple

from feature_service_proto_v1.feature_metadata_service_grpc import (
    FeatureMetadataServiceStub,
)
from feature_service_proto_v1.feature_metadata_service_pb2 import (
    GetAllFeaturesMetadataRequest,
    GetAllFeaturesMetadataResponse,
)
from feature_service_proto_v1.feature_service_grpc import FeatureServiceStub
from feature_service_proto_v1.feature_service_pb2 import (
    GetFeatureDataRequest,
    GetFeatureDataResponse,
    TimeAxis,
)
from gbi_common_py_utils.utils.environment import (
    DEV_TAG,
    LOCAL_TAG,
    PROD_TAG,
    get_environment_tag,
)
from grpclib.client import Channel

from agent_service.external.grpc_utils import (
    date_to_timestamp,
    get_default_grpc_metadata,
    grpc_retry,
)
from agent_service.utils.logs import async_perf_logger

logger = logging.getLogger(__name__)

DEFAULT_URLS = {
    LOCAL_TAG: ("feature-service-dev.boosted.ai", 50051),
    DEV_TAG: ("feature-service-dev.boosted.ai", 50051),
    PROD_TAG: ("feature-service-prod.boosted.ai", 50051),
}

METADATA_DEFAULT_URLS = {
    LOCAL_TAG: ("feature-metadata-service-dev.boosted.ai", 50051),
    DEV_TAG: ("feature-metadata-service-dev.boosted.ai", 50051),
    PROD_TAG: ("feature-metadata-service-prod.boosted.ai", 50051),
}


@lru_cache(maxsize=1)
def get_url_and_port() -> Tuple[str, int]:
    url = os.environ.get("FEATURE_SERVICE_URL")
    if url is not None:
        logger.warning(f"Found FEATURE_SERVICE_URL override: {url}")
        split_url, port = url.split(":")
        return split_url, int(port)

    env = get_environment_tag()
    if env not in DEFAULT_URLS:
        raise ValueError(f"No URL is set for environment {env}")
    return DEFAULT_URLS[env]


@lru_cache(maxsize=1)
def get_metadata_url_and_port() -> Tuple[str, int]:
    url = os.environ.get("FEATURE_METADATA_SERVICE_URL")
    if url is not None:
        logger.warning(f"Found FEATURE_METADATA_SERVICE_URL override: {url}")
        split_url, port = url.split(":")
        return split_url, int(port)

    env = get_environment_tag()
    if env not in METADATA_DEFAULT_URLS:
        raise ValueError(f"No URL is set for environment {env}")
    return METADATA_DEFAULT_URLS[env]


@contextmanager
def _get_service_stub() -> Generator[FeatureServiceStub, None, None]:
    try:
        url, port = get_url_and_port()
        channel = Channel(url, port)
        yield FeatureServiceStub(channel)
    finally:
        channel.close()


@contextmanager
def _get_metadata_service_stub() -> Generator[FeatureMetadataServiceStub, None, None]:
    try:
        url, port = get_metadata_url_and_port()
        channel = Channel(url, port)
        yield FeatureMetadataServiceStub(channel)
    finally:
        channel.close()


@grpc_retry
@async_perf_logger
async def get_feature_data(
    user_id: str,
    statistic_ids: List[str],
    stock_ids: List[int],
    from_date: datetime.date,
    to_date: datetime.date,
    target_currency: Optional[str] = None,
    ffill_days: int = 0,
    use_natural_axis: bool = False,
) -> GetFeatureDataResponse:
    """
    Optional Params Description:

    target_currency: currency to convert all outputs to. Otherwise outputs are
        converted to the currency of the first stock in the list.
    ffill_days: number of days to forward fill missing data.
    use_natural_axis: if True, feature service will return axes in quarters (YYYYQ)
        or Fiscal Quarters (FYYYYQ) instead of dates as axes depending on the best
        display format of the specific statistic (feature).
    """
    with _get_service_stub() as stub:
        req = GetFeatureDataRequest(
            feature_ids=statistic_ids,
            gbi_ids=stock_ids,
            from_time=date_to_timestamp(from_date),
            to_time=date_to_timestamp(to_date),
            iso_currency=target_currency,
            forward_fill_days=ffill_days,
            prefer_time_axis=(
                TimeAxis.TIME_AXIS_BEST_HUMAN_READABLE
                if use_natural_axis
                else TimeAxis.TIME_AXIS_DATE
            ),
        )
        resp: GetFeatureDataResponse = await stub.GetFeatureData(
            req,
            metadata=get_default_grpc_metadata(user_id=user_id),
        )
        return resp


@grpc_retry
@async_perf_logger
async def get_all_features_metadata(
    user_id: str, filtered: bool = False
) -> GetAllFeaturesMetadataResponse:
    with _get_metadata_service_stub() as stub:
        req = GetAllFeaturesMetadataRequest(filter_agent_enabled=filtered)
        resp: GetAllFeaturesMetadataResponse = await stub.GetAllFeaturesMetadata(
            req,
            metadata=get_default_grpc_metadata(user_id=user_id),
        )
        return resp
