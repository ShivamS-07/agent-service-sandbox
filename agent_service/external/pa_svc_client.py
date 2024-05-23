import logging
import os
from contextlib import contextmanager
from functools import lru_cache
from typing import Generator, List, Tuple

from gbi_common_py_utils.utils.environment import (
    DEV_TAG,
    LOCAL_TAG,
    PROD_TAG,
    get_environment_tag,
)
from grpclib.client import Channel
from pa_portfolio_service_proto_v1.pa_service_grpc import PAServiceStub
from pa_portfolio_service_proto_v1.watchlist_pb2 import (
    GetAllStocksInAllWatchlistsRequest,
    GetAllStocksInAllWatchlistsResponse,
    GetAllWatchlistsRequest,
    GetAllWatchlistsResponse,
    GetWatchlistStocksAndWeightsRequest,
    GetWatchlistStocksAndWeightsResponse,
)
from pa_portfolio_service_proto_v1.well_known_types_pb2 import UUID

from agent_service.external.grpc_utils import get_default_grpc_metadata, grpc_retry
from agent_service.utils.logs import async_perf_logger

logger = logging.getLogger(__name__)

DEFAULT_URLS = {
    LOCAL_TAG: ("pa-portfolio-service-dev.boosted.ai", 50051),
    DEV_TAG: ("pa-portfolio-service-dev.boosted.ai", 50051),
    PROD_TAG: ("pa-portfolio-service-prod.boosted.ai", 50051),
}


@lru_cache(maxsize=1)
def get_url_and_port() -> Tuple[str, int]:
    url = os.environ.get("PA_PORTFOLIO_SERVICE_URL")
    if url is not None:
        logger.warning(f"Found PA service url override: {url}")
        split_url, port = url.split(":")
        return split_url, int(port)

    env = get_environment_tag()
    if env not in DEFAULT_URLS:
        raise ValueError(f"No URL is set for environment {env}")
    return DEFAULT_URLS[env]


@contextmanager
def _get_service_stub() -> Generator[PAServiceStub, None, None]:
    try:
        url, port = get_url_and_port()
        channel = Channel(url, port)
        yield PAServiceStub(channel)
    finally:
        channel.close()


@grpc_retry
@async_perf_logger
async def get_all_watchlists(user_id: str) -> GetAllWatchlistsResponse:
    with _get_service_stub() as stub:
        response: GetAllWatchlistsResponse = await stub.GetAllWatchlists(
            GetAllWatchlistsRequest(), metadata=get_default_grpc_metadata(user_id=user_id)
        )
        if response.status.code != 0:
            raise ValueError(
                f"Failed to get all watchlists: {response.status.code} {response.status.message}"
            )
        return response


@grpc_retry
@async_perf_logger
async def get_watchlist_stocks(user_id: str, watchlist_id: str) -> List[int]:
    with _get_service_stub() as stub:
        response: GetWatchlistStocksAndWeightsResponse = await stub.GetWatchlistStocksAndWeights(
            GetWatchlistStocksAndWeightsRequest(watchlist_id=UUID(id=watchlist_id)),
            metadata=get_default_grpc_metadata(user_id=user_id),
        )
        if response.status.code != 0:
            raise ValueError(
                f"Failed to get watchlist stocks: {response.status.code} {response.status.message}"
            )
        return [stock.gbi_id for stock in response.weighted_stocks]


@grpc_retry
@async_perf_logger
async def get_all_stocks_in_all_watchlists(user_id: str) -> List[int]:
    with _get_service_stub() as stub:
        response: GetAllStocksInAllWatchlistsResponse = await stub.GetAllStocksInAllWatchlists(
            GetAllStocksInAllWatchlistsRequest(),
            metadata=get_default_grpc_metadata(user_id=user_id),
        )
        if response.status.code != 0:
            raise ValueError(
                f"Failed to get all stocks in all watchlists: {response.status.code}"
                f" {response.status.message}"
            )
        return list(response.gbi_ids)