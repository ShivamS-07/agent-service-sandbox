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
from stock_universe_service_proto_v1.security_metadata_service_grpc import (
    SecurityMetadataServiceStub,
)
from stock_universe_service_proto_v1.security_metadata_service_pb2 import (
    GetAllEtfSectorsRequest,
    GetAllEtfSectorsResponse,
    GetEtfHoldingsForDateRequest,
    GetEtfHoldingsForDateResponse,
)

from agent_service.external.grpc_utils import get_default_grpc_metadata, grpc_retry
from agent_service.utils.logs import async_perf_logger

logger = logging.getLogger(__name__)

DEFAULT_URLS = {
    LOCAL_TAG: ("security-metadata-service-dev.boosted.ai", 50051),
    DEV_TAG: ("security-metadata-service-dev.boosted.ai", 50051),
    PROD_TAG: ("security-metadata-service-prod.boosted.ai", 50051),
}


@lru_cache(maxsize=1)
def get_url_and_port() -> Tuple[str, int]:
    url = os.environ.get("SECURITY_METADATA_SERVICE_URL")
    if url is not None:
        logger.warning(f"Found SECURITY_METADATA_SERVICE_URL override: {url}")
        split_url, port = url.split(":")
        return split_url, int(port)

    env = get_environment_tag()
    if env not in DEFAULT_URLS:
        raise ValueError(f"No URL is set for environment {env}")
    return DEFAULT_URLS[env]


@contextmanager
def _get_service_stub() -> Generator[SecurityMetadataServiceStub, None, None]:
    try:
        url, port = get_url_and_port()
        channel = Channel(url, port)
        yield SecurityMetadataServiceStub(channel)
    finally:
        channel.close()


@grpc_retry
@async_perf_logger
async def get_etf_holdings(etf_id: int, user_id: str) -> GetEtfHoldingsForDateResponse:
    with _get_service_stub() as stub:
        req = GetEtfHoldingsForDateRequest(
            gbi_ids=[etf_id],
            # asof_date = date_to_timestamp(asof_date)
        )
        resp: GetEtfHoldingsForDateResponse = await stub.GetEtfHoldingsForDate(
            req,
            metadata=get_default_grpc_metadata(user_id=user_id),
        )
        # if resp.status.code != 0:
        #    raise HTTPException(
        #        status_code=500,
        #        detail=f"Failed to get etf holdings: {resp.status.message}",
        #    )
        return resp


@grpc_retry
@async_perf_logger
async def get_all_etf_sectors(gbi_ids: List[int], user_id: str) -> GetAllEtfSectorsResponse:
    with _get_service_stub() as stub:
        req = GetAllEtfSectorsRequest(
            gbi_ids=gbi_ids,
        )
        resp: GetAllEtfSectorsResponse = await stub.GetAllEtfSectors(
            req,
            metadata=get_default_grpc_metadata(user_id=user_id),
        )
        return resp
