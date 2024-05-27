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
from nlp_service_proto_v1.commentary_pb2 import (
    GetCommentaryTopicsRequest,
    GetCommentaryTopicsResponse,
)
from nlp_service_proto_v1.news_pb2 import (
    NEWS_DELTA_HORIZON_3M,
    GetMultiCompaniesNewsTopicsRequest,
    GetMultiCompaniesNewsTopicsResponse,
)
from nlp_service_proto_v1.nlp_grpc import NLPServiceStub
from nlp_service_proto_v1.themes_pb2 import (
    GetAllThemesForUserRequest,
    GetAllThemesForUserResponse,
    GetSecurityThemesRequest,
    GetSecurityThemesResponse,
)

from agent_service.external.grpc_utils import get_default_grpc_metadata, grpc_retry
from agent_service.utils.logs import async_perf_logger

logger = logging.getLogger(__name__)

DEFAULT_URLS = {
    LOCAL_TAG: ("nlp-service-dev.boosted.ai", 50051),
    DEV_TAG: ("nlp-service-dev.boosted.ai", 50051),
    PROD_TAG: ("nlp-service-prod.boosted.ai", 50051),
}


@lru_cache(maxsize=1)
def get_url_and_port() -> Tuple[str, int]:
    url = os.environ.get("NLP_SERVICE_URL")
    if url is not None:
        logger.warning(f"Found NLP service url override: {url}")
        split_url, port = url.split(":")
        return split_url, int(port)

    env = get_environment_tag()
    if env not in DEFAULT_URLS:
        raise ValueError(f"No URL is set for environment {env}")
    return DEFAULT_URLS[env]


@contextmanager
def _get_service_stub() -> Generator[NLPServiceStub, None, None]:
    try:
        url, port = get_url_and_port()
        channel = Channel(url, port)
        yield NLPServiceStub(channel)
    finally:
        channel.close()


@grpc_retry
@async_perf_logger
async def get_multi_companies_news_topics(
    user_id: str,
    gbi_ids: List[int],
) -> GetMultiCompaniesNewsTopicsResponse:
    # TODO for now just get everything, filter later
    horizon = NEWS_DELTA_HORIZON_3M
    with _get_service_stub() as stub:
        response: GetMultiCompaniesNewsTopicsResponse = await stub.GetMultiCompaniesNewsTopics(
            GetMultiCompaniesNewsTopicsRequest(gbi_ids=gbi_ids, horizon=horizon),
            metadata=get_default_grpc_metadata(user_id=user_id),
        )
        if response.status.code != 0:
            raise ValueError(
                f"Failed to get multi companies news sentiments: {response.status.message}"
            )
        return response


@grpc_retry
@async_perf_logger
async def get_security_themes(user_id: str, gbi_ids: List[int]) -> GetSecurityThemesResponse:
    with _get_service_stub() as stub:
        response: GetSecurityThemesResponse = await stub.GetSecurityThemes(
            GetSecurityThemesRequest(gbi_ids=gbi_ids),
            metadata=get_default_grpc_metadata(user_id=user_id),
        )
        if response.status.code != 0:
            raise ValueError(f"Failed to get security themes: {response.status.message}")
        return response


@grpc_retry
@async_perf_logger
async def get_top_themes(
    user_id: str,
    section_types: List[str],
    date_range: str,
    number_per_section: int,
) -> GetCommentaryTopicsResponse:
    with _get_service_stub() as stub:
        response: GetCommentaryTopicsResponse = await stub.GetCommentaryTopics(
            GetCommentaryTopicsRequest(
                section_types=section_types,
                date_range=date_range,
                number_per_section=number_per_section,
            ),
            metadata=get_default_grpc_metadata(user_id=user_id),
        )
        if response.status.code != 0:
            raise ValueError(f"Failed to get commentary topics: {response.status.message}")
        return response


@grpc_retry
@async_perf_logger
async def get_all_themes_for_user(user_id: str) -> GetAllThemesForUserResponse:
    with _get_service_stub() as stub:
        response: GetAllThemesForUserResponse = await stub.GetAllThemesForUser(
            GetAllThemesForUserRequest(),
            metadata=get_default_grpc_metadata(user_id=user_id),
        )
        if response.status.code != 0:
            raise ValueError(f"Failed to get all themes for user: {response.status.message}")
        return response
