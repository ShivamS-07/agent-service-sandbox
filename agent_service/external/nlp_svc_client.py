import datetime
import logging
import os
from contextlib import contextmanager
from functools import lru_cache
from typing import Generator, List, Optional, Tuple

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
from nlp_service_proto_v1.earnings_impacts_pb2 import (
    EventInfo,
    GenerateEarningsCallFromEventsRequest,
    GenerateEarningsCallFromEventsResponse,
    GetEarningsCallEventsRequest,
    GetEarningsCallEventsResponse,
    GetEarningsCallTranscriptsRequest,
    GetEarningsCallTranscriptsResponse,
    GetEarningsPeersAffectedByStocksRequest,
    GetEarningsPeersAffectedByStocksResponse,
    GetEarningsPeersAffectingStocksRequest,
    GetEarningsPeersAffectingStocksResponse,
    GetLatestEarningsCallEventsRequest,
    GetLatestEarningsCallEventsResponse,
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
from nlp_service_proto_v1.well_known_types_pb2 import UUID

from agent_service.external.grpc_utils import (
    date_to_timestamp,
    get_default_grpc_metadata,
    grpc_retry,
)
from agent_service.tools.portfolio import PortfolioID
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
    portfolio_id: Optional[PortfolioID],
) -> GetCommentaryTopicsResponse:
    with _get_service_stub() as stub:
        response: GetCommentaryTopicsResponse = await stub.GetCommentaryTopics(
            GetCommentaryTopicsRequest(
                section_types=section_types,
                date_range=date_range,
                number_per_section=number_per_section,
                portfolio_id=UUID(id=portfolio_id),
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


@grpc_retry
@async_perf_logger
async def get_earnings_peers_impacted_by_stocks(
    user_id: str, gbi_ids: List[int]
) -> GetEarningsPeersAffectedByStocksResponse:
    with _get_service_stub() as stub:
        response: GetEarningsPeersAffectedByStocksResponse = (
            await stub.GetEarningsPeersAffectedByStocks(
                GetEarningsPeersAffectedByStocksRequest(gbi_ids=gbi_ids),
                metadata=get_default_grpc_metadata(user_id=user_id),
            )
        )
        if response.status.code != 0:
            raise ValueError(f"Failed to get peers for: {gbi_ids}")
        return response


@grpc_retry
@async_perf_logger
async def get_earnings_peers_impacting_stocks(
    user_id: str, gbi_ids: List[int]
) -> GetEarningsPeersAffectingStocksResponse:
    with _get_service_stub() as stub:
        request = GetEarningsPeersAffectingStocksRequest(gbi_ids=gbi_ids)
        response: GetEarningsPeersAffectingStocksResponse = (
            await stub.GetEarningsPeersAffectingStocks(
                request, metadata=get_default_grpc_metadata(user_id=user_id)
            )
        )
        if response.status.code != 0:
            raise ValueError(
                f"Failed to get Earnings Peers affecting Companies: {response.status.message}"
            )
        return response


@grpc_retry
@async_perf_logger
async def get_earnings_call_events(
    user_id: str, gbi_ids: List[int], start_date: datetime.date, end_date: datetime.date
) -> GetEarningsCallEventsResponse:
    start_timestamp = date_to_timestamp(start_date)
    end_timestamp = date_to_timestamp(end_date)
    with _get_service_stub() as stub:
        request = GetEarningsCallEventsRequest(
            gbi_ids=gbi_ids, start_date=start_timestamp, end_date=end_timestamp
        )
        response: GetEarningsCallEventsResponse = await stub.GetEarningCallEvents(
            request, metadata=get_default_grpc_metadata(user_id=user_id)
        )
        if response.status.code != 0:
            raise ValueError(f"Failed to get Earnings Call Events: {response.status.message}")
        return response


@grpc_retry
@async_perf_logger
async def get_latest_earnings_call_events(
    user_id: str, gbi_ids: List[int]
) -> GetLatestEarningsCallEventsResponse:
    with _get_service_stub() as stub:
        request = GetLatestEarningsCallEventsRequest(gbi_ids=gbi_ids)
        response: GetLatestEarningsCallEventsResponse = await stub.GetLatestEarningCallEvents(
            request, metadata=get_default_grpc_metadata(user_id=user_id)
        )
        if response.status.code != 0:
            raise ValueError(f"Failed to get Earnings Call Events: {response.status.message}")
        return response


@grpc_retry
@async_perf_logger
async def get_earnings_call_transcripts(
    user_id: str, events: List[EventInfo]
) -> GetEarningsCallTranscriptsResponse:
    with _get_service_stub() as stub:
        request = GetEarningsCallTranscriptsRequest(earnings_event_info=events)
        response: GetEarningsCallTranscriptsResponse = await stub.GetEarningCallTranscripts(
            request, metadata=get_default_grpc_metadata(user_id=user_id)
        )
        if response.status.code != 0:
            raise ValueError(f"Failed to get Earning Transcripts: {response.status.message}")
        return response


@grpc_retry
@async_perf_logger
async def get_earnings_call_summaries_with_real_time_gen(
    user_id: str, events: List[EventInfo]
) -> GenerateEarningsCallFromEventsResponse:
    with _get_service_stub() as stub:
        request = GenerateEarningsCallFromEventsRequest(earnings_event_info=events)
        response: GenerateEarningsCallFromEventsResponse = (
            await stub.GenerateEarningsCallFromEvents(
                request, metadata=get_default_grpc_metadata(user_id=user_id)
            )
        )
        if response.status.code != 0:
            raise ValueError(f"Failed to get Earning Transcripts: {response.status.message}")
        return response
