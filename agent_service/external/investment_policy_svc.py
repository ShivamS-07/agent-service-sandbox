import logging
import os
from contextlib import contextmanager
from functools import lru_cache
from typing import Dict, Generator, List, Tuple

from gbi_common_py_utils.utils.environment import DEV_TAG, LOCAL_TAG, PROD_TAG
from grpclib.client import Channel
from pa_portfolio_service_proto_v1.investment_policy_match_pb2 import (
    GetAllStockInvestmentPoliciesRequest,
    GetAllStockInvestmentPoliciesResponse,
    GetMatchScoresForStockInvestmentPoliciesRequest,
    GetMatchScoresForStockInvestmentPoliciesResponse,
    MatchDetails,
)
from pa_portfolio_service_proto_v1.investment_policy_service_grpc import (
    InvestmentPolicyServiceStub,
)
from pa_portfolio_service_proto_v1.well_known_types_pb2 import UUID

from agent_service.external.grpc_utils import get_default_grpc_metadata, grpc_retry
from agent_service.utils.environment import EnvironmentUtils
from agent_service.utils.logs import async_perf_logger

logger = logging.getLogger(__name__)

DEFAULT_URLS = {
    LOCAL_TAG: ("investment-policy-service-dev.boosted.ai", 50051),
    DEV_TAG: ("investment-policy-service-dev.boosted.ai", 50051),
    # NOTE: prod url is useless, can't access
    PROD_TAG: ("investment-policy-service-prod.boosted.ai", 50051),
}


@lru_cache(maxsize=1)
def get_url_and_port() -> Tuple[str, int]:
    url = os.environ.get("INVESTMENT_POLICY_SERVICE_URL")
    if url is not None:
        logger.warning(f"Found Investment Policy service url override: {url}")
        split_url, port = url.split(":")
        return split_url, int(port)

    env = EnvironmentUtils.environment
    if env not in DEFAULT_URLS:
        raise ValueError(f"No URL is set for environment {env}")
    return DEFAULT_URLS[env]


@contextmanager
def _get_service_stub() -> Generator[InvestmentPolicyServiceStub, None, None]:
    try:
        url, port = get_url_and_port()
        channel = Channel(url, port)
        yield InvestmentPolicyServiceStub(channel)
    finally:
        channel.close()


@grpc_retry
@async_perf_logger
async def get_all_stock_investment_policies(user_id: str) -> GetAllStockInvestmentPoliciesResponse:
    with _get_service_stub() as stub:
        response: GetAllStockInvestmentPoliciesResponse = await stub.GetAllStockInvestmentPolicies(
            GetAllStockInvestmentPoliciesRequest(),
            metadata=get_default_grpc_metadata(user_id=user_id),
        )
        if response.status.code != 0:
            raise ValueError(
                f"Failed to get all stock investment policies: {response.status.message}"
            )
        return response


@grpc_retry
@async_perf_logger
async def get_stocks_match_scores_for_ism(
    user_id: str, ism_id: str, gbi_ids: List[int]
) -> Dict[int, MatchDetails]:
    """
    Given a list of gbi_ids, an ism_id, return a dictionary mapping gbi_id to the match details
    which contains some useful fields like `overall_match_score`, `label`, `reason`, etc.
    """
    with _get_service_stub() as stub:
        request = GetMatchScoresForStockInvestmentPoliciesRequest(
            gbi_ids=gbi_ids,
            style_ids=[UUID(id=ism_id)],
            sort_order=GetMatchScoresForStockInvestmentPoliciesRequest.SORT_ORDER_ENUM_BEST,
        )
        response: GetMatchScoresForStockInvestmentPoliciesResponse = (
            await stub.GetMatchScoresForStockInvestmentPolicies(
                request, metadata=get_default_grpc_metadata(user_id=user_id)
            )
        )
        if response.status.code != 0:
            raise ValueError(f"Failed to get match scores: {response.status.message}")

        return {match.gbi_id: match.match_list[0].match_details for match in response.style_matches}
