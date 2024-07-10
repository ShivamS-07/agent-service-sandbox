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
from pa_portfolio_service_proto_v1.backtest_data_service_grpc import (
    BacktestDataServiceStub,
)
from pa_portfolio_service_proto_v1.backtest_data_service_pb2 import (
    GetThemesWithImpactedStocksRequest,
    GetThemesWithImpactedStocksResponse,
    GetUniverseSectorPerformanceForDateRangeRequest,
    GetUniverseSectorPerformanceForDateRangeResponse,
    SectorPerformance,
    ThemeWithImpactedStocks,
    UniverseStockFactorExposuresRequest,
    UniverseStockFactorExposuresResponse,
)
from pa_portfolio_service_proto_v1.proto_sheet_pb2 import ProtoSheet
from pa_portfolio_service_proto_v1.well_known_types_pb2 import UUID
from pa_portfolio_service_proto_v1.workspace_pb2 import StockAndWeight

from agent_service.external.grpc_utils import get_default_grpc_metadata, grpc_retry
from agent_service.utils.date_utils import date_to_pb_timestamp
from agent_service.utils.logs import async_perf_logger

logger = logging.getLogger(__name__)

DEFAULT_URLS = {
    LOCAL_TAG: ("pa-portfolio-backtest-service-dev.boosted.ai", 50051),
    DEV_TAG: ("pa-portfolio-backtest-service-dev.boosted.ai", 50051),
    PROD_TAG: ("pa-portfolio-backtest-service-prod.boosted.ai", 50051),
}


@lru_cache(maxsize=1)
def get_url_and_port() -> Tuple[str, int]:
    url = os.environ.get("PA_PORTFOLIO_BACKTEST_SERVICE_URL")
    if url is not None:
        logger.warning(f"Found PA Portfolio Backtest service url override: {url}")
        split_url, port = url.split(":")
        return split_url, int(port)

    env = get_environment_tag()
    if env not in DEFAULT_URLS:
        raise ValueError(f"No URL is set for environment {env}")
    return DEFAULT_URLS[env]


@contextmanager
def _get_service_stub() -> Generator[BacktestDataServiceStub, None, None]:
    try:
        url, port = get_url_and_port()
        channel = Channel(url, port)
        yield BacktestDataServiceStub(channel)
    finally:
        channel.close()


@grpc_retry
@async_perf_logger
async def universe_stock_factor_exposures(
    user_id: str,
    universe_id: str,
    gbi_ids: Optional[List[int]] = None,
    factors: Optional[List[str]] = None,
    risk_model_id: Optional[int] = None,
) -> ProtoSheet:
    with _get_service_stub() as stub:
        response: UniverseStockFactorExposuresResponse = await stub.UniverseStockFactorExposures(
            UniverseStockFactorExposuresRequest(
                universe_id=UUID(id=universe_id),
                gbi_ids=gbi_ids,
                factors=factors,
                risk_model_id=risk_model_id,
            ),
            metadata=get_default_grpc_metadata(user_id=user_id),
        )
        if response.status.code != 0:
            raise ValueError(
                f"Failed to get watchlist stocks: {response.status.code} {response.status.message}"
            )
        return response.exposures


@grpc_retry
@async_perf_logger
async def get_themes_with_impacted_stocks(
    stocks: List[StockAndWeight], start_date: datetime.date, end_date: datetime.date, user_id: str
) -> List[ThemeWithImpactedStocks]:
    req = GetThemesWithImpactedStocksRequest(
        stocks=stocks,
        start_date=date_to_pb_timestamp(start_date),
        end_date=date_to_pb_timestamp(end_date),
    )
    with _get_service_stub() as stub:
        resp: GetThemesWithImpactedStocksResponse = await stub.GetThemesWithImpactedStocks(
            req, metadata=get_default_grpc_metadata(user_id=user_id)
        )
    return list(resp.data)


@grpc_retry
@async_perf_logger
async def get_universe_sector_performance_for_date_range(
    start_date: datetime.date, end_date: datetime.date, stock_universe_id: str, user_id: str
) -> List[SectorPerformance]:
    req = GetUniverseSectorPerformanceForDateRangeRequest(
        start_date=date_to_pb_timestamp(start_date),
        end_date=date_to_pb_timestamp(end_date),
        stock_universe_id=UUID(id=stock_universe_id),
    )
    with _get_service_stub() as stub:
        resp: GetUniverseSectorPerformanceForDateRangeResponse = (
            await stub.GetUniverseSectorPerformanceForDateRange(
                req, metadata=get_default_grpc_metadata(user_id=user_id)
            )
        )

    return list(resp.sector_performance_list)
