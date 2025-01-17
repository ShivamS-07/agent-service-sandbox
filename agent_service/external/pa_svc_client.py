import logging
import os
from contextlib import contextmanager
from functools import lru_cache
from typing import Generator, List, Optional, Tuple

from cache import AsyncTTL
from gbi_common_py_utils.utils.environment import (
    DEV_TAG,
    LOCAL_TAG,
    PROD_TAG,
    get_environment_tag,
)
from grpclib.client import Channel
from pa_portfolio_service_proto_v1.marketplace_messages_pb2 import (
    GetAllAuthorizedLiveStrategyIdsRequest,
    GetAllAuthorizedLiveStrategyIdsResponse,
    GetFullStrategiesInfoRequest,
    GetFullStrategiesInfoResponse,
    ListAllAuthorizedStrategiesRequest,
    ListAllAuthorizedStrategiesResponse,
    StrategyIdentifier,
    SubscribeToMarketplaceStrategyRequest,
)
from pa_portfolio_service_proto_v1.pa_service_common_messages_pb2 import PORTFOLIO_AUTH_ENUM_READ
from pa_portfolio_service_proto_v1.pa_service_grpc import PAServiceStub
from pa_portfolio_service_proto_v1.portfolio_crud_actions_pb2 import (
    RecalcStrategiesRequest,
    RecalcStrategiesResponse,
)
from pa_portfolio_service_proto_v1.watchlist_pb2 import (
    DeleteWatchlistRequest,
    DeleteWatchlistResponse,
    GetAllStocksInAllWatchlistsRequest,
    GetAllStocksInAllWatchlistsResponse,
    GetAllWatchlistsRequest,
    GetAllWatchlistsResponse,
    GetWatchlistStocksAndWeightsRequest,
    GetWatchlistStocksAndWeightsResponse,
    RenameWatchlistRequest,
    RenameWatchlistResponse,
)
from pa_portfolio_service_proto_v1.well_known_types_pb2 import UUID, StockHolding
from pa_portfolio_service_proto_v1.workspace_pb2 import (
    CreateTSWorkspaceRequest,
    CreateTSWorkspaceResponse,
    DeleteWorkspaceRequest,
    DeleteWorkspaceResponse,
    GetAllWorkspacesRequest,
    GetAllWorkspacesResponse,
    GetTransitiveHoldingsFromStocksAndWeightsRequest,
    GetTransitiveHoldingsFromStocksAndWeightsResponse,
    GetTSWorkspacesHoldingsRequest,
    GetTSWorkspacesHoldingsResponse,
    ModifyWorkspaceHistoricalHoldingsRequest,
    ModifyWorkspaceHistoricalHoldingsResponse,
    RenameWorkspaceRequest,
    RenameWorkspaceResponse,
    StockAndWeight,
    WorkspaceAuth,
    WorkspaceMetadata,
    WorkspaceTrade,
)

from agent_service.external.grpc_utils import get_default_grpc_metadata, grpc_retry
from agent_service.utils.logs import async_perf_logger

ONE_MINUTE = 60

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
    svc_target = DEFAULT_URLS[env]
    logger.warning(f"using pa portfolio svc at: {svc_target=}")
    return svc_target


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
async def rename_watchlist(user_id: str, watchlist_id: str, name: str) -> bool:
    with _get_service_stub() as stub:
        response: RenameWatchlistResponse = await stub.RenameWatchlist(
            RenameWatchlistRequest(watchlist_id=UUID(id=watchlist_id), name=name),
            metadata=get_default_grpc_metadata(user_id=user_id),
        )
        if response.status.code != 0:
            raise ValueError(
                f"Failed to rename watchlist {watchlist_id}: {response.status.code}"
                f" {response.status.message}"
            )
        return True


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


@grpc_retry
@async_perf_logger
async def delete_watchlist(user_id: str, watchlist_id: str) -> bool:
    with _get_service_stub() as stub:
        response: DeleteWatchlistResponse = await stub.DeleteWatchlist(
            DeleteWatchlistRequest(watchlist_id=UUID(id=watchlist_id)),
            metadata=get_default_grpc_metadata(user_id=user_id),
        )
        if response.status.code != 0:
            raise ValueError(
                f"Failed to delete watchlist: {response.status.code} {response.status.message}"
            )
        return True


@grpc_retry
@async_perf_logger
async def delete_workspace(user_id: str, workspace_id: str) -> bool:
    with _get_service_stub() as stub:
        response: DeleteWorkspaceResponse = await stub.DeleteWorkspace(
            DeleteWorkspaceRequest(workspace_id=UUID(id=workspace_id)),
            metadata=get_default_grpc_metadata(user_id=user_id),
        )
        if response.status.code != 0:
            raise ValueError(
                f"Failed to delete workspace: {response.status.code} {response.status.message}"
            )
        return True


@grpc_retry
@async_perf_logger
async def get_all_holdings_in_workspace(
    user_id: str, workspace_id: str
) -> GetTSWorkspacesHoldingsResponse.WorkspaceToHoldings:
    with _get_service_stub() as stub:
        response: GetTSWorkspacesHoldingsResponse = await stub.GetTSWorkspacesHoldings(
            GetTSWorkspacesHoldingsRequest(workspace_ids=[UUID(id=workspace_id)]),
            metadata=get_default_grpc_metadata(user_id=user_id),
        )
        if response.status.code != 0:
            raise ValueError(
                f"Failed to get all stocks in a workspace: {response.status.code}"
                f" {response.status.message}"
            )
        return response.workspaceStocks[0]


@AsyncTTL(time_to_live=ONE_MINUTE, maxsize=10)
@grpc_retry
@async_perf_logger
async def get_all_workspaces(
    user_id: str, workspace_ids: Optional[List[str]] = None
) -> List[WorkspaceMetadata]:
    # just need this cached long enough to call it again in the same plan
    with _get_service_stub() as stub:
        response: GetAllWorkspacesResponse = await stub.GetAllWorkspaces(
            GetAllWorkspacesRequest(
                workspace_ids=(
                    None
                    if not workspace_ids
                    else [UUID(id=workspace_id) for workspace_id in workspace_ids]
                ),
                min_auth_level=WorkspaceAuth.WORKSPACE_AUTH_READ,
            ),
            metadata=get_default_grpc_metadata(user_id=user_id),
        )
        if response.status.code != 0:
            raise ValueError(
                f"Failed to get all workspaces for the user: {response.status.code}"
                f" {response.status.message}"
            )

    return [workspace for workspace in response.workspaces]


@grpc_retry
@async_perf_logger
async def get_full_strategy_info(
    user_id: str, workspace_id: str
) -> GetFullStrategiesInfoResponse.FullStrategyInfo:
    with _get_service_stub() as stub:
        response: GetFullStrategiesInfoResponse = await stub.GetFullStrategiesInfo(
            GetFullStrategiesInfoRequest(strategy_ids=[UUID(id=workspace_id)]),
            metadata=get_default_grpc_metadata(user_id=user_id),
        )
        if not list(response.strategies):
            raise ValueError(f"No strategies found with the given workspace id: {workspace_id}")

    res = list(response.strategies)[0]
    return res


@grpc_retry
@async_perf_logger
async def create_ts_workspace(
    user_id: str, holdings: List[StockHolding], workspace_name: str
) -> Tuple[str, str]:
    with _get_service_stub() as stub:
        response: CreateTSWorkspaceResponse = await stub.CreateTSWorkspace(
            CreateTSWorkspaceRequest(name=workspace_name, holdings=holdings),
            metadata=get_default_grpc_metadata(user_id=user_id),
        )
        if response.status.code != 0:
            raise ValueError(
                f"Failed to create TS workspace: {response.status.code}"
                f" {response.status.message}"
            )
        return (response.workspace_id.id, response.strategy_id.id)


@grpc_retry
@async_perf_logger
async def rename_workspace(user_id: str, workspace_id: str, name: str) -> bool:
    with _get_service_stub() as stub:
        response: RenameWorkspaceResponse = await stub.RenameWorkspace(
            RenameWorkspaceRequest(workspace_id=UUID(id=workspace_id), name=name),
            metadata=get_default_grpc_metadata(user_id=user_id),
        )
        if response.status.code != 0:
            raise ValueError(
                f"Failed to rename workspace {workspace_id}: {response.status.code}"
                f" {response.status.message}"
            )
        return True


@grpc_retry
@async_perf_logger
async def modify_workspace_historical_holdings(
    user_id: str, workspace_id: str, holdings: List[StockHolding]
) -> None:
    with _get_service_stub() as stub:
        response: ModifyWorkspaceHistoricalHoldingsResponse = (
            await stub.ModifyWorkspaceHistoricalHoldings(
                ModifyWorkspaceHistoricalHoldingsRequest(
                    workspace_id=UUID(id=workspace_id),
                    workspace_trades=[
                        WorkspaceTrade(
                            gbi_id=holding.gbi_id,
                            final_weight=holding.weight,
                            trade_date=holding.date,
                        )
                        for holding in holdings
                    ],
                ),
                metadata=get_default_grpc_metadata(user_id=user_id),
            )
        )
        if response.status.code != 0:
            raise ValueError(
                f"Failed to modify workspace historical holdings: {response.status.code} {response.status.message}"
            )


@grpc_retry
@async_perf_logger
async def recalc_strategies(user_id: str, strategy_ids: List[str]) -> None:
    with _get_service_stub() as stub:
        response: RecalcStrategiesResponse = await stub.RecalcStrategies(
            RecalcStrategiesRequest(
                strategy_ids=[UUID(id=strategy_id) for strategy_id in strategy_ids]
            ),
            metadata=get_default_grpc_metadata(user_id=user_id),
        )
        if response.status.code != 0:
            raise ValueError(
                f"Failed to recalc strategies: {response.status.code}" f" {response.status.message}"
            )


@grpc_retry
@async_perf_logger
async def get_transitive_holdings_from_stocks_and_weights(
    user_id: str, weighted_securities: List[StockAndWeight]
) -> List[StockAndWeight]:
    with _get_service_stub() as stub:
        response: GetTransitiveHoldingsFromStocksAndWeightsResponse = (
            await stub.GetTransitiveHoldingsFromStocksAndWeights(
                GetTransitiveHoldingsFromStocksAndWeightsRequest(
                    weighted_securities=weighted_securities
                ),
                metadata=get_default_grpc_metadata(user_id=user_id),
            )
        )
        if response.status.code != 0:
            raise ValueError(
                f"Failed to recalc strategies: {response.status.code}" f" {response.status.message}"
            )

    return [weighted_security for weighted_security in response.weighted_securities]


@grpc_retry
@async_perf_logger
async def get_list_all_authorized_strategies(
    user_id: str,
) -> List[ListAllAuthorizedStrategiesResponse.AuthorizedStrategy]:
    with _get_service_stub() as stub:
        response: ListAllAuthorizedStrategiesResponse = await stub.ListAllAuthorizedStrategies(
            ListAllAuthorizedStrategiesRequest(),
            metadata=get_default_grpc_metadata(user_id=user_id),
        )
        return list(response.strategies)


@grpc_retry
@async_perf_logger
async def get_all_authorized_live_strategy_ids(user_id: str) -> List[str]:
    with _get_service_stub() as stub:
        response: GetAllAuthorizedLiveStrategyIdsResponse = (
            await stub.GetAllAuthorizedLiveStrategyIds(
                GetAllAuthorizedLiveStrategyIdsRequest(min_auth_level=PORTFOLIO_AUTH_ENUM_READ),
                metadata=get_default_grpc_metadata(user_id=user_id),
            )
        )
        if response.status.code != 0:
            raise ValueError(
                f"Failed to fetch authorized live strategy IDs for user {user_id}: {response.status.code}"  # noqa
                f" {response.status.message}"
            )
        return [x.id for x in response.strategy_ids]


@grpc_retry
@async_perf_logger
async def subscribe_to_marketplace_strategy(user_id: str, is_prod: bool) -> None:
    if not is_prod:
        # S&P500 Growth: https://insights-dev.boosted.ai/models/9590826b-6db9-4167-a60f-e66ca59e7acd/aef39126-d8d0-46f6-898d-61f1d87e1418  # noqa
        model_id = "9590826b-6db9-4167-a60f-e66ca59e7acd"
        portfolio_id = "aef39126-d8d0-46f6-898d-61f1d87e1418"
    else:
        # S&P 500 Tactical - Direct Trading: https://insights.boosted.ai/models/26098a7f-6ffe-4773-9a66-57aad699b1e2/a21a2ccb-bbe5-411f-8bf2-78af41e45c55/  # noqa
        model_id = "26098a7f-6ffe-4773-9a66-57aad699b1e2"
        portfolio_id = "a21a2ccb-bbe5-411f-8bf2-78af41e45c55"
    with _get_service_stub() as stub:
        await stub.SubscribeToMarketplaceStrategy(
            SubscribeToMarketplaceStrategyRequest(
                strategy=StrategyIdentifier(
                    model_id=UUID(id=model_id), portfolio_id=UUID(id=portfolio_id)
                )
            ),
            metadata=get_default_grpc_metadata(user_id=user_id),
        )
