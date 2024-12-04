from fastapi import APIRouter, Depends
from starlette import status

from agent_service.endpoints.authz_helper import User, parse_header
from agent_service.endpoints.models import (
    GetEtfAllocationsResponse,
    GetEtfHoldingsResponse,
    GetEtfSummaryResponse,
)
from agent_service.endpoints.routers.utils import get_agent_svc_impl

router = APIRouter(prefix="/api/etf")


@router.get(
    "/{gbi_id}/summary", response_model=GetEtfSummaryResponse, status_code=status.HTTP_200_OK
)
async def get_etf_summary(gbi_id: int, user: User = Depends(parse_header)) -> GetEtfSummaryResponse:
    agent_svc_impl = get_agent_svc_impl()
    return await agent_svc_impl.get_etf_summary(user=user, gbi_id=gbi_id)


@router.get(
    "/{gbi_id}/holdings", response_model=GetEtfHoldingsResponse, status_code=status.HTTP_200_OK
)
async def get_etf_holdings(
    gbi_id: int, user: User = Depends(parse_header)
) -> GetEtfHoldingsResponse:
    agent_svc_impl = get_agent_svc_impl()
    return await agent_svc_impl.get_etf_holdings(user=user, gbi_id=gbi_id)


@router.get(
    "/{gbi_id}/allocations",
    response_model=GetEtfAllocationsResponse,
    status_code=status.HTTP_200_OK,
)
async def get_etf_allocations(
    gbi_id: int, user: User = Depends(parse_header)
) -> GetEtfAllocationsResponse:
    agent_svc_impl = get_agent_svc_impl()
    return await agent_svc_impl.get_etf_allocations(user=user, gbi_id=gbi_id)
