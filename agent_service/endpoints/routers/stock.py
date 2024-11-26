from fastapi import APIRouter, Depends
from starlette import status

from agent_service.endpoints.authz_helper import User, parse_header
from agent_service.endpoints.models import (
    GetCompanyDescriptionResponse,
    GetDividendYieldResponse,
    GetEarningsSummaryResponse,
    GetExecutiveEarningsSummaryResponse,
    GetHistoricalPricesRequest,
    GetHistoricalPricesResponse,
    GetMarketDataResponse,
    GetNewsSummaryRequest,
    GetNewsSummaryResponse,
    GetNewsTopicsRequest,
    GetNewsTopicsResponse,
    GetOrderedSecuritiesRequest,
    GetOrderedSecuritiesResponse,
    GetPreviousEarningsResponse,
    GetPriceDataResponse,
    GetRealTimePriceResponse,
    GetSecurityProsConsResponse,
    GetSecurityResponse,
    GetUpcomingEarningsResponse,
)
from agent_service.endpoints.routers.utils import get_agent_svc_impl

router = APIRouter(prefix="/api/stock")


@router.post(
    "/get-ordered-securities",
    response_model=GetOrderedSecuritiesResponse,
    status_code=status.HTTP_200_OK,
)
async def get_ordered_securities(
    req: GetOrderedSecuritiesRequest, user: User = Depends(parse_header)
) -> GetOrderedSecuritiesResponse:
    agent_svc_impl = get_agent_svc_impl()
    return await agent_svc_impl.get_ordered_securities(req, user)


@router.post("/news-summary", response_model=GetNewsSummaryResponse, status_code=status.HTTP_200_OK)
async def get_news_summary(
    req: GetNewsSummaryRequest, user: User = Depends(parse_header)
) -> GetNewsSummaryResponse:
    agent_svc_impl = get_agent_svc_impl()
    return await agent_svc_impl.get_news_summary(req, user)


@router.post(
    "/historical-prices",
    response_model=GetHistoricalPricesResponse,
    status_code=status.HTTP_200_OK,
)
async def get_historical_prices(
    req: GetHistoricalPricesRequest, user: User = Depends(parse_header)
) -> GetHistoricalPricesResponse:
    agent_svc_impl = get_agent_svc_impl()
    return await agent_svc_impl.get_historical_prices(req, user)


@router.get(
    "/{gbi_id}/real-time-price",
    response_model=GetRealTimePriceResponse,
    status_code=status.HTTP_200_OK,
)
async def get_real_time_price(
    gbi_id: int, user: User = Depends(parse_header)
) -> GetRealTimePriceResponse:
    agent_svc_impl = get_agent_svc_impl()
    return await agent_svc_impl.get_real_time_price(user=user, gbi_id=gbi_id)


@router.get(
    "/{gbi_id}/market-data",
    response_model=GetMarketDataResponse,
    status_code=status.HTTP_200_OK,
)
async def get_market_data(gbi_id: int, user: User = Depends(parse_header)) -> GetMarketDataResponse:
    agent_svc_impl = get_agent_svc_impl()
    return await agent_svc_impl.get_market_data(user=user, gbi_id=gbi_id)


@router.get(
    "/{gbi_id}/price-data",
    response_model=GetPriceDataResponse,
    status_code=status.HTTP_200_OK,
)
async def get_price_data(gbi_id: int, user: User = Depends(parse_header)) -> GetPriceDataResponse:
    agent_svc_impl = get_agent_svc_impl()
    return await agent_svc_impl.get_price_data(user=user, gbi_id=gbi_id)


@router.get(
    "/{gbi_id}/dividend-yield",
    response_model=GetDividendYieldResponse,
    status_code=status.HTTP_200_OK,
)
async def get_dividend_yield(
    gbi_id: int, user: User = Depends(parse_header)
) -> GetDividendYieldResponse:
    agent_svc_impl = get_agent_svc_impl()
    return await agent_svc_impl.get_dividend_yield(user=user, gbi_id=gbi_id)


@router.get("/{gbi_id}/executive-earnings-summary")
async def get_executive_earnings_summary(
    gbi_id: int, user: User = Depends(parse_header)
) -> GetExecutiveEarningsSummaryResponse:
    agent_svc_impl = get_agent_svc_impl()
    return await agent_svc_impl.get_executive_earnings_summary(user=user, gbi_id=gbi_id)


@router.get("/{gbi_id}/previous-earnings")
async def get_previous_earnings(
    gbi_id: int, user: User = Depends(parse_header)
) -> GetPreviousEarningsResponse:
    agent_svc_impl = get_agent_svc_impl()
    return await agent_svc_impl.get_previous_earnings(user=user, gbi_id=gbi_id)


@router.get("/{gbi_id}/upcoming-earnings")
async def get_upcoming_earnings(
    gbi_id: int, user: User = Depends(parse_header)
) -> GetUpcomingEarningsResponse:
    agent_svc_impl = get_agent_svc_impl()
    return await agent_svc_impl.get_upcoming_earnings(user=user, gbi_id=gbi_id)


@router.get("/{gbi_id}/company-description")
async def get_company_description(
    gbi_id: int, user: User = Depends(parse_header)
) -> GetCompanyDescriptionResponse:
    agent_svc_impl = get_agent_svc_impl()
    return await agent_svc_impl.get_company_description(user=user, gbi_id=gbi_id)


@router.get("/{gbi_id}/pros-cons")
async def get_security_pros_cons(
    gbi_id: int, user: User = Depends(parse_header)
) -> GetSecurityProsConsResponse:
    agent_svc_impl = get_agent_svc_impl()
    return await agent_svc_impl.get_security_pros_cons(user=user, gbi_id=gbi_id)


@router.get("/{gbi_id}/earnings-summary")
async def get_earnings_summary(
    gbi_id: int, user: User = Depends(parse_header)
) -> GetEarningsSummaryResponse:
    agent_svc_impl = get_agent_svc_impl()
    return await agent_svc_impl.get_earnings_summary(user=user, gbi_id=gbi_id)


@router.get("/{gbi_id}/security")
async def get_security(gbi_id: int, user: User = Depends(parse_header)) -> GetSecurityResponse:
    agent_svc_impl = get_agent_svc_impl()
    return await agent_svc_impl.get_security(user=user, gbi_id=gbi_id)


@router.post(
    "/news-topics",
    response_model=GetNewsTopicsResponse,
    status_code=status.HTTP_200_OK,
)
async def get_news_topics(
    req: GetNewsTopicsRequest, user: User = Depends(parse_header)
) -> GetNewsTopicsResponse:
    agent_svc_impl = get_agent_svc_impl()
    return await agent_svc_impl.get_news_topics(req=req, user=user)
