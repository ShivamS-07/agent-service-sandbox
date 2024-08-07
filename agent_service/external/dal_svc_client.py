import datetime
import logging
import os
from functools import lru_cache
from typing import Any, Dict, List, Optional

import aiohttp
from gbi_common_py_utils.utils.environment import (
    DEV_TAG,
    LOCAL_TAG,
    PROD_TAG,
    get_environment_tag,
)
from pydantic import BaseModel

from agent_service.types import ActionType

logger = logging.getLogger(__name__)

DEFAULT_URLS = {
    LOCAL_TAG: "http://dal-dev.boosted.ai",
    DEV_TAG: "http://dal-dev.boosted.ai",
    PROD_TAG: "http://dal-prod.boosted.ai",
}
DAL_SERVICE_CLIENT = None


class SecurityRow(BaseModel):
    gbi_id: int
    weight: Optional[float]
    date: Optional[datetime.datetime]


class ParsePortfolioWorkspaceResponse(BaseModel):
    securities: List[SecurityRow]
    parse_failures: List[str]
    parse_warnings: List[str]


class PreviousTradesMetadata(BaseModel):
    gbi_id: int
    trade_date: datetime.date
    action: str
    allocation_change: float


@lru_cache(maxsize=1)
def get_base_url() -> str:
    url = os.environ.get("DAL_URL")
    if url is not None:
        logger.warning(f"Found DAL service url override: {url}")
        return url

    env = get_environment_tag()
    if env not in DEFAULT_URLS:
        raise ValueError(f"No URL is set for environment {env}")
    return DEFAULT_URLS[env]


class DALServiceClient:
    def __init__(self) -> None:
        self.url_base = get_base_url()
        self.session = aiohttp.ClientSession()

    async def _post(
        self, endpoint: str, body: dict, headers: Optional[dict] = None
    ) -> aiohttp.ClientResponse:
        if headers:
            return await self.session.post(self.url_base + endpoint, json=body, headers=headers)
        return await self.session.post(self.url_base + endpoint, json=body)

    async def parse_file(self, b64data: str, content_type: str) -> ParsePortfolioWorkspaceResponse:
        response: aiohttp.ClientResponse = await self._post(
            "/api/v0/portfolio-workspace/parse_file",
            {
                "file_bytes_base64": b64data,
                "content_type": content_type,
            },
        )

        if response.content_type != "application/json":
            raw_text = await response.text()
            logger.warn(f"DAL parse_file unexpected response {raw_text}")
            raise Exception(
                f"DAL parse_file response type was not JSON, instead it was {response.content_type}"
            )

        response_content: Dict[str, Any] = await response.json()

        if response.status != 200:
            logger.warn(
                f"Failed to parse file from DAL: {response_content.get('message')} {response_content.get('detail')}"
            )

        return ParsePortfolioWorkspaceResponse(
            securities=[
                SecurityRow(
                    gbi_id=security_row["gbi_id"],
                    weight=security_row.get("weight"),
                    date=(
                        datetime.datetime.strptime(security_row["date"], "%Y-%m-%d")
                        if "date" in security_row
                        else None
                    ),
                )
                for security_row in response_content.get("securities", [])
            ],
            parse_failures=response_content.get("parse_failures", []),
            parse_warnings=response_content.get("parse_warnings", []),
        )

    async def fetch_previous_trades(
        self, model_id: str, portfolio_id: str, start_date: str, end_date: str
    ) -> List[PreviousTradesMetadata]:
        response: aiohttp.ClientResponse = await self._post(
            "/api/v0/portfolio-analysis/get-data/",
            {
                "portfolio_id": portfolio_id,
                "model_id": model_id,
                "min_date": start_date,
                "max_date": end_date,
                "fields": [
                    "shares_traded",
                    "shares_owned",
                    "price",
                    "valid_signal",
                    "allocation_traded",
                ],
                "return_format": "json",
            },
        )

        response_content: Dict[str, Any] = await response.json()

        if response.status != 200:
            logger.warn(
                f"Failed to fetch data from DAL: {response_content.get('message')} {response_content.get('detail')}"
            )

        # Generate the list of PreviousTradesMetadata objects
        gbi_ids = response_content["rows"]
        trade_dates = response_content["columns"]
        field_indices = response_content["field_map"]
        allocation_traded_idx = field_indices["allocation_traded"]
        shares_traded_idx = field_indices["shares_traded"]

        trades_metadata: List[PreviousTradesMetadata] = []
        for idx, gbi_id in enumerate(gbi_ids):
            stock_trade_list: List[Any] = response_content["data"][idx]
            for i in range(len(stock_trade_list)):
                trade_entry = stock_trade_list[i]
                allocation_change = trade_entry[allocation_traded_idx]
                shares_traded = trade_entry[shares_traded_idx]
                action = ActionType.BUY if shares_traded > 0 else ActionType.SELL

                if allocation_change != 0.0:
                    trades_metadata.append(
                        PreviousTradesMetadata(
                            gbi_id=int(gbi_id),
                            trade_date=datetime.datetime.strptime(
                                trade_dates[i], "%Y-%m-%d"
                            ).date(),
                            action=action,
                            allocation_change=allocation_change,
                        )
                    )

        # Sort in descending order of trade date
        sorted_trades = sorted(trades_metadata, key=lambda x: x.trade_date, reverse=True)

        return sorted_trades

    async def create_watchlist_from_file(
        self, b64data: str, content_type: str, name: str, user_id: str, jwt: Optional[str]
    ) -> Optional[str]:
        """
        Creates a watchlist from base 64 encoded file data. Returns the watchlist ID.
        """

        headers = {
            "User-Id": user_id,
        }

        if jwt:
            headers["Authorization"] = f"Cognito {jwt}"

        response: aiohttp.ClientResponse = await self._post(
            "/api/v0/watchlist/create_watchlist_from_file/",
            {
                "file_bytes_base64": b64data,
                "content_type": content_type,
                "name": name,
            },
            headers=headers,
        )

        if response.content_type != "application/json":
            raw_text = await response.text()
            logger.warn(f"DAL create_watchlist_from_file unexpected response {raw_text}")
            raise Exception(
                f"DAL create_watchlist_from_file response type was not JSON, instead it was {response.content_type}"
            )

        response_content: Dict[str, Any] = await response.json()

        if response.status != 200:
            logger.warn(
                f"Failed to create watchlist from file from DAL: "
                f"{response_content.get('message')} {response_content.get('detail')}"
            )

        return response_content.get("watchlist_id")


def get_dal_client() -> DALServiceClient:
    global DAL_SERVICE_CLIENT
    if DAL_SERVICE_CLIENT is None:
        DAL_SERVICE_CLIENT = DALServiceClient()
    return DAL_SERVICE_CLIENT
