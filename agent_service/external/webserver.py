import logging
from functools import lru_cache
from typing import Optional

import aiohttp
from gbi_common_py_utils.utils.environment import (
    DEV_TAG,
    LOCAL_TAG,
    PROD_TAG,
    get_environment_tag,
)

from agent_service.external.grpc_utils import create_jwt
from agent_service.utils.logs import async_perf_logger

logger = logging.getLogger(__name__)

####################################################################################################
# Utils
####################################################################################################
DEFAULT_URLS = {
    LOCAL_TAG: "https://insights-dev.boosted.ai",
    DEV_TAG: "https://insights-dev.boosted.ai",
    PROD_TAG: "https://insights.boosted.ai",
}


@lru_cache(maxsize=1)
def get_url() -> str:
    env = get_environment_tag()
    if env not in DEFAULT_URLS:
        raise ValueError(f"No URL is set for environment {env}")
    return DEFAULT_URLS[env]


async def _get_graphql(user_id: str, query: str, variables: dict) -> Optional[dict]:
    try:
        jwt_token = create_jwt(user_id=user_id)
        headers = {"Authorization": f"Cognito {jwt_token}"}
        json_req = {"query": query, "variables": variables}
        base_url = get_url()
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url=f"{base_url}/api/graphql",
                json=json_req,
                headers=headers,
            ) as resp:
                if not resp.ok:
                    logger.error(
                        f"Response is not OK in _get_graphql.\n"
                        f"Query: {query}\n"
                        f"variables: {variables}:\n"
                    )
                    return None

                json_resp = await resp.json()
                if "errors" in json_resp:
                    logger.error(
                        f"Error in _get_graphql.\n"
                        f"Query: {query}\n"
                        f"variables: {variables}:\n"
                        f"{json_resp['errors']}"
                    )
                    return None

                return json_resp
    except Exception as e:
        logger.exception(
            f"Exception in _get_graphql.\n" f"Query: {query}\n" f"variables: {variables}:\n" f"{e}"
        )
        return None


####################################################################################################
# GetOrderedSecurities
####################################################################################################
@async_perf_logger
async def get_ordered_securities(
    user_id: str,
    searchText: str,
    preferEtfs: bool = False,
    includeDepositary: bool = False,
    includeForeign: bool = False,
    order: list[str] = ["volume"],
    priorityCountry: Optional[str] = None,
    priorityExchange: Optional[str] = None,
    priorityCurrency: Optional[str] = None,
    maxItems: int = 0,
) -> list[dict]:
    gql_query = """
    query GetOrderedSecurities($input: StockFilterInput!) {
        orderedStockFilter(input: $input) {
            gbiId
            symbol
            isin
            name
            currency
            country
            primaryExchange
            gics
            assetType
            securityType
            from
            to
            sector {
                id
                name
                topParentName
            }
            isPrimaryTradingItem
            hasRecommendations
        }
    }
    """
    variables = {
        "input": {
            "searchText": searchText,
            "preferEtfs": preferEtfs,
            "includeDepositary": includeDepositary,
            "includeForeign": includeForeign,
            "order": order,
            "priorityCountry": priorityCountry,
            "priorityExchange": priorityExchange,
            "priorityCurrency": priorityCurrency,
            "maxItems": maxItems,
        }
    }
    resp = await _get_graphql(user_id=user_id, query=gql_query, variables=variables)
    if resp is None:
        return []
    elif "data" not in resp or "orderedStockFilter" not in resp["data"]:
        logger.error(f"Invalid response in GQL `get_ordered_securities`:\n{resp}")
        return []

    return resp["data"]["orderedStockFilter"]
