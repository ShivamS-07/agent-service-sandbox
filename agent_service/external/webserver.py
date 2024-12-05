import datetime
import logging
from functools import lru_cache
from typing import Optional

from gbi_common_py_utils.utils.environment import (
    DEV_TAG,
    LOCAL_TAG,
    PROD_TAG,
    get_environment_tag,
)

from agent_service.external.grpc_utils import create_jwt
from agent_service.external.utils import get_http_session
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
        session = get_http_session()
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


####################################################################################################
# GetCompanyDescription
####################################################################################################
@async_perf_logger
async def get_company_description(user_id: str, gbi_id: int) -> Optional[str]:
    gql_query = """
    query GetCompanyDescription($gbiId: Int!) {
        getCompanyDescription(gbiId: $gbiId)
    }
    """
    variables = {
        "gbiId": gbi_id,
    }
    resp = await _get_graphql(user_id=user_id, query=gql_query, variables=variables)
    if resp is None or "data" not in resp or "getCompanyDescription" not in resp["data"]:
        return None
    return resp["data"]["getCompanyDescription"]


####################################################################################################
# GetEarningsSummaries
####################################################################################################
@async_perf_logger
async def get_earnings_summary(user_id: str, gbi_id: int) -> Optional[list[dict]]:
    gql_query = """
    query GetEarningsSummaries($gbiIds: [Int!]!) {
        getEarningsSummaries(gbiIds: $gbiIds) {
            gbiId
            reports {
                date
                title
                details {
                    header
                    isAligned
                    detail
                    sentiment
                    references {
                        valid
                        referenceLines {
                            highlights
                            paragraphs
                        }
                        justification
                    }
                }
                highlights
                qaDetails {
                    header
                    isAligned
                    detail
                    sentiment
                    references {
                        valid
                        referenceLines {
                            highlights
                            paragraphs
                        }
                        justification
                    }
                }
                qaHighlights
                quarter
                year
            }
        }
    }
    """
    variables = {
        "gbiIds": [gbi_id],
    }
    resp = await _get_graphql(user_id=user_id, query=gql_query, variables=variables)
    if (
        resp is None
        or "data" not in resp
        or "getEarningsSummaries" not in resp["data"]
        or len(resp["data"]["getEarningsSummaries"]) == 0
    ):
        return None
    return resp["data"]["getEarningsSummaries"][0]["reports"]


####################################################################################################
# GetSecurityProsCons
####################################################################################################
@async_perf_logger
async def get_security_pros_cons(user_id: str, gbi_id: int) -> Optional[dict]:
    gql_query = """
    query GetSecurityProsCons($gbiId: Int!) {
        securityProsCons(gbiId: $gbiId) {
            pros {
                summary
                details
            }
            cons {
                summary
                details
            }
        }
    }
    """
    variables = {
        "gbiId": gbi_id,
    }
    resp = await _get_graphql(user_id=user_id, query=gql_query, variables=variables)
    if resp is None or "data" not in resp or "securityProsCons" not in resp["data"]:
        return None
    return resp["data"]["securityProsCons"]


####################################################################################################
# GetSecurities
####################################################################################################
@async_perf_logger
async def get_security(user_id: str, gbi_id: int) -> Optional[dict]:
    gql_query = """
    query GetSecurities($ids: [Int!]) {
        securities(ids: $ids) {
            gbiId
            symbol
            name
            isin
            country
            currency
            primaryExchange
            sector {
                id
                name
                topParentName
            }
            securityType
        }
    }
    """
    variables = {
        "ids": [gbi_id],
    }
    resp = await _get_graphql(user_id=user_id, query=gql_query, variables=variables)
    if (
        resp is None
        or "data" not in resp
        or "securities" not in resp["data"]
        or len(resp["data"]["securities"]) == 0
    ):
        return None
    return resp["data"]["securities"][0]


####################################################################################################
# GetStockNewsSummary
####################################################################################################
@async_perf_logger
async def get_stock_news_summary(
    user_id: str, gbi_id: int, delta_horizon: str, show_hypotheses: bool = False
) -> Optional[dict]:
    gql_query = """
    query GetStockNewsSummary(
        $gbiId: Int!
        $deltaHorizon: String!
        $showHypotheses: Boolean!
    ) {
        getStockNewsSummary(
            gbiId: $gbiId
            deltaHorizon: $deltaHorizon
            showHypotheses: $showHypotheses
        ) {
            sentiment
            summary
            sourceCounts {
                sourceId
                sourceName
                domainUrl
                count
                deltaCount
                isTopSource
                sentiment
            }
        }
    }
    """
    variables = {
        "gbiId": gbi_id,
        "deltaHorizon": delta_horizon,
        "showHypotheses": show_hypotheses,
    }
    resp = await _get_graphql(user_id=user_id, query=gql_query, variables=variables)
    if resp is None or "data" not in resp or "getStockNewsSummary" not in resp["data"]:
        return None
    return resp["data"]["getStockNewsSummary"]


####################################################################################################
# GetStockNewsTopics
####################################################################################################
@async_perf_logger
async def get_stock_news_topics(
    user_id: str, gbi_id: int, delta_horizon: str, show_hypotheses: bool = False
) -> Optional[dict]:
    gql_query = """
    query GetStockNewsTopics(
        $gbiId: Int!
        $deltaHorizon: String!
        $showHypotheses: Boolean!
    ) {
        getStockNewsTopics(
            gbiId: $gbiId
            deltaHorizon: $deltaHorizon
            showHypotheses: $showHypotheses
        ) {
            topics {
                topicId
                topicLabel
                topicDescription
                topicImpact
                topicPolarity
                topicRating
                topicStatus
                previousTopicPolarity
                newsItems {
                    headline
                    isTopSource
                    newsId
                    publishedAt
                    source
                    url
                }
                dailyNewsCounts {
                    date
                    newsCount
                }
                isCrossCompanyTopic
                topicStockRationale
                originalTopicImpact
                originalTopicPolarity
            }
            sentimentHistory {
                date
                sentimentScore
            }
        }
    }
    """
    variables = {
        "gbiId": gbi_id,
        "deltaHorizon": delta_horizon,
        "showHypotheses": show_hypotheses,
    }
    resp = await _get_graphql(user_id=user_id, query=gql_query, variables=variables)
    if resp is None or "data" not in resp or "getStockNewsTopics" not in resp["data"]:
        return None
    return resp["data"]["getStockNewsTopics"]


####################################################################################################
# GetStockHistoricalPrices
####################################################################################################
@async_perf_logger
async def get_stock_historical_prices(
    user_id: str, gbi_id: int, start_date: datetime.date, end_date: datetime.date
) -> Optional[list[dict]]:
    gql_query = """
    query GetHisoricalPrices(
        $gbiIds: [Int!]!
        $startDate: String!
        $endDate: String!
    ) {
        adjustedCumulativeReturns(
            gbiIds: $gbiIds
            startDate: $startDate
            endDate: $endDate
        ) {
            date
            field
            value
        }
    }
    """
    variables = {
        "gbiIds": [gbi_id],
        "startDate": start_date.strftime("%Y-%m-%d"),
        "endDate": end_date.strftime("%Y-%m-%d"),
    }
    resp = await _get_graphql(user_id=user_id, query=gql_query, variables=variables)
    if resp is None or "data" not in resp or "adjustedCumulativeReturns" not in resp["data"]:
        return None
    return resp["data"]["adjustedCumulativeReturns"]


####################################################################################################
# GetStockRealTimePrice
####################################################################################################
@async_perf_logger
async def get_stock_real_time_price(user_id: str, gbi_id: int) -> Optional[dict]:
    gql_query = """
    query GetRealTimePrices($gbiIds: [Int!]!) {
        getRealTimePrices(gbiIds: $gbiIds) {
            gbiId
            latestPrice
            lastClosePrice
            lastUpdate
            lastClosePriceUpdate
        }
    }
    """
    variables = {
        "gbiIds": [gbi_id],
    }
    resp = await _get_graphql(user_id=user_id, query=gql_query, variables=variables)
    if (
        resp is None
        or "data" not in resp
        or "getRealTimePrices" not in resp["data"]
        or len(resp["data"]["getRealTimePrices"]) == 0
    ):
        return None
    return resp["data"]["getRealTimePrices"][0]


####################################################################################################
# GetStockMarketData
####################################################################################################
@async_perf_logger
async def get_stock_market_data(user_id: str, gbi_id: int) -> Optional[list[dict]]:
    gql_query = """
    query GetFeatureValues(
        $gbiIds: [Int!]!
        $featureIds: [String!]!
        $startDate: String!
        $endDate: String!
        $getMostRecent: Boolean
    ) {
        featureValues(
            gbiIds: $gbiIds
            featureIds: $featureIds
            startDate: $startDate
            endDate: $endDate
            getMostRecent: $getMostRecent
        ) {
            gbiId
            date
            field
            value
        }
    }
    """
    variables = {
        "gbiIds": [gbi_id],
        "featureIds": ["spiq_close", "spiq_100161", "spiq_market_cap", "spiq_low", "spiq_high"],
        "startDate": datetime.datetime.now().strftime("%Y-%m-%d"),
        "endDate": datetime.datetime.now().strftime("%Y-%m-%d"),
        "getMostRecent": True,
    }
    resp = await _get_graphql(user_id=user_id, query=gql_query, variables=variables)
    if (
        resp is None
        or "data" not in resp
        or "featureValues" not in resp["data"]
        or len(resp["data"]["featureValues"]) == 0
    ):
        return None
    return resp["data"]["featureValues"]


####################################################################################################
# GetStockDividendYield
####################################################################################################
@async_perf_logger
async def get_stock_dividend_yield(user_id: str, gbi_id: int) -> Optional[float]:
    gql_query = """
    query GetStocksDividendYield($gbiIds: [Int!]!, $date: String!) {
        bulkSecurityDividendYield(gbiIds: $gbiIds, date: $date) {
            gbiId
            dividendYield
        }
    }
    """
    variables = {
        "gbiIds": [gbi_id],
        "date": datetime.datetime.now().strftime("%Y-%m-%d"),
    }
    resp = await _get_graphql(user_id=user_id, query=gql_query, variables=variables)
    if (
        resp is None
        or "data" not in resp
        or "bulkSecurityDividendYield" not in resp["data"]
        or len(resp["data"]["bulkSecurityDividendYield"]) == 0
        or "dividendYield" not in resp["data"]["bulkSecurityDividendYield"][0]
    ):
        return None
    return resp["data"]["bulkSecurityDividendYield"][0]["dividendYield"]


####################################################################################################
# GetStockPriceData
####################################################################################################
@async_perf_logger
async def get_stock_price_data(user_id: str, gbi_id: int) -> Optional[dict]:
    gql_query = """
    query StocksPriceData(
        $gbiIds: [Int!]!
        $startDate: String!
        $endDate: String!
    ) {
        bulkSecurityPriceData(
            gbiIds: $gbiIds
            startDate: $startDate
            endDate: $endDate
        ) {
            gbiId
            lastClosePrice
            marketCap
            percentWeight
            periodLowPrice
            periodHighPrice
            pricePercentile
        }
    }
    """
    variables = {
        "gbiIds": [gbi_id],
        "startDate": (datetime.datetime.now() - datetime.timedelta(days=365)).strftime("%Y-%m-%d"),
        "endDate": datetime.datetime.now().strftime("%Y-%m-%d"),
    }
    resp = await _get_graphql(user_id=user_id, query=gql_query, variables=variables)
    if (
        resp is None
        or "data" not in resp
        or "bulkSecurityPriceData" not in resp["data"]
        or len(resp["data"]["bulkSecurityPriceData"]) == 0
    ):
        return None
    return resp["data"]["bulkSecurityPriceData"][0]


####################################################################################################
# GetExecutiveEarningsSummary
####################################################################################################
@async_perf_logger
async def get_executive_earnings_summary(user_id: str, gbi_id: int) -> Optional[list[dict]]:
    gql_query = """
    query GetLatestEarningsComparison($gbiIds: [Int!]!) {
        getLatestEarningsChanges(gbiIds: $gbiIds) {
            gbiId
            changes {
                header
                detail
            }
        }
    }
    """
    variables = {
        "gbiIds": [gbi_id],
    }
    resp = await _get_graphql(user_id=user_id, query=gql_query, variables=variables)
    if (
        resp is None
        or "data" not in resp
        or "getLatestEarningsChanges" not in resp["data"]
        or len(resp["data"]["getLatestEarningsChanges"]) == 0
        or "changes" not in resp["data"]["getLatestEarningsChanges"][0]
    ):
        return None
    return resp["data"]["getLatestEarningsChanges"][0]["changes"]


####################################################################################################
# GetEarningsStatus
####################################################################################################
@async_perf_logger
async def get_earnings_status(user_id: str, gbi_id: int) -> Optional[dict]:
    gql_query = """
    query StocksEarningsStatus($gbiId: Int!) {
        getEarningsDataStatus(gbiId: $gbiId) {
            providerDataPresent
            providerLatestYear
            providerLatestQuarter
            earningsSummaryAutogenerated
            latestSummaryGenerated
            latestSummaryStatus
        }
    }
    """
    variables = {
        "gbiId": gbi_id,
    }
    resp = await _get_graphql(user_id=user_id, query=gql_query, variables=variables)
    if resp is None or "data" not in resp or "getEarningsDataStatus" not in resp["data"]:
        return None
    return resp["data"]["getEarningsDataStatus"]


####################################################################################################
# GetUpcomingEarnings
####################################################################################################
@async_perf_logger
async def get_upcoming_earnings(user_id: str, gbi_id: int) -> Optional[dict]:
    gql_query = """
    query StocksUpcomingEarnings($gbiIds: [Int!]!, $date: String!) {
        bulkUpcomingEarningsReports(gbiIds: $gbiIds, date: $date) {
            gbiId
            date
            earningsTimestamp
            quarter
            year
        }
    }
    """
    variables = {
        "gbiIds": [gbi_id],
        "date": datetime.datetime.now().strftime("%Y-%m-%d"),
    }
    resp = await _get_graphql(user_id=user_id, query=gql_query, variables=variables)
    if (
        resp is None
        or "data" not in resp
        or "bulkUpcomingEarningsReports" not in resp["data"]
        or len(resp["data"]["bulkUpcomingEarningsReports"]) == 0
    ):
        return None
    return resp["data"]["bulkUpcomingEarningsReports"][0]


####################################################################################################
# GetEtfSummary
####################################################################################################
@async_perf_logger
async def get_etf_summary(user_id: str, gbi_id: int) -> Optional[dict]:
    gql_query = """
    query GetETFSummary($gbiId: Int!) {
        etfSummary(etfId: $gbiId) {
            description
            benchmarks {
                id
                name
            }
            markets {
                id
                name
            }
            styles {
                id
                name
            }
            spiqSectors {
                id
                name
            }
            dividendFrequency {
            id
                name
            }
            lastYear {
                low
                high
            }
            lastDay {
                low
                high
            }
            lastClosePrice
            consensusPriceTarget
            size
            average3MDailyVolume
            dividendYield
            netExpenseRatio
            inceptionDate
            activeUntilDate
            performanceAsOfDate
        }
    }
    """
    variables = {
        "gbiId": gbi_id,
    }
    resp = await _get_graphql(user_id=user_id, query=gql_query, variables=variables)
    if resp is None or "data" not in resp or "etfSummary" not in resp["data"]:
        return None
    return resp["data"]["etfSummary"]


####################################################################################################
# GetEtfAllocations
####################################################################################################
@async_perf_logger
async def get_etf_allocations(user_id: str, gbi_id: int) -> Optional[dict]:
    gql_query = """
    query etfAllocations($etfGbiId: Int!, $startDate: String, $endDate: String) {
        etfAllocations(
            etfGbiId: $etfGbiId
            startDate: $startDate
            endDate: $endDate
        ) {
            marketCapAllocations {
                allocations {
                    marketCap {
                        label
                        maxMarketCap
                        minMarketCap
                    }
                    weight
                }
                asOfDate
                etfGbiId
            }
            sectorRegionAllocations {
                startDate
                endDate
                isoCurrency
                regionAllocations {
                    country {
                        name
                    }
                    isoCountryCode
                    endGrossWeight
                    endShortWeight
                    endLongWeight
                    endNetWeight
                }
                sectorAllocations {
                    sector {
                        name
                    }
                    sectorId
                    endGrossWeight
                    endShortWeight
                    endLongWeight
                    endNetWeight
                }
            }
        }
    }
    """
    variables = {
        "etfGbiId": gbi_id,
        "startDate": datetime.datetime.now().strftime("%Y-%m-%d"),
        "endDate": datetime.datetime.now().strftime("%Y-%m-%d"),
    }
    resp = await _get_graphql(user_id=user_id, query=gql_query, variables=variables)
    if resp is None or "data" not in resp or "etfAllocations" not in resp["data"]:
        return None
    return resp["data"]["etfAllocations"]


####################################################################################################
# GetEtfHoldings
####################################################################################################
@async_perf_logger
async def get_etf_holdings(user_id: str, gbi_id: int) -> Optional[dict]:
    gql_query = """
    query EtfHoldingsForDate(
        $etfIds: [Int!]!
        $asOfDate: String!
        $flattenFundOfFunds: Boolean
    ) {
        etfHoldingsForDate(
            etfIds: $etfIds
            asofDate: $asOfDate
            flattenFundOfFunds: $flattenFundOfFunds
        ) {
            asOfDate
            holdings {
                security {
                    gbiId
                    country
                    currency
                    isin
                    name
                    primaryExchange
                    sector {
                        id
                        name
                        topParentName
                    }
                    symbol
                    securityType
                }
                gbiId
                weight
            }
        }
    }
    """
    variables = {
        "etfIds": [gbi_id],
        "asOfDate": datetime.datetime.now().strftime("%Y-%m-%d"),
        "flattenFundOfFunds": False,
    }
    resp = await _get_graphql(user_id=user_id, query=gql_query, variables=variables)
    if (
        resp is None
        or "data" not in resp
        or "etfHoldingsForDate" not in resp["data"]
        or len(resp["data"]["etfHoldingsForDate"]) == 0
    ):
        return None
    return resp["data"]["etfHoldingsForDate"][0]


####################################################################################################
# GetEtfHoldingsStats
####################################################################################################
@async_perf_logger
async def get_etf_holdings_stats(user_id: str, gbi_id: int) -> Optional[dict]:
    gql_query = """
    query GetEtfFundHoldingsStatistics(
        $etfGbiId: Int!
        $recommendationHorizon: RecommendationHorizon!
        $flattenFundOfFunds: Boolean
    ) {
        etfFundHoldingsStatistics(
            etfGbiId: $etfGbiId
            recommendationHorizon: $recommendationHorizon
            flattenFundOfFunds: $flattenFundOfFunds
        ) {
            etfGbiId
            asOfDate
            individualStatistics {
                current
                dividendYield
                previous
                security {
                    gbiId
                    weight
                    security {
                        gbiId
                        country
                        currency
                        isin
                        name
                        primaryExchange
                        sector {
                            id
                            name
                            topParentName
                        }
                        symbol
                        securityType
                    }
                }
            }
            overallStatistics {
                marketCap
                priceToBook
                priceToCashFlow
                priceToEarnings
                salesToEv
            }
        }
    }
    """
    variables = {
        "etfGbiId": gbi_id,
        "recommendationHorizon": "RecommendationHorizon1M",
        "flattenFundOfFunds": False,
    }
    resp = await _get_graphql(user_id=user_id, query=gql_query, variables=variables)
    if resp is None or "data" not in resp or "etfFundHoldingsStatistics" not in resp["data"]:
        return None
    return resp["data"]["etfFundHoldingsStatistics"]
