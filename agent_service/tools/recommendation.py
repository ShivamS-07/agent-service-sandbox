import copy
from typing import Any, Dict, List

from pydantic import field_validator

from agent_service.external.discover_svc_client import get_temporary_discover_block_data
from agent_service.external.investment_policy_svc import (
    get_all_stock_investment_policies,
)
from agent_service.io_types.misc import StockID
from agent_service.tool import ToolArgs, ToolCategory, ToolRegistry, tool
from agent_service.tools.stocks import GetStockUniverseInput, get_stock_universe
from agent_service.tools.tool_log import tool_log
from agent_service.types import PlanRunContext
from agent_service.utils.prefect import get_prefect_logger

# Define as the template. When you want to use it, please DEEPCOPY it!
SETTINGS_TEMPLATE: Dict[str, Any] = {
    "horizon_settings": {"news_horizon": "1W"},
    "ism_settings": {
        "ism_id": None,
        "match_labels": ["Perfect Match"],
        "weight": 0.5,
    },
    "block_type": "stock",
    "universe_ids": [],
    "sector_settings": {"sector_ids": []},
    "rating_settings": {"boundary": {"lb": None, "ub": None}, "weight": 1.0},
    "strategy_settings": {  # these are exclusive
        "all_strategies": False,
        "subscribed_strategies": True,
        "strategy_ids": None,
    },
    "theme_settings": {
        "has_recent_major_developments": False,
        "recent_major_developments_days": 7,
        "themes": [],
    },
    "news_settings": {
        "sentiment_boundaries": [{"lb": -1, "ub": 1}],
        "prev_sentiment_boundaries": [],
        "weight": 0.5,
    },
    "earnings_settings": {
        "eps": {"at_least": None, "consecutive": None, "exact": None},
        "revenue": {"at_least": None, "consecutive": None, "exact": None},
    },
    "gbi_ids": [],  # if not empty, `universe_ids` will be ignored
}


class GetRecommendedStocksInput(ToolArgs):
    """Note: This is to find the recommended stocks from the provided stock list. It takes into
    many factors like news, ISM, ratings, etc.
    """

    stock_ids: List[StockID]  # if empty, we will default to use SP500
    buy: bool  # whether to get buy or sell recommendations
    horizon: str = "1M"  # 1M, 3M, 1Y
    delta_horizon: str = "1M"  # 1W, 1M, 3M, 6M, 9M, 1Y
    news_horizon: str = "1W"  # 1W, 1M, 3M
    num_stocks_to_return: int = 5

    @field_validator("horizon", mode="before")
    @classmethod
    def validate_horizon(cls, value: Any) -> Any:
        if not isinstance(value, str):
            raise ValueError("horizon must be a string")
        elif value not in ("1M", "3M", "1Y"):
            raise ValueError("horizon must be one of 1M, 3M, 1Y")

        return value

    @field_validator("delta_horizon", mode="before")
    @classmethod
    def validate_delta_horizon(cls, value: Any) -> Any:
        if not isinstance(value, str):
            raise ValueError("delta horizon must be a string")
        elif value not in ("1W", "1M", "3M", "6M", "9M", "1Y"):
            raise ValueError("delta horizon must be one of 1W, 1M, 3M, 6M, 9M, 1Y")

        return value

    @field_validator("news_horizon", mode="before")
    @classmethod
    def validate_news_horizon(cls, value: Any) -> Any:
        if not isinstance(value, str):
            raise ValueError("news horizon must be a string")
        elif value not in ("1W", "1M", "3M"):
            raise ValueError("news horizon must be one of 1W, 1M, 3M")

        return value


@tool(
    description="Given a list of stock ID's, a boolean indicating whether to buy or sell, the "
    "investment horizon and delta horizon, and the number of wanted stocks, returns a list of "
    "stock ID's that are recommended to buy or sell as it is indicated. If no stock ID's are "
    "provided, searching from the S&P 500 stocks.",
    category=ToolCategory.STOCK,
    tool_registry=ToolRegistry,
)
async def get_recommended_stocks(
    args: GetRecommendedStocksInput, context: PlanRunContext
) -> List[StockID]:
    # NOTE: You can't use `get_dummy()` to create a dummy context for this tool because it requires
    # an actual user ID to make the gRPC call (specifically for PA SVC)

    logger = get_prefect_logger(__name__)

    if args.stock_ids:
        if len(args.stock_ids) < args.num_stocks_to_return:
            raise ValueError(
                "The number of stocks to return is greater than the number of stocks provided."
            )
        elif len(args.stock_ids) == args.num_stocks_to_return:
            logger.warning(
                "The number of stocks to return is equal to the number of stocks provided. Return directly"  # noqa
            )
            return args.stock_ids
        stock_ids = args.stock_ids
    else:
        # we perhaps can store the SP500 stocks as a log output but not for now as they are GBI IDs
        await tool_log(log="No stock IDs provided. Using S&P 500 stocks.", context=context)
        stock_ids = await get_stock_universe(  # type: ignore
            args=GetStockUniverseInput(universe_name="SPDR S&P 500 ETF Trust"), context=context
        )

    ism_resp = await get_all_stock_investment_policies(context.user_id)
    ism_id = None
    if ism_resp.investment_policies:
        ism = max(ism_resp.investment_policies, key=lambda x: x.last_updated.ToDatetime())
        ism_id = ism.investment_policy_id.id
        await tool_log(log=f'Using Investment Style "{ism.name}" to search stocks', context=context)

    settings_blob = copy.deepcopy(SETTINGS_TEMPLATE)
    settings_blob["ism_settings"]["ism_id"] = ism_id
    settings_blob["gbi_ids"] = [stock.gbi_id for stock in stock_ids]
    settings_blob["ism_settings"]["match_labels"] = [
        "Perfect Match",
        "Strong Match",
        "Medium Match",
    ]

    if args.buy:
        settings_blob["rating_settings"]["boundary"]["lb"] = 2.5
        settings_blob["news_settings"]["sentiment_boundaries"] = [{"lb": 0, "ub": 1}]
    else:
        settings_blob["rating_settings"]["boundary"]["ub"] = 2.5
        settings_blob["news_settings"]["sentiment_boundaries"] = [{"lb": -1, "ub": 0}]

    resp = await get_temporary_discover_block_data(
        context.user_id, settings_blob, args.horizon, args.delta_horizon
    )
    if len(resp.rows) < args.num_stocks_to_return:
        # TODO: We can loose the constraints, but this is already very loose. What to do?
        raise ValueError("Cannot find enough stocks to meet the requirement.")

    return await StockID.from_gbi_id_list(
        [row.gbi_id for row in resp.rows[: args.num_stocks_to_return]]
    )
