import datetime
from typing import Any, List, Optional

import pandas as pd
from pa_portfolio_service_proto_v1.pa_service_common_messages_pb2 import TimeDelta
from pa_portfolio_service_proto_v1.watchlist_pb2 import StockWithWeight
from pa_portfolio_service_proto_v1.workspace_pb2 import WorkspaceAuth
from pydantic import field_validator

from agent_service.external.pa_backtest_svc_client import (
    get_stock_performance_for_date_range,
    get_stocks_sector_performance_for_date_range,
)
from agent_service.external.pa_svc_client import (
    get_all_holdings_in_workspace,
    get_all_workspaces,
    get_full_strategy_info,
)
from agent_service.io_types.dates import DateRange
from agent_service.io_types.stock import StockID
from agent_service.io_types.table import (
    STOCK_ID_COL_NAME_DEFAULT,
    StockTable,
    Table,
    TableColumnMetadata,
    TableColumnType,
)
from agent_service.tool import ToolArgs, ToolCategory, ToolRegistry, tool
from agent_service.tools.tool_log import tool_log
from agent_service.types import PlanRunContext
from agent_service.utils.constants import get_B3_prefix
from agent_service.utils.date_utils import convert_horizon_to_days
from agent_service.utils.postgres import get_psql
from agent_service.utils.prefect import get_prefect_logger

PortfolioID = str
# Get the postgres connection
db = get_psql()


class GetPortfolioWorkspaceHoldingsInput(ToolArgs):
    portfolio_id: PortfolioID


@tool(
    description=(
        "This function returns a list of stocks and the weight at which they are held in a specific portfolio. "
        "Use this function if you want return all the stocks in a portfolio given a portfolio Id. "
        "Do not use this function to find portfolio names or if no portfolio Id is present. "
        "If you only need a list of stocks without weights, MAKE SURE you use the "
        "`get_stock_identifier_list_from_table` function on the table output by this function!"
    ),
    category=ToolCategory.STOCK,
    tool_registry=ToolRegistry,
    is_visible=False,
)
async def get_portfolio_holdings(
    args: GetPortfolioWorkspaceHoldingsInput, context: PlanRunContext
) -> StockTable:
    logger = get_prefect_logger(__name__)
    workspace = await get_all_holdings_in_workspace(context.user_id, args.portfolio_id)
    gbi_ids = [holding.gbi_id for holding in workspace.holdings]

    logger.info(f"found {len(gbi_ids)} holdings")

    stock_ids = await StockID.from_gbi_id_list(gbi_ids)
    gbi_id2_stock_id = {s.gbi_id: s for s in stock_ids}
    data = {
        STOCK_ID_COL_NAME_DEFAULT: [
            gbi_id2_stock_id[holding.gbi_id] for holding in workspace.holdings
        ],
        "Weight": [holding.weight for holding in workspace.holdings],
    }
    df = pd.DataFrame(data)
    table = StockTable.from_df_and_cols(
        data=df,
        columns=[
            TableColumnMetadata(label=STOCK_ID_COL_NAME_DEFAULT, col_type=TableColumnType.STOCK),
            TableColumnMetadata(label="Weight", col_type=TableColumnType.FLOAT),
        ],
    )
    await tool_log(
        f"Found {len(stock_ids)} holdings in portfolio", context=context, associated_data=table
    )
    return table


class GetPortfolioInput(ToolArgs):
    portfolio_name: str


@tool(
    description=(
        "This function returns a portfolio id given a portfolio name or mention (e.g. my portfolio). "
        "It MUST be used when the client mentions any 'portfolio' in the request. "
        "This function will try to match the given name with the portfolio names for that clients "
        "and return the closest match. "
    ),
    category=ToolCategory.PORTFOLIO,
    tool_registry=ToolRegistry,
    is_visible=True,
)
async def convert_portfolio_mention_to_portfolio_id(
    args: GetPortfolioInput, context: PlanRunContext
) -> PortfolioID:
    logger = get_prefect_logger(__name__)
    # Use PA Service to get all portfolios for the user
    workspaces = await get_all_workspaces(user_id=context.user_id)

    # Find portfolios with the perfect matched names
    perfect_matches = []
    user_owned_portfolios = []
    for workspace in workspaces:
        if str(args.portfolio_name).lower() in str(workspace.name).lower():
            perfect_matches.append(workspace)
        if workspace.user_auth_level == WorkspaceAuth.WORKSPACE_AUTH_OWNER:
            user_owned_portfolios.append(workspace)

    # If only 1 perfect match, return the id
    if len(perfect_matches) == 1:
        logger.info(f"only 1 perfect match: {perfect_matches[0]}")

        portfolio = perfect_matches[0]

    # If more than 1 perfect matches, return the one which edited most recently
    elif len(perfect_matches) > 1:
        sorted_perfect_matches = sorted(
            perfect_matches,
            key=lambda x: x.last_updated.seconds if x.last_updated else x.created_at.seconds,
            reverse=True,
        )
        logger.info(f"More than 1 perfect match, most recent: {sorted_perfect_matches[0]}")

        portfolio = sorted_perfect_matches[0]

    # If no perfect matches, return the user owned portfolio which edited most recently
    elif len(user_owned_portfolios) > 0:
        sorted_user_owned_portfolios = sorted(
            user_owned_portfolios,
            key=lambda x: x.last_updated.seconds if x.last_updated else x.created_at.seconds,
            reverse=True,
        )
        logger.info(f"no perfect matches, most recent: {sorted_user_owned_portfolios[0]}")
        portfolio = sorted_user_owned_portfolios[0]
    else:
        # If no perfect matches and no user owned portfolios, return first portfolio id
        logger.info(
            "no perfect matches and no user owned portfolios,"
            f" return first portfolio id: {workspaces[0]}"
        )
        portfolio = workspaces[0]

    base_url = f"{get_B3_prefix()}/dashboard/portfolios/summary"
    portfolio_name_with_link_markdown = (
        f"[{portfolio.name}]({base_url}/{portfolio.workspace_id.id})"
    )
    await tool_log(log=f"Portfolio found: {portfolio_name_with_link_markdown}", context=context)
    return portfolio.workspace_id.id


def map_input_to_closest_horizon(input_horizon: str, supported_horizons: List[str]) -> str:
    if input_horizon in supported_horizons:
        return input_horizon

    input_days = convert_horizon_to_days(input_horizon)
    supported_horizon_to_days = {
        horizon: convert_horizon_to_days(horizon) for horizon in supported_horizons
    }

    min_pair = min(supported_horizon_to_days.items(), key=lambda x: abs(x[1] - input_days))
    return min_pair[0]


TIME_DELTA_MAP = {
    "1W": TimeDelta.TIME_DELTA_ONE_WEEK,
    "1M": TimeDelta.TIME_DELTA_ONE_MONTH,
    "3M": TimeDelta.TIME_DELTA_THREE_MONTHS,
    "6M": TimeDelta.TIME_DELTA_SIX_MONTHS,
    "9M": TimeDelta.TIME_DELTA_NINE_MONTHS,
    "1Y": TimeDelta.TIME_DELTA_ONE_YEAR,
}


class GetPortfolioPerformanceInput(ToolArgs):
    portfolio_id: PortfolioID
    performance_level: str = "overall"
    date_range: Optional[DateRange] = None
    sector_performance_horizon: Optional[str] = "1M"  # 1W, 1M, 3M, 6M, 9M, 1Y

    @field_validator("performance_level", mode="before")
    @classmethod
    def validate_performance_level(cls, value: Any) -> Any:
        if not isinstance(value, str):
            raise ValueError("performance level must be a string")

        if value not in ["overall", "stock", "sector"]:
            raise ValueError("performance level must be one of ('overall', 'stock', 'sector')")
        return value

    @field_validator("sector_performance_horizon", mode="before")
    @classmethod
    def validate_sector_performance_horizon(cls, value: Any) -> Any:
        if not isinstance(value, str):
            raise ValueError("sector_performance_horizon must be a string")

        return map_input_to_closest_horizon(
            value.upper(), supported_horizons=["1W", "1M", "3M", "6M", "9M", "1Y"]
        )


@tool(
    description=(
        "This function returns the performance of a portfolio given a portfolio id "
        "and a performance level and date range. "
        "\nThe performance level MUST one of  ('overall', 'stock', 'sector')."
        "\nThe date range is optional and defaults to the last month."
        "\nThe sector performance horizon is optional and defaults to 1 month. "
        "sector_performance_horizon must be one of ('1W', '1M', '3M', '6M', '9M', '1Y')."
        "\nWhen the performance level is 'overall', it returns the overall performance of the portfolio, "
        "including the monthly returns and the returns versus benchmark. "
        "Table schema for overall performance level: "
        "month: string, return: float, return-vs-benchmark: float"
        "\nWhen the performance level is 'stock', it returns the performance of each stock in the portfolio. "
        "Table schema for stock performance level: "
        "stock: StockID, return: float "
        "\nWhen the performance level is 'sector', it returns the performance of each sector in the portfolio. "
        "Table schema for sector performance level: "
        "sector: string, return: float,weight: float, weighted-return: float"
    ),
    category=ToolCategory.PORTFOLIO,
    tool_registry=ToolRegistry,
    is_visible=True,
)
async def get_portfolio_performance(
    args: GetPortfolioPerformanceInput, context: PlanRunContext
) -> Table:
    # if no date range is provided, use the last month as date range
    if args.date_range is None:
        date_range = DateRange(
            start_date=datetime.date.today() - datetime.timedelta(days=30),
            end_date=datetime.date.today(),
        )
    else:
        date_range = args.date_range

    # get the linked_portfolio_id
    linked_portfolio_id = str(db.get_workspace_linked_id(args.portfolio_id))

    # get the performance for overall performance level
    if args.performance_level == "overall":
        # get the full strategy info for the linked_portfolio_id
        portfolio_details = await get_full_strategy_info(context.user_id, linked_portfolio_id)
        performance_details = portfolio_details.backtest_results.performance_info

        # Create a DataFrame for the monthly returns
        data_length = len(performance_details.monthly_gains.headers)
        df = pd.DataFrame(
            {
                "month": list(performance_details.monthly_gains.headers),
                "return": [
                    v.float_val for v in performance_details.monthly_gains.row_values[0].values
                ],
                "return-vs-benchmark": [
                    v.float_val
                    for v in performance_details.monthly_gains_v_benchmark.row_values[0].values
                ],
            },
            index=range(data_length),
        )
        # create a Table
        table = Table.from_df_and_cols(
            data=df,
            columns=[
                TableColumnMetadata(label="month", col_type=TableColumnType.STRING),
                TableColumnMetadata(label="return", col_type=TableColumnType.FLOAT),
                TableColumnMetadata(label="return-vs-benchmark", col_type=TableColumnType.FLOAT),
            ],
        )
    elif args.performance_level == "stock":
        # get portfolio holdings
        portfolio_holdings_table: StockTable = await get_portfolio_holdings(  # type: ignore
            GetPortfolioWorkspaceHoldingsInput(portfolio_id=args.portfolio_id), context
        )
        portfolio_holdings_df = portfolio_holdings_table.to_df()
        # get gbi_ids
        gbi_ids = [stock.gbi_id for stock in portfolio_holdings_df[STOCK_ID_COL_NAME_DEFAULT]]
        # get the stock performance for the date range
        stock_performance = await get_stock_performance_for_date_range(
            gbi_ids=gbi_ids,
            start_date=date_range.start_date,
            end_date=date_range.end_date,
            user_id=context.user_id,
        )
        # Create a DataFrame for the stock performance
        data = {
            STOCK_ID_COL_NAME_DEFAULT: await StockID.from_gbi_id_list(gbi_ids),
            "return": [stock.performance for stock in stock_performance.stock_performance_list],
            "portfolio-weight": portfolio_holdings_df["Weight"].values,
        }
        df = pd.DataFrame(data)
        df["weighted-return"] = (df["return"] * df["portfolio-weight"]).values
        # sort the DataFrame by weighted-return
        df = df.sort_values(by="weighted-return", ascending=False)
        # create a Table
        table = StockTable.from_df_and_cols(
            data=df,
            columns=[
                TableColumnMetadata(
                    label=STOCK_ID_COL_NAME_DEFAULT, col_type=TableColumnType.STOCK
                ),
                TableColumnMetadata(label="return", col_type=TableColumnType.FLOAT),
                TableColumnMetadata(label="portfolio-weight", col_type=TableColumnType.FLOAT),
                TableColumnMetadata(label="weighted-return", col_type=TableColumnType.FLOAT),
            ],
        )
    elif args.performance_level == "sector":
        # get portfolio holdings
        portfolio_holdings_table: StockTable = await get_portfolio_holdings(  # type: ignore
            GetPortfolioWorkspaceHoldingsInput(portfolio_id=args.portfolio_id), context
        )
        portfolio_holdings_df = portfolio_holdings_table.to_df()
        gbi_ids = [stock.gbi_id for stock in portfolio_holdings_df[STOCK_ID_COL_NAME_DEFAULT]]
        weights = portfolio_holdings_df["Weight"].values
        # convert dict to list of StockAndWeight
        stocks_and_weights = [
            StockWithWeight(
                gbi_id=gbi_id,
                weight=weight,
            )
            for gbi_id, weight in zip(gbi_ids, weights)
        ]
        # map the sector_performance_horizon to TimeDelta
        if args.sector_performance_horizon is None:
            time_delta = TimeDelta().TIME_DELTA_ONE_MONTH
        else:
            time_delta = TIME_DELTA_MAP[args.sector_performance_horizon]

        # get the stock performance for the date range
        sector_performance = await get_stocks_sector_performance_for_date_range(
            user_id=context.user_id,
            stocks_and_weights=stocks_and_weights,
            time_delta=time_delta,
        )

        # Create a DataFrame for the stock performance
        data = {
            "sector": [sector.sector_name for sector in sector_performance],
            "return": [sector.sector_performance for sector in sector_performance],
            "weight": [sector.sector_weight for sector in sector_performance],
            "weighted-return": [
                sector.weighted_sector_performance for sector in sector_performance
            ],
        }

        df = pd.DataFrame(data)
        # sort the DataFrame by weighted-return
        df = df.sort_values(by="weighted-return", ascending=False)
        # create a Table
        table = StockTable.from_df_and_cols(
            data=df,
            columns=[
                TableColumnMetadata(label="sector", col_type=TableColumnType.STRING),
                TableColumnMetadata(label="return", col_type=TableColumnType.FLOAT),
                TableColumnMetadata(label="weight", col_type=TableColumnType.FLOAT),
                TableColumnMetadata(label="weighted-return", col_type=TableColumnType.FLOAT),
            ],
        )
    else:
        raise ValueError("Invalid performance level")
    return table
