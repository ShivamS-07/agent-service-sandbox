import asyncio
import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from pa_portfolio_service_proto_v1.workspace_pb2 import StockAndWeight, WorkspaceAuth
from pydantic import field_validator

from agent_service.external.dal_svc_client import get_dal_client
from agent_service.external.feature_svc_client import get_return_for_stocks
from agent_service.external.investment_policy_svc import (
    get_portfolio_investment_policy_for_workspace,
)
from agent_service.external.pa_svc_client import (
    get_all_holdings_in_workspace,
    get_all_workspaces,
    get_full_strategy_info,
    get_transitive_holdings_from_stocks_and_weights,
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
from agent_service.tools.feature_data import get_latest_price
from agent_service.tools.tool_log import tool_log
from agent_service.types import PlanRunContext
from agent_service.utils.constants import get_B3_prefix
from agent_service.utils.postgres import get_psql
from agent_service.utils.prefect import get_prefect_logger
from agent_service.utils.tool_diff import (
    add_task_id_to_stocks_history,
    get_prev_run_info,
)

PORTFOLIO_ADD_STOCK_DIFF = "{company} was added to the portfolio: {portfolio}"
PORTFOLIO_REMOVE_STOCK_DIFF = "{company} was removed from the portfolio: {portfolio}"

PortfolioID = str
# Get the postgres connection
db = get_psql()
logger = get_prefect_logger(__name__)


class GetPortfolioHoldingsInput(ToolArgs):
    portfolio_id: PortfolioID
    expand_etfs: bool = False
    date_range: DateRange = DateRange(
        start_date=datetime.date.today() - datetime.timedelta(days=30),
        end_date=datetime.date.today(),
    )
    fetch_stats: Optional[bool] = True


async def get_workspace_name(user_id: str, portfolio_id: str) -> Optional[str]:
    workspaces = await get_all_workspaces(user_id=user_id)
    for workspace in workspaces:
        if portfolio_id == workspace.workspace_id.id:
            return workspace.name

    return None


@tool(
    description=(
        "This function returns a list of stocks and the weight at which they are held in a specific portfolio, "
        "and optionally their price and performance. "
        "Use this function if you want return all the stocks in a portfolio given a portfolio Id. "
        "\n- A PortfolioID is not a portfolio name. A PortfolioID is not a stock identifier. "
        "\n- This tool should only be used to get portfolio holdings, "
        "for ETF holdings you should use get_stock_universe tool instead. "
        "Do not use this function to find portfolio names or if no portfolio Id is present. "
        "If you only need a list of stocks without weights, MAKE SURE you use the "
        "`get_stock_identifier_list_from_table` function on the table output by this function! "
        "\n- There is a fetch_stats flag which is optional and defaults to True. "
        "If the eventual answer is just holdings, the flag should be on. "
        "If the holdings are being fetched and then something else is being done with them later, "
        "the flag should be turned off to avoid automatically fetching these extra columns."
        "\n- Output will contain these columns: 'Stock', 'Weight', and optionally 'Price' and 'Performance', "
        "and optionally, if fetch_stats == True, 'Price' and 'Performance'."
        "\n- 'expand_etfs' is an optional parameter that defaults to False. "
        "If `expand_etfs` set to True, the function will expand ETFs into stock level "
        "and adjust the weights accordingly. "
        "'date_range' is an optional parameter that defaults to the last month. "
        "It will be used to get the performance of the stocks in the portfolio. "
    ),
    category=ToolCategory.STOCK,
    tool_registry=ToolRegistry,
    is_visible=False,
)
async def get_portfolio_holdings(
    args: GetPortfolioHoldingsInput, context: PlanRunContext
) -> StockTable:
    workspace = await get_all_holdings_in_workspace(context.user_id, args.portfolio_id)

    workspace_name = await get_workspace_name(
        user_id=context.user_id, portfolio_id=args.portfolio_id
    )
    if not workspace_name:
        workspace_name = args.portfolio_id

    gbi_ids = [holding.gbi_id for holding in workspace.holdings]
    weights = [holding.weight for holding in workspace.holdings]
    weights_map = {holding.gbi_id: holding.weight for holding in workspace.holdings}
    if args.expand_etfs:
        await tool_log("Expanding ETFs in portfolio holdings", context=context)
        # get all the stocks in the portfolio ETFs
        weighted_securities: List[StockAndWeight] = [
            StockAndWeight(gbi_id=gbi_id, weight=weight) for gbi_id, weight in zip(gbi_ids, weights)
        ]
        expanded_weighted_securities: List[StockAndWeight] = (
            await get_transitive_holdings_from_stocks_and_weights(
                user_id=context.user_id, weighted_securities=weighted_securities
            )
        )
        gbi_ids = [holding.gbi_id for holding in expanded_weighted_securities]
        weights_map = {holding.gbi_id: holding.weight for holding in expanded_weighted_securities}

    stock_list = await StockID.from_gbi_id_list(gbi_ids)
    data = {
        STOCK_ID_COL_NAME_DEFAULT: stock_list,
        "Weight": [weights_map.get(holding.gbi_id, np.nan) for holding in stock_list],
    }

    if args.fetch_stats:
        # get latest price + stock performance data
        gbi_id2_price, stock_performance_map = await asyncio.gather(
            get_latest_price(context, stock_list),
            get_return_for_stocks(
                gbi_ids=gbi_ids,
                start_date=args.date_range.start_date,
                end_date=args.date_range.end_date,
                user_id=context.user_id,
            ),
        )

        # convert stock performance to dict for mapping id to performance

        data.update(
            {
                "Price": [gbi_id2_price.get(holding.gbi_id, np.nan) for holding in stock_list],
                "Return": [
                    stock_performance_map.get(holding.gbi_id, np.nan) for holding in stock_list
                ],
            }
        )

    df = pd.DataFrame(data)
    # normalize weights
    df["Weight"] = df["Weight"] / df["Weight"].sum()
    df = df.sort_values(by="Weight", ascending=False)
    columns = [
        TableColumnMetadata(label=STOCK_ID_COL_NAME_DEFAULT, col_type=TableColumnType.STOCK),
        TableColumnMetadata(label="Weight", col_type=TableColumnType.PERCENT),
    ]
    if args.fetch_stats:
        columns.extend(
            [
                TableColumnMetadata(label="Price", col_type=TableColumnType.FLOAT),
                TableColumnMetadata(label="Return", col_type=TableColumnType.PERCENT),
            ]
        )
    table = StockTable.from_df_and_cols(data=df, columns=columns)
    await tool_log(
        f"Found {len(gbi_ids)} holdings in portfolio", context=context, associated_data=table
    )

    try:  # since everything associated with diffing is optional, put in try/except
        # we need to add the task id to all runs, including the first one, so we can track changes
        if context.task_id:
            stock_list = add_task_id_to_stocks_history(stock_list, context.task_id)
            if context.diff_info is not None:
                # 2nd arg is the name of the function we are in
                prev_run_info = await get_prev_run_info(context, "get_portfolio_holdings")
                if prev_run_info is not None:
                    prev_output_table: StockTable = prev_run_info.output  # type:ignore
                    prev_output = prev_output_table.get_stocks()
                    # corner case here where S&P 500 change causes output to change, but not going to
                    # bother with it on first pass
                    curr_stock_set = set(stock_list)
                    prev_stock_set = set(prev_output)
                    added_stocks = curr_stock_set - prev_stock_set
                    removed_stocks = prev_stock_set - curr_stock_set
                    context.diff_info[context.task_id] = {
                        "added": {
                            added_stock: PORTFOLIO_ADD_STOCK_DIFF.format(
                                company=added_stock.company_name, portfolio=workspace_name
                            )
                            for added_stock in added_stocks
                        },
                        "removed": {
                            removed_stock: PORTFOLIO_REMOVE_STOCK_DIFF.format(
                                company=removed_stock.company_name, portfolio=workspace_name
                            )
                            for removed_stock in removed_stocks
                        },
                    }

    except Exception as e:
        logger.warning(f"Error creating diff info from previous run: {e}")

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


class GetPortfolioPerformanceInput(ToolArgs):
    portfolio_id: PortfolioID
    performance_level: str = "security"
    date_range: DateRange = DateRange(
        start_date=datetime.date.today() - datetime.timedelta(days=30),
        end_date=datetime.date.today(),
    )

    @field_validator("performance_level", mode="before")
    @classmethod
    def validate_performance_level(cls, value: Any) -> Any:
        if not isinstance(value, str):
            raise ValueError("performance level must be a string")

        if value not in ["overall", "stock", "sector", "daily", "security"]:
            raise ValueError(
                "performance level must be one of ('overall', 'stock', 'security', 'sector', 'daily')"
            )
        return value


@tool(
    description=(
        "This function returns the performance of a portfolio given a portfolio id "
        "and a performance level and date range. "
        "\nThe performance level MUST one of  ('overall', 'stock', 'sector', 'daily', 'security'). "
        "The default performance level is 'security'. "
        "\nThe date range is optional and defaults to the last month."
        "\nWhen the performance level is 'overall', it returns the overall performance of the portfolio, "
        "which is the sum of the weighted returns of all securities in the portfolio for the given date range. "
        "Table schema for overall performance level: "
        "portfolio: string, return: float"
        "\nWhen the performance level is 'stock', it returns the performance of all stocks in the portfolio. "
        "this will expand all ETFs into stock level. "
        "Only use 'stock' if you want to expand ETFs into stock level and see the performance of each stock. "
        "Table schema for stock performance level: "
        "Security: StockID, return: float, portfolio-weight: float, weighted-return: float"
        "\nWhen the performance level is 'security', it returns the performance of all securitys in the portfolio. "
        "Use 'security' if you want to see the performance of each security in the portfolio. If they are ETFs, "
        "they will not be expanded into stock level. "
        "Table schema for security performance level: "
        "Security: StockID, return: float, portfolio-weight: float, weighted-return: float"
        "\nWhen the performance level is 'sector', it returns the performance of each sector in the portfolio. "
        "Table schema for sector performance level: "
        "sector: string,weight: float, weighted-return: float"
        "\nWhen the performance level is 'daily', it returns the daily returns of the portfolio. "
        "Table schema for daily performance level: "
        "date: string, return: float, return-vs-benchmark: float"
    ),
    category=ToolCategory.PORTFOLIO,
    tool_registry=ToolRegistry,
    is_visible=True,
)
async def get_portfolio_performance(
    args: GetPortfolioPerformanceInput, context: PlanRunContext
) -> Table:

    if args.performance_level == "daily":
        # get the linked_portfolio_id
        linked_portfolio_id: str = await get_linked_portfolio_id(context.user_id, args.portfolio_id)
        table = await get_performance_daily_level(
            user_id=context.user_id,
            linked_portfolio_id=linked_portfolio_id,
            date_range=args.date_range,
        )

    elif args.performance_level == "stock":
        table = await get_performance_security_level(
            portfolio_id=args.portfolio_id,
            date_range=args.date_range,
            context=context,
            expand_etfs=True,
        )

    elif args.performance_level == "security":
        table = await get_performance_security_level(
            portfolio_id=args.portfolio_id,
            date_range=args.date_range,
            context=context,
            expand_etfs=False,
        )

    elif args.performance_level == "sector":
        table = await get_performance_sector_level(
            portfolio_id=args.portfolio_id, date_range=args.date_range, context=context
        )

    elif args.performance_level == "overall":
        table = await get_performance_overall_level(
            portfolio_id=args.portfolio_id, date_range=args.date_range, context=context
        )

    else:
        raise ValueError(f"Invalid performance level: {args.performance_level}")

    await tool_log(
        log="Universe performance table retrieved.", context=context, associated_data=table
    )

    return table


class GetPortfolioBenchmarkHoldingsInput(ToolArgs):
    portfolio_id: PortfolioID
    expand_etfs: bool = False


@tool(
    description=(
        "This function returns a list of stocks and the weight at which they are held in the benchmark "
        "related to a specific portfolio given a portfolio Id. "
        "A PortfolioID is not a portfolio name. "
        "A PortfolioID is not a stock identifier. "
        "'expand_etfs' is an optional parameter that defaults to False. "
        "If set to True, the function will expand ETFs into stock level and adjust the weights accordingly. "
        "Only set this flag to True if user explicitly wants to see the expanded ETFs or on stock level. "
    ),
    category=ToolCategory.STOCK,
    tool_registry=ToolRegistry,
    is_visible=True,
)
async def get_portfolio_benchmark_holdings(
    args: GetPortfolioBenchmarkHoldingsInput, context: PlanRunContext
) -> StockTable:
    # get policy details
    policy_details = await get_portfolio_investment_policy_for_workspace(
        user_id=context.user_id, workspace_id=args.portfolio_id
    )

    # get benchmark holdings (gbi_ids and weights)
    benchmark_holdings = policy_details.policy.custom_benchmark.weights
    gbi_ids = [holding.gbi_id for holding in benchmark_holdings]
    weights = [holding.weight / 100 for holding in benchmark_holdings]
    weights_map = {holding.gbi_id: holding.weight / 100 for holding in benchmark_holdings}

    if args.expand_etfs:
        await tool_log("Expanding ETFs in benchmark holdings", context=context)
        # get all the stocks in the benchmark ETFs
        weighted_securities: List[StockAndWeight] = [
            StockAndWeight(gbi_id=gbi_id, weight=weight) for gbi_id, weight in zip(gbi_ids, weights)
        ]
        expanded_weighted_securities: List[StockAndWeight] = (
            await get_transitive_holdings_from_stocks_and_weights(
                user_id=context.user_id, weighted_securities=weighted_securities
            )
        )
        gbi_ids = [holding.gbi_id for holding in expanded_weighted_securities]
        weights_map = {holding.gbi_id: holding.weight for holding in expanded_weighted_securities}

    stock_list = await StockID.from_gbi_id_list(gbi_ids)
    data = {
        STOCK_ID_COL_NAME_DEFAULT: stock_list,
        "Weight": [weights_map.get(holding.gbi_id, np.nan) for holding in stock_list],
    }
    df = pd.DataFrame(data)
    df = df.sort_values(by="Weight", ascending=False)
    table = StockTable.from_df_and_cols(
        data=df,
        columns=[
            TableColumnMetadata(label=STOCK_ID_COL_NAME_DEFAULT, col_type=TableColumnType.STOCK),
            TableColumnMetadata(label="Weight", col_type=TableColumnType.FLOAT),
        ],
    )
    await tool_log(
        f"Found {len(gbi_ids)} holdings in benchmark", context=context, associated_data=table
    )
    return table


class GetPortfolioTradesInput(ToolArgs):
    portfolio_id: PortfolioID


@tool(
    description=(
        "This function returns a list of trades made in a specific portfolio, given a portfolio Id "
        "Use this function if you want to retrieve all trades for a portfolio identified by a PortfolioID. "
        "A PortfolioID is not a portfolio name. "
        "A PortfolioID is not a stock identifier. "
        "Do not use this function to find portfolio names or if no portfolio Id is present. "
        "If you need a list of portfolio holdings, MAKE SURE you use the `get_portfolio_holdings` tool! "
        "Output will contain the following columns: Security, Date, Action, Allocation Change. "
    ),
    category=ToolCategory.STOCK,
    tool_registry=ToolRegistry,
    is_visible=False,
)
async def get_portfolio_trades(
    args: GetPortfolioTradesInput, context: PlanRunContext
) -> StockTable:

    logger = get_prefect_logger(__name__)

    workspaces = await get_all_workspaces(
        user_id=context.user_id, workspace_ids=[args.portfolio_id]
    )
    if len(workspaces) == 0:
        raise ValueError(f"Workspace doesn't exist for id: {args.portfolio_id}")
    workspace = workspaces[0]

    # Logic used in web-server
    TODAY = datetime.datetime.now().date()
    BEGIN_SEARCH = TODAY - relativedelta(years=2)
    start_date = BEGIN_SEARCH
    end_date = TODAY

    dal_client = get_dal_client()
    previous_trades = await dal_client.fetch_previous_trades(
        workspace.linked_strategy.model_id.id,
        workspace.linked_strategy.strategy_id.id,
        start_date.isoformat(),
        end_date.isoformat(),
    )

    logger.info(f"found {len(previous_trades)} trades")

    gbi_ids = [trade.gbi_id for trade in previous_trades]

    stock_ids = await StockID.from_gbi_id_list(gbi_ids)
    gbi_id2_stock_id = {s.gbi_id: s for s in stock_ids}
    data = {
        STOCK_ID_COL_NAME_DEFAULT: [gbi_id2_stock_id[trade.gbi_id] for trade in previous_trades],
        "Date": [trade.trade_date for trade in previous_trades],
        "Action": [trade.action for trade in previous_trades],
        "Allocation Change": [trade.allocation_change for trade in previous_trades],
    }
    df = pd.DataFrame(data)
    table = StockTable.from_df_and_cols(
        data=df,
        columns=[
            TableColumnMetadata(label=STOCK_ID_COL_NAME_DEFAULT, col_type=TableColumnType.STOCK),
            TableColumnMetadata(label="Date", col_type=TableColumnType.DATE),
            TableColumnMetadata(label="Action", col_type=TableColumnType.STRING),
            TableColumnMetadata(label="Allocation Change", col_type=TableColumnType.PERCENT),
        ],
    )
    await tool_log(
        f"Found {len(stock_ids)} trades in portfolio", context=context, associated_data=table
    )
    return table


# helper functions


async def get_linked_portfolio_id(user_id: str, portfolio_id: str) -> str:
    linked_portfolio_id: str = ""
    try:
        workspaces = await get_all_workspaces(user_id=user_id)
        for workspace in workspaces:
            if portfolio_id == workspace.workspace_id.id:
                if workspace.linked_strategy:
                    linked_portfolio_id = workspace.linked_strategy.strategy_id.id
                elif workspace.analyze_strategy:
                    linked_portfolio_id = workspace.analyze_strategy.strategy_id.id
    except Exception as e:
        raise ValueError(f"Error getting linked_portfolio_id for {portfolio_id}: {e}")

    return linked_portfolio_id


def get_sector_for_stock_ids(stock_ids: List[int]) -> Dict[int, str]:

    db = get_psql()
    sql = """
    SELECT ms.gbi_security_id, gs."name"
    FROM master_security ms
    left join gic_sector gs
    on CAST(SUBSTRING(CAST(ms.gics AS TEXT), 1, 2) AS INT) = gs.id
    WHERE ms.gbi_security_id = ANY(%(stock_ids)s)
    """
    rows = db.generic_read(
        sql,
        params={
            "stock_ids": stock_ids,
        },
    )
    gbi_id2_sector = {row["gbi_security_id"]: row["name"] for row in rows}
    return gbi_id2_sector


async def get_performance_daily_level(
    user_id: str, linked_portfolio_id: str, date_range: DateRange
) -> Table:
    # get the full strategy info for the linked_portfolio_id
    portfolio_details = await get_full_strategy_info(user_id, linked_portfolio_id)
    chart_info = portfolio_details.backtest_results.chart_info.chart_info
    # Create a DataFrame for the daily returns
    data_length = len(list(chart_info))
    df = pd.DataFrame(
        {
            "date": [x.date for x in chart_info],
            "return": [x.value for x in chart_info],
            "return-vs-benchmark": [x.benchmark for x in chart_info],
        },
        index=range(data_length),
    )
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")
    # filter based on date range
    df = df[
        (df["date"].dt.date >= date_range.start_date) & (df["date"].dt.date <= date_range.end_date)
    ]
    # transform df into date, field, value format
    df = df.melt(id_vars=["date"], var_name="field", value_name="value")
    # create a Table
    table = Table.from_df_and_cols(
        data=df,
        columns=[
            TableColumnMetadata(label="date", col_type=TableColumnType.DATE),
            TableColumnMetadata(label="field", col_type=TableColumnType.STRING),
            TableColumnMetadata(label="value", col_type=TableColumnType.PERCENT),
        ],
    )
    return table


async def get_performance_security_level(
    portfolio_id: PortfolioID,
    date_range: DateRange,
    context: PlanRunContext,
    expand_etfs: bool = False,
) -> Table:
    # get portfolio holdings
    portfolio_holdings_table: StockTable = await get_portfolio_holdings(  # type: ignore
        GetPortfolioHoldingsInput(
            portfolio_id=portfolio_id, expand_etfs=expand_etfs, fetch_stats=False
        ),
        context,
    )
    portfolio_holdings_df = portfolio_holdings_table.to_df()
    # get gbi_ids
    gbi_ids = [stock.gbi_id for stock in portfolio_holdings_df[STOCK_ID_COL_NAME_DEFAULT]]
    # get the stock performance for the date range
    performance_map = await get_return_for_stocks(
        gbi_ids=gbi_ids,
        start_date=date_range.start_date,
        end_date=date_range.end_date,
        user_id=context.user_id,
    )
    returns = [performance_map.get(gbi_id, np.nan) for gbi_id in gbi_ids]
    weights = portfolio_holdings_df["Weight"].values
    # Create a DataFrame for the stock performance
    df = pd.DataFrame(
        {
            STOCK_ID_COL_NAME_DEFAULT: await StockID.from_gbi_id_list(gbi_ids),
            "return": returns,
            "portfolio-weight": weights,
        }
    )
    df["weighted-return"] = (
        df["return"].astype(float) * df["portfolio-weight"].astype(float)
    ).values
    # sort the DataFrame by weighted-return
    df = df.sort_values(by="weighted-return", ascending=False)
    # create a Table
    table = StockTable.from_df_and_cols(
        data=df,
        columns=[
            TableColumnMetadata(label=STOCK_ID_COL_NAME_DEFAULT, col_type=TableColumnType.STOCK),
            TableColumnMetadata(label="return", col_type=TableColumnType.PERCENT),
            TableColumnMetadata(label="portfolio-weight", col_type=TableColumnType.PERCENT),
            TableColumnMetadata(label="weighted-return", col_type=TableColumnType.PERCENT),
        ],
    )
    return table


async def get_performance_sector_level(
    portfolio_id: PortfolioID, date_range: DateRange, context: PlanRunContext
) -> Table:
    # get portfolio holdings
    portfolio_holdings_table: StockTable = await get_portfolio_holdings(  # type: ignore
        GetPortfolioHoldingsInput(portfolio_id=portfolio_id, expand_etfs=True, fetch_stats=False),
        context,
    )
    portfolio_holdings_df = portfolio_holdings_table.to_df()
    gbi_ids = [stock.gbi_id for stock in portfolio_holdings_df[STOCK_ID_COL_NAME_DEFAULT]]
    weights = portfolio_holdings_df["Weight"].values
    sector_map = get_sector_for_stock_ids(gbi_ids)
    sectors = [sector_map.get(gbi_id, "No Sector") for gbi_id in gbi_ids]

    # get the stock performance for the date range
    performance_map = await get_return_for_stocks(
        gbi_ids=gbi_ids,
        start_date=date_range.start_date,
        end_date=date_range.end_date,
        user_id=context.user_id,
    )
    returns = [performance_map.get(gbi_id, np.nan) for gbi_id in gbi_ids]
    # Create a DataFrame for the stock performance
    df = pd.DataFrame(
        {
            "sector": sectors,
            "return": returns,
            "weight": weights,
        }
    )
    df["weighted-return"] = (df["return"].astype(float) * df["weight"].astype(float)).values

    # group by sector and calculate the weighted return
    df = df.groupby("sector", as_index=False).agg({"weight": "sum", "weighted-return": "sum"})

    # sort the DataFrame by weighted-return
    df = df.sort_values(by="weighted-return", ascending=False)
    # create a Table
    table = StockTable.from_df_and_cols(
        data=df,
        columns=[
            TableColumnMetadata(label="sector", col_type=TableColumnType.STRING),
            TableColumnMetadata(label="weight", col_type=TableColumnType.PERCENT),
            TableColumnMetadata(label="weighted-return", col_type=TableColumnType.PERCENT),
        ],
    )
    return table


async def get_performance_overall_level(
    portfolio_id: PortfolioID, date_range: DateRange, context: PlanRunContext
) -> Table:
    # calculate total return for the given date range
    # get portfolio holdings
    portfolio_holdings_table: StockTable = await get_portfolio_holdings(  # type: ignore
        GetPortfolioHoldingsInput(portfolio_id=portfolio_id, expand_etfs=False, fetch_stats=False),
        context,
    )
    portfolio_holdings_df = portfolio_holdings_table.to_df()
    # get gbi_ids
    gbi_ids = [stock.gbi_id for stock in portfolio_holdings_df[STOCK_ID_COL_NAME_DEFAULT]]
    # get the stock performance for the date range
    performance_map = await get_return_for_stocks(
        gbi_ids=gbi_ids,
        start_date=date_range.start_date,
        end_date=date_range.end_date,
        user_id=context.user_id,
    )
    returns = [performance_map.get(gbi_id, np.nan) for gbi_id in gbi_ids]
    weights = portfolio_holdings_df["Weight"].values
    # Create a DataFrame for the stock performance
    df = pd.DataFrame(
        {
            STOCK_ID_COL_NAME_DEFAULT: await StockID.from_gbi_id_list(gbi_ids),
            "return": returns,
            "portfolio-weight": weights,
        }
    )
    total_return = np.nansum(df["return"].astype(float) * df["portfolio-weight"].astype(float))

    # Create a DataFrame for the universe performance
    data = {"portfolio": ["portfolio"], "return": [total_return]}
    df = pd.DataFrame(data)
    # create a Table
    table = StockTable.from_df_and_cols(
        data=df,
        columns=[
            TableColumnMetadata(label="portfolio", col_type=TableColumnType.STRING),
            TableColumnMetadata(label="return", col_type=TableColumnType.PERCENT),
        ],
    )
    return table
