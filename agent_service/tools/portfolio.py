import asyncio
import datetime
import difflib
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from data_access_layer.core.dao.portfolio.pa_dao import PortfolioAnalysisDAO
from dateutil.relativedelta import relativedelta
from pa_portfolio_service_proto_v1.marketplace_messages_pb2 import (
    ListAllAuthorizedStrategiesResponse,
)
from pa_portfolio_service_proto_v1.workspace_pb2 import (
    StockAndWeight,
    WorkspaceAuth,
    WorkspaceMetadata,
)
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
    get_list_all_authorized_strategies,
    get_transitive_holdings_from_stocks_and_weights,
)
from agent_service.GPT.constants import GPT4_O_MINI, NO_PROMPT
from agent_service.GPT.requests import GPT
from agent_service.io_type_utils import ComplexIOBase, io_type
from agent_service.io_types.dates import DateRange
from agent_service.io_types.stock import StockID
from agent_service.io_types.table import (
    STOCK_ID_COL_NAME_DEFAULT,
    StockTable,
    Table,
    TableColumnMetadata,
    TableColumnType,
)
from agent_service.planner.errors import EmptyOutputError, NotFoundError
from agent_service.tool import (
    ToolArgMetadata,
    ToolArgs,
    ToolCategory,
    default_tool_registry,
    tool,
)
from agent_service.tools.feature_data import get_latest_price
from agent_service.tools.tool_log import tool_log
from agent_service.types import PlanRunContext
from agent_service.utils.constants import get_B3_prefix
from agent_service.utils.date_utils import get_now_utc
from agent_service.utils.gpt_logging import GptJobIdType, GptJobType, create_gpt_context
from agent_service.utils.postgres import get_psql
from agent_service.utils.prefect import get_prefect_logger
from agent_service.utils.prompt_utils import Prompt
from agent_service.utils.tool_diff import (
    add_task_id_to_stocks_history,
    get_prev_run_info,
)

PORTFOLIO_ADD_STOCK_DIFF = "{company} was added to the portfolio: {portfolio}"
PORTFOLIO_REMOVE_STOCK_DIFF = "{company} was removed from the portfolio: {portfolio}"
PORTFOLIO_PERFORMANCE_LEVELS = ["overall", "stock", "sector", "daily", "security"]
PORTFOLIO_HOLDING_TABLE_NAME_EXPANDED = "Portfolio Holdings - ETFs Expanded"
PORTFOLIO_HOLDING_TABLE_NAME_NOT_EXPANDED = "Portfolio Holdings - ETFs Not Expanded"
PORTFOLIO_PERFORMANCE_TABLE_BASE_NAME = "Portfolio Performance - "

BENCHMARK_PERFORMANCE_LEVELS = ["stock", "security"]
BENCHMARK_PERFORMANCE_TABLE_BASE_NAME = "Benchmark Performance - "
BENCHMARK_HOLDING_TABLE_NAME_EXPANDED = "Benchmark Holdings - ETFs Expanded"
BENCHMARK_HOLDING_TABLE_NAME_NOT_EXPANDED = "Benchmark Holdings - ETFs Not Expanded"

PortfolioID = str

# The required name match strengths for portfolios
# roughly this is like a string edit distance normalized to 0.0-1.0
# https://docs.python.org/3/library/difflib.html#difflib.get_close_matches
STD_MATCH_SIMILARITY = 0.7
STRICT_MATCH_SIMILARITY = 0.8
WEAK_MATCH_SIMILARITY = 0.6

# Get the postgres connection
db = get_psql()
logger = get_prefect_logger(__name__)


class GetPortfolioHoldingsInput(ToolArgs):
    portfolio_id: PortfolioID
    expand_etfs: bool = False
    fetch_default_stats: bool = False
    fetch_stats: Optional[bool] = None  # old name for fetch_default_stats
    date_range: Optional[DateRange] = None  # ignored

    arg_metadata = {
        "fetch_stats": ToolArgMetadata(hidden_from_planner=True),
        "date_range": ToolArgMetadata(hidden_from_planner=True),
    }


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
        "\n- There is a fetch_default_stats input which must be set to True or False. "
        " fetch_default_stats should only be set to True if "
        " the holdings table returned by this function mostly answers the user's question"
        " and is then passed to the prepare_output function,"
        " If the question is more involved and requires more than 4 steps in your plan"
        " or client asks for specific statistics by name"
        " then set fetch_default_stats = False"
        " when fetch_default_stats is True  pricing and returns information will be added into the table."
        " As this is useful default information to have when looking at the holdings."
        "\n- Output will contain these columns: 'Stock', 'Weight' "
        "and optionally, if fetch_default_stats == True, 'Price' and 'Performance' columns also."
        "\n- 'expand_etfs' is an optional parameter that defaults to False. "
        "If `expand_etfs` set to True, the function will expand ETFs into stock level "
        "and adjust the weights accordingly. "
    ),
    category=ToolCategory.PORTFOLIO,
    tool_registry=default_tool_registry(),
    is_visible=False,
)
async def get_portfolio_holdings(
    args: GetPortfolioHoldingsInput, context: PlanRunContext
) -> StockTable:
    # respect the old name
    if args.fetch_stats is not None:
        args.fetch_default_stats = args.fetch_stats

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
        # get all the stocks in the portfolio ETFs
        weighted_securities: List[StockAndWeight] = [
            StockAndWeight(gbi_id=gbi_id, weight=weight) for gbi_id, weight in zip(gbi_ids, weights)
        ]
        expanded_weighted_securities: List[
            StockAndWeight
        ] = await get_transitive_holdings_from_stocks_and_weights(
            user_id=context.user_id, weighted_securities=weighted_securities
        )
        gbi_ids = [holding.gbi_id for holding in expanded_weighted_securities]
        weights_map = {holding.gbi_id: holding.weight for holding in expanded_weighted_securities}

    stock_list = await StockID.from_gbi_id_list(gbi_ids)
    data = {
        STOCK_ID_COL_NAME_DEFAULT: stock_list,
        "Weight": [weights_map.get(holding.gbi_id, np.nan) for holding in stock_list],
    }

    if args.fetch_default_stats:
        date_range: DateRange = DateRange(
            start_date=datetime.date.today() - datetime.timedelta(days=30),
            end_date=datetime.date.today(),
        )

        # get latest price + stock performance data
        gbi_id2_price, stock_performance_map = await asyncio.gather(
            get_latest_price(context, stock_list),
            get_return_for_stocks(
                gbi_ids=gbi_ids,
                start_date=date_range.start_date,
                end_date=date_range.end_date,
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

    columns = [
        TableColumnMetadata(label=STOCK_ID_COL_NAME_DEFAULT, col_type=TableColumnType.STOCK),
        TableColumnMetadata(label="Weight", col_type=TableColumnType.PERCENT),
    ]
    if args.fetch_default_stats:
        columns.extend(
            [
                TableColumnMetadata(label="Price", col_type=TableColumnType.FLOAT),
                TableColumnMetadata(label="Return", col_type=TableColumnType.PERCENT),
            ]
        )
    table = StockTable.from_df_and_cols(data=df, columns=columns)

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
    table.title = (
        PORTFOLIO_HOLDING_TABLE_NAME_EXPANDED
        if args.expand_etfs
        else PORTFOLIO_HOLDING_TABLE_NAME_NOT_EXPANDED
    )
    return table


class GetPortfolioInput(ToolArgs):
    portfolio_name: str
    portfolio_uuid: Optional[str] = None


@tool(
    description=(
        "This function returns a portfolio id object given a portfolio name "
        "or mention (e.g. my portfolio) as well as a portfolio's UUID if available. "
        "It MUST be used when the client mentions any 'portfolio' in the request. "
        "This function will try to match the given name with the portfolio names for that clients "
        "and return the closest match. "
        "portfolio_id should be included in addition to the name ONLY if a portfolio's "
        "UUID is explicitly mentioned in user input! "
        "In that case, you still MUST call this function to resolve the ID to an object. "
        "E.g. 'My portfolio' (Portfolio ID: <some UUID>)."
    ),
    category=ToolCategory.PORTFOLIO,
    tool_registry=default_tool_registry(),
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
    all_portfolios = defaultdict(list)
    for workspace in workspaces:
        if args.portfolio_uuid and workspace.workspace_id.id == args.portfolio_uuid:
            return args.portfolio_uuid
        if str(args.portfolio_name).lower() == str(workspace.name).lower():
            perfect_matches.append(workspace)

        all_portfolios[str(workspace.name).lower()].append(workspace)

        if workspace.user_auth_level == WorkspaceAuth.WORKSPACE_AUTH_OWNER:
            user_owned_portfolios.append(workspace)

    # TODO we could add a fallback for matching by portfolio_id if str looks like a UUID
    partial_matches = []
    # if generically talking about a portfolio revert to the default most recent portfolio
    if len(perfect_matches) == 0 and not is_generic_portfolio_search(args.portfolio_name):
        # find a close textual match
        search_str = str(args.portfolio_name).lower()
        cutoff = STD_MATCH_SIMILARITY

        # be more strict if portfolio is in the search
        if "portfolio" in search_str:
            cutoff = STRICT_MATCH_SIMILARITY

        close_names = difflib.get_close_matches(
            search_str, list(all_portfolios.keys()), n=1, cutoff=cutoff
        )

        # if portfolio was in the name but couldnt find it
        # remove portfolio and try again less strict
        if not close_names and "portfolio" in search_str:
            search_str = search_str.replace("portfolio", "")
            cutoff = WEAK_MATCH_SIMILARITY
            close_names = difflib.get_close_matches(
                search_str, list(all_portfolios.keys()), n=1, cutoff=cutoff
            )

        if close_names:
            partial_matches = all_portfolios[close_names[0]]
        else:
            # the target portfolio is named 'foo portfolio'
            # but we only searched for 'foo'
            best_similarity = 0.0
            best_name = None
            for k, v in all_portfolios.items():
                close = difflib.SequenceMatcher(
                    None, search_str, k.replace("portfolio", "")
                ).ratio()
                if close >= cutoff and close > best_similarity:
                    best_similarity = close
                    best_name = k

            if best_name:
                partial_matches = all_portfolios[best_name]

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

    elif len(partial_matches) == 1:
        portfolio = partial_matches[0]
        logger.info(f"found 1 partial match {args.portfolio_name}: {portfolio}")

    # If more than 1 perfect matches, return the one which edited most recently
    elif len(partial_matches) > 1:
        sorted_matches = sorted(
            partial_matches,
            key=lambda x: x.last_updated.seconds if x.last_updated else x.created_at.seconds,
            reverse=True,
        )
        logger.info(
            f"More than 1 partial match, {args.portfolio_name}:"
            f" most recent: {sorted_matches[0]}"
        )

        portfolio = sorted_matches[0]

    elif is_generic_portfolio_search(args.portfolio_name):
        if len(user_owned_portfolios) > 0:
            # If no partial matches, return the user owned portfolio which was edited most recently
            sorted_user_owned_portfolios = sorted(
                user_owned_portfolios,
                key=lambda x: x.last_updated.seconds if x.last_updated else x.created_at.seconds,
                reverse=True,
            )
            logger.info(f"no partial matches, most recent: {sorted_user_owned_portfolios[0]}")
            portfolio = sorted_user_owned_portfolios[0]
        elif len(workspaces) > 0:
            # If no partial matches and no user owned portfolios, return first portfolio id
            logger.info(
                "no partial matches and no user owned portfolios,"
                f" return first portfolio id: {workspaces[0]}"
            )
            portfolio = workspaces[0]
        else:
            raise NotFoundError("User does not have access to any portfolios")
    else:
        # Final check, use GPT as a fallback for portfolio and watchlist lookup
        # Need to get the section of the string that portfolio refers to
        portfolio = await portfolio_match_by_gpt(
            context=context,
            portfolio_name=args.portfolio_name,
            portfolios=workspaces,
        )
        if not portfolio:
            raise NotFoundError(f"No portfolio found matching: '{args.portfolio_name}'")

    base_url = f"{get_B3_prefix()}/dashboard/portfolios/summary"
    portfolio_name_with_link_markdown = (
        f"[{portfolio.name}]({base_url}/{portfolio.workspace_id.id})"
    )

    if portfolio.name.lower() == args.portfolio_name.lower():
        await tool_log(log=f"Portfolio found: {portfolio_name_with_link_markdown}", context=context)
    else:
        await tool_log(
            log=f"Intepreting '{args.portfolio_name}' as: {portfolio_name_with_link_markdown}",
            context=context,
        )

    return portfolio.workspace_id.id


def is_generic_portfolio_search(search_str: Optional[str]) -> bool:
    """
    return true if search str generically refers to a portfolio an not a specific name
    """
    if not search_str:
        return True
    search_str = search_str.lower()
    search_str = re.sub(r"\b's\b", " ", search_str)
    search_str = re.sub(r"\bmy\b", " ", search_str)
    search_str = re.sub(r"\byour\b", " ", search_str)
    search_str = re.sub(r"\bour\b", " ", search_str)
    search_str = re.sub(r"\ba\b", " ", search_str)
    search_str = re.sub(r"\ban\b", " ", search_str)
    search_str = re.sub(r"\bany\b", " ", search_str)
    search_str = re.sub(r"\bthe\b", " ", search_str)
    search_str = re.sub(r"\bthat\b", " ", search_str)
    search_str = re.sub(r"\bthis\b", " ", search_str)
    search_str = re.sub(r"\bsome\b", " ", search_str)
    search_str = re.sub(r"\bportfolios\b", " ", search_str)
    search_str = re.sub(r"\bportfolio\b", " ", search_str)
    search_str = re.sub(r"\bport\b", " ", search_str)
    search_str = re.sub(r"\bclient\b", " ", search_str)

    # remove any non alphanumeric chars
    search_str = re.sub(r"\W+", "", search_str, flags=re.MULTILINE)
    search_str = search_str.replace("_", "")

    return not search_str


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

        if value not in PORTFOLIO_PERFORMANCE_LEVELS:
            raise ValueError(f"performance level must be one of {PORTFOLIO_PERFORMANCE_LEVELS}")
        return value


@tool(
    description=(
        "This function returns the performance of a portfolio given a portfolio id "
        "and a performance level and date range. "
        f"\nThe performance level MUST one of {PORTFOLIO_PERFORMANCE_LEVELS}. "
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
    tool_registry=default_tool_registry(),
    enabled=False,
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
        log=f"Portfolio performance on {args.performance_level}-level retrieved.",
        context=context,
        associated_data=table,
    )

    table.title = PORTFOLIO_PERFORMANCE_TABLE_BASE_NAME + args.performance_level

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
    category=ToolCategory.PORTFOLIO,
    tool_registry=default_tool_registry(),
    enabled=False,
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
        # get all the stocks in the benchmark ETFs
        weighted_securities: List[StockAndWeight] = [
            StockAndWeight(gbi_id=gbi_id, weight=weight) for gbi_id, weight in zip(gbi_ids, weights)
        ]
        expanded_weighted_securities: List[
            StockAndWeight
        ] = await get_transitive_holdings_from_stocks_and_weights(
            user_id=context.user_id, weighted_securities=weighted_securities
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
    table.title = (
        BENCHMARK_HOLDING_TABLE_NAME_EXPANDED
        if args.expand_etfs
        else BENCHMARK_HOLDING_TABLE_NAME_NOT_EXPANDED
    )
    return table


class GetPortfolioBenchmarkPerformanceInput(ToolArgs):
    portfolio_id: PortfolioID
    date_range: DateRange = DateRange(
        start_date=datetime.date.today() - datetime.timedelta(days=30),
        end_date=datetime.date.today(),
    )
    performance_level: str = "stock"

    @field_validator("performance_level", mode="before")
    @classmethod
    def validate_performance_level(cls, value: Any) -> Any:
        if not isinstance(value, str):
            raise ValueError("performance level must be a string")

        if value not in BENCHMARK_PERFORMANCE_LEVELS:
            raise ValueError(f"performance level must be one of {PORTFOLIO_PERFORMANCE_LEVELS}")
        return value


@tool(
    description=(
        "This function returns the performance of the benchmark related to a specific portfolio given a portfolio Id. "
        f"The performance level MUST be one of {BENCHMARK_PERFORMANCE_LEVELS}. "
        "\nWhen the performance level is 'stock', it returns the performance of all stocks in the benchmark. "
        "this will expand all ETFs into stock level. "
        "Only use 'stock' if you want to expand ETFs into stock level and see the performance of each stock. "
        "Table schema for stock performance level: "
        "Security: StockID, return: float, benchmark-weight: float, weighted-return: float"
        "\nWhen the performance level is 'security', it returns the performance of all securitys in the benchmark. "
        "Use 'security' if you want to see the performance of each security in the benchmark. If they are ETFs, "
        "they will not be expanded into stock level. "
        "Table schema for security performance level: "
        "Security: StockID, return: float, benchmark-weight: float, weighted-return: float"
    ),
    category=ToolCategory.PORTFOLIO,
    tool_registry=default_tool_registry(),
    enabled=False,
)
async def get_portfolio_benchmark_performance(
    args: GetPortfolioBenchmarkPerformanceInput, context: PlanRunContext
) -> StockTable:
    if args.performance_level == "stock":
        expand_etfs = True
    else:
        expand_etfs = False

    benchmark_holdings_table = await get_portfolio_benchmark_holdings(
        GetPortfolioBenchmarkHoldingsInput(portfolio_id=args.portfolio_id, expand_etfs=expand_etfs),
        context,
    )
    benchmark_holdings_df = benchmark_holdings_table.to_df()  # type: ignore
    gbi_ids = [stock.gbi_id for stock in benchmark_holdings_df[STOCK_ID_COL_NAME_DEFAULT]]
    benchmark_stock_performance_map = await get_return_for_stocks(
        gbi_ids=gbi_ids,
        start_date=args.date_range.start_date,
        end_date=args.date_range.end_date,
        user_id=context.user_id,
    )
    returns = [benchmark_stock_performance_map.get(gbi_id, np.nan) for gbi_id in gbi_ids]
    data = {
        STOCK_ID_COL_NAME_DEFAULT: await StockID.from_gbi_id_list(gbi_ids),
        "return": returns,
        "benchmark-weight": benchmark_holdings_df["Weight"].values,
    }
    benchmark_stock_performance_df = pd.DataFrame(data)
    benchmark_stock_performance_df["weighted-return"] = (
        benchmark_stock_performance_df["return"].astype(float)
        * benchmark_stock_performance_df["benchmark-weight"].astype(float)
    ).values
    benchmark_stock_performance_df = benchmark_stock_performance_df.sort_values(
        by="weighted-return", ascending=False
    )
    table = StockTable.from_df_and_cols(
        data=benchmark_stock_performance_df,
        columns=[
            TableColumnMetadata(label=STOCK_ID_COL_NAME_DEFAULT, col_type=TableColumnType.STOCK),
            TableColumnMetadata(label="return", col_type=TableColumnType.PERCENT),
            TableColumnMetadata(label="benchmark-weight", col_type=TableColumnType.PERCENT),
            TableColumnMetadata(label="weighted-return", col_type=TableColumnType.PERCENT),
        ],
    )
    table.title = BENCHMARK_PERFORMANCE_TABLE_BASE_NAME + args.performance_level
    await tool_log(
        f"Found {len(gbi_ids)} benchmark holdings performance.",
        context=context,
        associated_data=table,
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
    category=ToolCategory.PORTFOLIO,
    tool_registry=default_tool_registry(),
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
        raise NotFoundError(f"Workspace doesn't exist for id: {args.portfolio_id}")
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
            portfolio_id=portfolio_id, expand_etfs=expand_etfs, fetch_default_stats=False
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
        GetPortfolioHoldingsInput(
            portfolio_id=portfolio_id, expand_etfs=True, fetch_default_stats=False
        ),
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
        GetPortfolioHoldingsInput(
            portfolio_id=portfolio_id, expand_etfs=False, fetch_default_stats=False
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


PORTFOLIO_MATCH_PROMPT_STR = """
Given a portfolio name, find the best match from a list of portfolio options.
Consider possible typos, transposed letters, or minor spelling errors when finding the closest match.
Even if the match is not exact, return the portfolio that seems most relevant or closest in meaning.
Only respond with "no_match" if no portfolio name reasonably aligns with the given name.
When returning "no_match", return it without any punctuation or anything.
Here are the portfolios to consider: {portfolios}.
Here is the text to potentially match to one of them: {portfolio_name}.
"""


async def portfolio_match_by_gpt(
    context: PlanRunContext,
    portfolio_name: str,
    portfolios: List[WorkspaceMetadata],
) -> Optional[WorkspaceMetadata]:
    """
    Match a specific "portfolio" mentioned by a prompt to the closest WorkspaceMetadata
    object from the given list of portfolios.
    If there is no connection to be made at all, just return None.
    """
    if not portfolios:
        return None

    gpt_context = create_gpt_context(
        GptJobType.AGENT_TOOLS, context.agent_id, GptJobIdType.AGENT_ID
    )

    llm = GPT(context=gpt_context, model=GPT4_O_MINI)

    PORTFOLIO_MATCH_PROMPT = Prompt(
        name="PORTFOLIO_MATCH_PROMPT", template=PORTFOLIO_MATCH_PROMPT_STR
    )

    portfolio_names = [pf.name for pf in portfolios]
    prompt = PORTFOLIO_MATCH_PROMPT.format(
        portfolio_name=portfolio_name, portfolios=portfolio_names
    )

    result_str = await llm.do_chat_w_sys_prompt(
        main_prompt=prompt,
        sys_prompt=NO_PROMPT,
    )

    logger.info(f"Comparing portfolio name: {portfolio_name} got match: {result_str}")
    # First pass case sensitive
    for portfolio in portfolios:
        if portfolio.name == result_str:
            return portfolio

    # Just in case GPT returned all lowercase, second pass case in-sensitive
    for portfolio in portfolios:
        if portfolio.name.lower() == result_str.lower():
            return portfolio

    return None


####################################################################################################
# Strategy
####################################################################################################
class GetStrategyInput(ToolArgs):
    strategy_name: str
    strategy_uuid: Optional[str] = None


@io_type
class StrategyMetadata(ComplexIOBase):
    model_id: str
    strategy_id: str
    model_name: str
    strategy_name: str
    is_live: bool
    created_date: datetime.datetime


@tool(
    description=(
        "This function returns the metadata of a Strategy given a strategy name or mention (e.g. my strategy) as well as a strategy's UUID if available. It MUST be used when the client mentions any 'strategy' in the request. Here, the 'strategy' also has alias like 'quant strategy', 'B1 strategy', or 'Boosted strategy' that refers to the trading strategies people created for back-testing purpose. This function will try to match the given name with the strategy names for that client and return the closest match. `strategy_uuid` should be included in addition to the name ONLY if a strategy's UUID is explicitly mentioned in user input! In that case, you still MUST call this function to resolve the ID to an object. E.g. 'My strategy' (Strategy ID: <some UUID>)."  # noqa
    ),
    category=ToolCategory.STRATEGY,
    tool_registry=default_tool_registry(),
)
async def convert_strategy_mention_to_strategy(
    args: GetStrategyInput, context: PlanRunContext
) -> StrategyMetadata:
    """
    Get the stocks that are inside a strategy. Note this is not the holdings of a strategy, but rather
    the stocks that are "ranked" by the B1 strategy

    Complex implementation:
    1. If `strategy_id` is provided and can match to a strategy, return the stocks under the strategy.
    2. If `strategy_name` matches exactly to strategies, return the stocks under the most recent LIVE one.
    3. If `strategy_name` matches partially to strategies (contain), return the stocks under the most recent LIVE one.
    4. If no direct match, use `difflib` to calculate the similarity and sort the strategies.
        - If there are close matches (>=0.6), use GPT to find the best one among them.
        - Otherwise, use GPT to find the best one among all strategies.
        Note that experiments show that the order of the names may affect the result, so sort it
        by the similarity before passing to GPT
    """

    logger = get_prefect_logger(__name__)

    logger.info("Getting user's all authorized strategies")
    strategies = await get_list_all_authorized_strategies(context.user_id)
    if not strategies:
        raise EmptyOutputError(message="User has no access to any strategies!")

    await tool_log(log=f"Found {len(strategies)} strategies", context=context)

    lower_strategy_name = args.strategy_name.lower()
    perfect_matches: List[ListAllAuthorizedStrategiesResponse.AuthorizedStrategy] = []
    contain_matches: List[ListAllAuthorizedStrategiesResponse.AuthorizedStrategy] = []
    concat_names = []
    for strategy in strategies:
        if args.strategy_uuid and args.strategy_uuid == strategy.strategy_id.id:
            logger.info(f"Found strategy by id: {strategy.strategy_id.id}")
            perfect_matches.append(strategy)
            break

        concat_name = f"{strategy.model_name} - {strategy.strategy_name}"
        concat_names.append(concat_name)

        if args.strategy_name == concat_name:
            perfect_matches.append(strategy)
        elif lower_strategy_name in concat_name.lower():
            contain_matches.append(strategy)

    num_perfect_matches = len(perfect_matches)
    num_contain_matches = len(contain_matches)
    if num_perfect_matches > 0:
        logger.info(
            f"Found {num_perfect_matches} perfect matched strategies. "
            "Choose the most recent LIVE one."
        )
        strategy = max(
            perfect_matches, key=lambda x: (x.is_strategy_live, x.created_date.ToDatetime())
        )
    elif num_contain_matches > 0:
        logger.info(
            f"Found {num_contain_matches} partial matched strategies. "
            "Choose the most recent LIVE one."
        )
        strategy = max(
            contain_matches, key=lambda x: (x.is_strategy_live, x.created_date.ToDatetime())
        )
    else:
        logger.info("No direct match. Using `difflib` to sort the strategies.")
        name_ratio_pairs = [
            (name, difflib.SequenceMatcher(None, args.strategy_name, name).ratio())
            for name in set(concat_names)
        ]
        name_ratio_pairs.sort(key=lambda x: x[1], reverse=True)
        close_matches = [name for name, ratio in name_ratio_pairs if ratio >= WEAK_MATCH_SIMILARITY]
        if close_matches:
            logger.info(f"Found {len(close_matches)} close match. Using GPT to find the best one.")
            strategy_name = await get_best_matched_strategy_name_by_gpt(
                args.strategy_name, close_matches, context
            )
        else:
            names = [name for name, _ in name_ratio_pairs]
            strategy_name = await get_best_matched_strategy_name_by_gpt(
                args.strategy_name, names, context
            )

        if strategy_name:
            matches = [
                s
                for s, concat_name in zip(strategies, concat_names)
                if concat_name == strategy_name
            ]
            strategy = max(matches, key=lambda x: (x.is_strategy_live, x.created_date.ToDatetime()))
        else:
            logger.warning(
                f"No strategy found for the mention <{args.strategy_name}>. Use the most recent LIVE one."  # noqa
            )
            strategy = max(
                strategies, key=lambda x: (x.is_strategy_live, x.created_date.ToDatetime())
            )

    await tool_log(
        log=(
            f"Found strategy <{strategy.model_name} - {strategy.strategy_name}> "
            f"(id={strategy.strategy_id.id})"
        ),
        context=context,
    )
    return StrategyMetadata(
        model_id=strategy.model_id.id,
        strategy_id=strategy.strategy_id.id,
        model_name=strategy.model_name,
        strategy_name=strategy.strategy_name,
        is_live=strategy.is_strategy_live,
        created_date=strategy.created_date.ToDatetime(),
    )


class GetStrategyStocksInput(ToolArgs):
    strategy_metadata: StrategyMetadata


@tool(
    description="Given the metadata of a strategy, this tool returns a Table of the quantitative rankings for the stocks inside the strategy. It should only be used after the tool `convert_strategy_mention_to_strategy` when the user asks for the stock rankings in a SPECIFIC strategy.",  # noqa
    category=ToolCategory.STRATEGY,
    tool_registry=default_tool_registry(),
)
async def get_strategy_rankings(
    args: GetStrategyStocksInput, context: PlanRunContext
) -> StockTable:
    # TODO: we can support PIT data later
    today = (get_now_utc() + datetime.timedelta(days=1)).date()
    pa_dao = PortfolioAnalysisDAO()
    nc = pa_dao.get_data(
        model_id=args.strategy_metadata.model_id,
        portfolio_id=args.strategy_metadata.strategy_id,
        fields=["rank"],
        start_date=today,
        end_date=today,
        get_most_recent=True,
        need_clean=True,
    )
    if nc is None:
        raise EmptyOutputError(message="No ranking data found for the strategy")

    rank_date: datetime.date = nc.columns[0]
    row_mask = np.where(~np.isnan(nc.np_data[:, 0, 0]))[0]
    gbi_id_to_rank = {int(nc.rows[row_idx]): int(nc.np_data[row_idx, 0, 0]) for row_idx in row_mask}

    await tool_log(log=f"Found {len(gbi_id_to_rank)} stocks ranked on {rank_date}", context=context)

    stock_list = await StockID.from_gbi_id_list(list(gbi_id_to_rank.keys()))
    df = pd.DataFrame(
        data={
            STOCK_ID_COL_NAME_DEFAULT: stock_list,
            "Rank": [gbi_id_to_rank[stock.gbi_id] for stock in stock_list],
        },
    )
    df.sort_values(by="Rank", ignore_index=True, inplace=True)
    df["Rank"] += 1  # 0-based to 1-based

    table = StockTable.from_df_and_cols(
        data=df,
        columns=[
            TableColumnMetadata(label=STOCK_ID_COL_NAME_DEFAULT, col_type=TableColumnType.STOCK),
            TableColumnMetadata(label="Rank", col_type=TableColumnType.INTEGER),
        ],
    )
    table.title = "Strategy Rankings"

    return table


@tool(
    description="Given the metadata of a strategy, this tool returns a Table of the stocks along with their weight percentages inside the strategy. It should only be used after the tool `convert_strategy_mention_to_strategy` when the user asks for the stock holdings in a SPECIFIC strategy.",  # noqa
    category=ToolCategory.STRATEGY,
    tool_registry=default_tool_registry(),
)
async def get_strategy_holdings(
    args: GetStrategyStocksInput, context: PlanRunContext
) -> StockTable:
    # holdings is a bit tricky because it's daily data but the strategy isn't always LIVE
    # `get_most_recent` will default to most recent rebalance date and when the frequency is not
    # daily then the result is unmatched. Thus, we need to fetch the last signalDate and fetch a
    # range of data from that date to today and use the last data point as the most recent one

    today = (get_now_utc() + datetime.timedelta(days=1)).date()
    pa_dao = PortfolioAnalysisDAO()
    dates = pa_dao.get_dates_data(
        model_id=args.strategy_metadata.model_id,
        portfolio_id=args.strategy_metadata.strategy_id,
        end_date=today.isoformat(),
        fields=["signalDates"],
    )
    if not dates or "signalDates" not in dates or not dates["signalDates"]:
        raise EmptyOutputError(message="No dates data found for the strategy")

    start_date = datetime.date.fromisoformat(dates["signalDates"][-1])

    nc = pa_dao.get_data(
        model_id=args.strategy_metadata.model_id,
        portfolio_id=args.strategy_metadata.strategy_id,
        fields=["allocation"],
        start_date=start_date,
        end_date=today,
        need_clean=True,
    )
    if nc is None:
        raise EmptyOutputError(message="No holding data found for the strategy")

    holding_date: datetime.date = nc.columns[-1]
    row_mask = np.where(~np.isnan(nc.np_data[:, -1, 0]))[0]
    gbi_id_to_weight = {int(nc.rows[row_idx]): nc.np_data[row_idx, -1, 0] for row_idx in row_mask}

    await tool_log(log=f"Found {len(gbi_id_to_weight)} holdings on {holding_date}", context=context)

    stock_list = await StockID.from_gbi_id_list(list(gbi_id_to_weight.keys()))
    df = pd.DataFrame(
        data={
            STOCK_ID_COL_NAME_DEFAULT: stock_list,
            "Weight": [gbi_id_to_weight[stock.gbi_id] for stock in stock_list],
        },
    )
    df.sort_values(by="Weight", ascending=False, inplace=True, ignore_index=True)

    table = StockTable.from_df_and_cols(
        data=df,
        columns=[
            TableColumnMetadata(label=STOCK_ID_COL_NAME_DEFAULT, col_type=TableColumnType.STOCK),
            TableColumnMetadata(label="Weight", col_type=TableColumnType.PERCENT),
        ],
    )
    table.title = "Strategy Holdings"

    return table


async def get_best_matched_strategy_name_by_gpt(
    strategy_mention: str,
    strategy_names: List[str],
    context: PlanRunContext,
) -> Optional[str]:
    if len(strategy_names) == 0:
        return None
    elif len(strategy_names) == 1:
        return strategy_names[0]

    prompt_str = "I have an indexed list of strategy names that are case-sensitive, and I'd like you to find the best match for a user-specified strategy from this list. For example, it should look like: '0 <Strategy0>', '1 <Strategy1>'... Return the integer of the index of the strategy whose name matches the following mention the best. Note that the user's input might not be exact or fully accurate, so consider possible typos, transposed letters, or minor spelling errors when finding the closest match. Even if the match is not exact, return the strategy that seems most relevant or closest in meaning. Your response MUST ONLY be the index of the best matched strategy.\nHere is the strategy mention:{strategy_mention}.\nHere are the strategy names:\n{strategy_names}"  # noqa
    main_prompt = Prompt(template=prompt_str, name="B1_STRATEGY_MATCHING")
    filled_main_prompt = main_prompt.format(
        strategy_mention=strategy_mention,
        strategy_names="\n".join([f"{i} <{name}>" for i, name in enumerate(strategy_names)]),
    )

    gpt_context = create_gpt_context(
        GptJobType.AGENT_TOOLS, context.agent_id, GptJobIdType.AGENT_ID
    )
    llm = GPT(gpt_context, GPT4_O_MINI)
    result = await llm.do_chat_w_sys_prompt(filled_main_prompt, sys_prompt=NO_PROMPT, max_tokens=10)

    try:
        return strategy_names[int(result)]
    except Exception as e:
        logger = get_prefect_logger(__name__)
        logger.exception(f"Failed to get the best matched strategy name by GPT: {e}")
        return None
