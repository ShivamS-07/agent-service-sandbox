import pandas as pd
from pa_portfolio_service_proto_v1.workspace_pb2 import WorkspaceAuth

from agent_service.external.pa_svc_client import (
    get_all_holdings_in_workspace,
    get_all_workspaces,
)
from agent_service.io_types.table import (
    STOCK_ID_COL_NAME_DEFAULT,
    Table,
    TableColumnMetadata,
    TableColumnType,
)
from agent_service.tool import ToolArgs, ToolCategory, ToolRegistry, tool
from agent_service.types import PlanRunContext
from agent_service.utils.prefect import get_prefect_logger
from agent_service.utils.stock_metadata import get_stock_metadata

logger = get_prefect_logger(__name__)

PortfolioID = str


class GetPortfolioWorkspaceHoldingsInput(ToolArgs):
    portfolio_id: PortfolioID


@tool(
    description=(
        "This function returns a list of stocks and the weight at which they are held in a  specific portfolio. "
        "Use this function if you want return all the stocks in a portfolio given a portfolio Id."
        "Do not use this function to find portfolio names or if no portfolio Id is present"
    ),
    category=ToolCategory.STOCK,
    tool_registry=ToolRegistry,
    is_visible=False,
)
async def get_portfolio_holdings(
    args: GetPortfolioWorkspaceHoldingsInput, context: PlanRunContext
) -> Table:
    workspace = await get_all_holdings_in_workspace(context.user_id, args.portfolio_id)
    gbi_ids = [holding.gbi_id for holding in workspace.holdings]
    stock_meta = await get_stock_metadata(gbi_ids=gbi_ids)
    data = {
        STOCK_ID_COL_NAME_DEFAULT: [stock_meta[holding.gbi_id] for holding in workspace.holdings],
        "Weight": [holding.weight for holding in workspace.holdings],
    }
    df = pd.DataFrame(data)
    table = Table.from_df_and_cols(
        data=df,
        columns=[
            TableColumnMetadata(label=STOCK_ID_COL_NAME_DEFAULT, col_type=TableColumnType.STOCK),
            TableColumnMetadata(label="Weight", col_type=TableColumnType.FLOAT),
        ],
    )
    return table


class GetPortfolioInput(ToolArgs):
    portfolio_name: str


@tool(
    description=(
        "This function returns a portfolio id given a portfolio name or mention (e.g. my portfolio). "
        "It can be used when the client mentions a 'portfolio' keyword in the chat "
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
        return perfect_matches[0].workspace_id.id

    # If more than 1 perfect matches, return the one which edited most recently
    if len(perfect_matches) > 1:
        sorted_perfect_matches = sorted(
            perfect_matches,
            key=lambda x: x.last_updated.seconds if x.last_updated else x.created_at.seconds,
            reverse=True,
        )
        return sorted_perfect_matches[0].workspace_id.id

    # If no perfect matches, return the user owned portfolio which edited most recently
    if len(user_owned_portfolios) > 0:
        sorted_user_owned_portfolios = sorted(
            user_owned_portfolios,
            key=lambda x: x.last_updated.seconds if x.last_updated else x.created_at.seconds,
            reverse=True,
        )
        return sorted_user_owned_portfolios[0].workspace_id.id

    # If no perfect matches and no user owned portfolios, return first portfolio id
    return workspaces[0].workspace_id.id
