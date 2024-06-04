import pandas as pd

from agent_service.external.pa_svc_client import get_all_holdings_in_workspace
from agent_service.io_types.table import (
    STOCK_ID_COL_NAME_DEFAULT,
    Table,
    TableColumn,
    TableColumnType,
)
from agent_service.tool import ToolArgs, ToolCategory, ToolRegistry, tool
from agent_service.types import PlanRunContext
from agent_service.utils.prefect import get_prefect_logger

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
    data = {
        "Stock": [holding.gbi_id for holding in workspace.holdings],
        "Weight": [holding.weight for holding in workspace.holdings],
    }
    df = pd.DataFrame(data)
    table = Table(
        data=df,
        columns=[
            TableColumn(label=STOCK_ID_COL_NAME_DEFAULT, col_type=TableColumnType.STOCK),
            TableColumn(label="Weight", col_type=TableColumnType.FLOAT),
        ],
    )
    return table
