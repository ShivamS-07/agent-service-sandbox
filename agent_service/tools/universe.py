import datetime
from typing import Any, List

import numpy as np
import pandas as pd
from pydantic import field_validator

from agent_service.external.feature_svc_client import get_return_for_stocks
from agent_service.io_types.dates import DateRange
from agent_service.io_types.table import (
    STOCK_ID_COL_NAME_DEFAULT,
    StockID,
    StockTable,
    Table,
    TableColumnMetadata,
    TableColumnType,
)
from agent_service.tool import ToolArgs, ToolCategory, ToolRegistry, tool
from agent_service.tools.portfolio import get_sector_for_stock_ids
from agent_service.tools.stocks import (
    GetStockUniverseInput,
    StockIdentifierLookupInput,
    get_stock_info_for_universe,
    get_stock_universe_table_from_universe_company_id,
    stock_identifier_lookup,
)
from agent_service.types import PlanRunContext


class GetUniversePerformanceInput(ToolArgs):
    universe_name: str
    performance_level: str = "overall"
    date_range: DateRange = DateRange(
        start_date=datetime.date.today() - datetime.timedelta(days=30),
        end_date=datetime.date.today(),
    )

    @field_validator("performance_level", mode="before")
    @classmethod
    def validate_performance_level(cls, value: Any) -> Any:
        if not isinstance(value, str):
            raise ValueError("performance level must be a string")

        if value not in ["overall", "security", "sector"]:
            raise ValueError("performance level must be one of ('overall', 'security', 'sector')")
        return value


@tool(
    description=(
        "This function returns the performance of a universe/benchmark given universe name, "
        "performance level, and date range. "
        "\nThe performance level MUST be one of ('overall', 'security', 'sector')."
        "\nThe date range is optional and defaults to the last month."
        "\nWhen the performance level is 'overall', it returns the performance of the universe as a whole. "
        "Table schema for overall performance level: "
        "Security: StockID, return: float "
        "\nWhen the performance level is 'security', it returns the performance of each security in the universe. "
        "Table schema for security performance level: "
        "Security: StockID, return: float "
        "\nWhen the performance level is 'sector', it returns the performance of each sector in the universe. "
        "Table schema for sector performance level: "
        "sector: string, weight: float, weighted-return: float"
    ),
    category=ToolCategory.STATISTICS,
    tool_registry=ToolRegistry,
    is_visible=False,
)
async def get_universe_performance(
    args: GetUniversePerformanceInput, context: PlanRunContext
) -> Table:
    # get the universe id from the universe name
    etf_stock = await get_stock_info_for_universe(
        GetStockUniverseInput(universe_name=args.universe_name), context
    )
    universe_spiq_company_id = etf_stock["spiq_company_id"]

    # get the universe stocks and weight as table
    universe_holdings_table = await get_stock_universe_table_from_universe_company_id(
        universe_spiq_company_id, context
    )

    universe_holdings_df = universe_holdings_table.to_df()
    # get gbi_ids
    gbi_ids = [stock.gbi_id for stock in universe_holdings_df[STOCK_ID_COL_NAME_DEFAULT]]
    weights = list(universe_holdings_df["Weight"].values)
    if args.performance_level == "security":
        table = await get_performance_security_level(
            gbi_ids=gbi_ids,
            weights=weights,
            date_range=args.date_range,
            context=context,
        )
    elif args.performance_level == "sector":
        table = await get_performance_sector_level(
            gbi_ids=gbi_ids,
            weights=weights,
            date_range=args.date_range,
            context=context,
        )

    elif args.performance_level == "overall":
        table = await get_performance_overall_level(
            universe_name=args.universe_name,
            date_range=args.date_range,
            context=context,
        )

    return table


async def get_performance_security_level(
    gbi_ids: List[int], weights: List[float], date_range: DateRange, context: PlanRunContext
) -> Table:
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
            STOCK_ID_COL_NAME_DEFAULT: await StockID.from_gbi_id_list(gbi_ids),
            "weight": weights,
            "return": returns,
        }
    )
    df["weighted-return"] = (df["return"].astype(float) * df["weight"].astype(float)).values
    # sort the DataFrame by weighted-return
    df = df.sort_values(by="weighted-return", ascending=False)
    # create a Table
    table = StockTable.from_df_and_cols(
        data=df,
        columns=[
            TableColumnMetadata(label=STOCK_ID_COL_NAME_DEFAULT, col_type=TableColumnType.STOCK),
            TableColumnMetadata(label="weight", col_type=TableColumnType.PERCENT),
            TableColumnMetadata(label="return", col_type=TableColumnType.PERCENT),
            TableColumnMetadata(label="weighted-return", col_type=TableColumnType.PERCENT),
        ],
    )
    return table


async def get_performance_sector_level(
    gbi_ids: List[int], weights: List[float], date_range: DateRange, context: PlanRunContext
) -> Table:
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
        data={
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
    universe_name: str, date_range: DateRange, context: PlanRunContext
) -> Table:
    # we can treat the universe as a stock to get the performance
    # get the gbi_id for universe
    universe_stockid_obj: StockID = await stock_identifier_lookup(  # type: ignore
        StockIdentifierLookupInput(stock_name=universe_name),
        context,
    )
    # get the universe performance for the date range
    performance_map = await get_return_for_stocks(
        gbi_ids=[universe_stockid_obj.gbi_id],
        start_date=date_range.start_date,
        end_date=date_range.end_date,
        user_id=context.user_id,
    )
    # Create a DataFrame for the universe performance
    data = {
        STOCK_ID_COL_NAME_DEFAULT: [universe_stockid_obj],
        "return": [performance_map.get(universe_stockid_obj.gbi_id, np.nan)],
    }
    df = pd.DataFrame(data)
    # create a Table
    table = StockTable.from_df_and_cols(
        data=df,
        columns=[
            TableColumnMetadata(label=STOCK_ID_COL_NAME_DEFAULT, col_type=TableColumnType.STOCK),
            TableColumnMetadata(label="return", col_type=TableColumnType.PERCENT),
        ],
    )
    return table
