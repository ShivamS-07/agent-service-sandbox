import datetime
from typing import Any, Optional

import pandas as pd
from pa_portfolio_service_proto_v1.pa_service_common_messages_pb2 import TimeDelta
from pa_portfolio_service_proto_v1.watchlist_pb2 import StockWithWeight
from pydantic import field_validator

from agent_service.external.pa_backtest_svc_client import (
    get_stock_performance_for_date_range,
    get_stocks_sector_performance_for_date_range,
)
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
from agent_service.tools.portfolio import TIME_DELTA_MAP, map_input_to_closest_horizon
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
        "This function returns the performance of a universe/benchmark given a universe name "
        "and a performance level and date range. "
        "\nThe performance level MUST one of  ('overall', 'stock', 'sector')."
        "\nThe date range is optional and defaults to the last month."
        "\nThe sector performance horizon is optional and defaults to 1 month. "
        "sector_performance_horizon must be one of ('1W', '1M', '3M', '6M', '9M', '1Y')."
        "sector_performance_horizon MUST be provided when the performance level is 'sector'."
        "\nWhen the performance level is 'overall', it returns the performance of the universe as a whole. "
        "\nWhen the performance level is 'stock', it returns the performance of each stock in the universe. "
        "Table schema for stock performance level: "
        "stock: StockID, return: float "
        "\nWhen the performance level is 'sector', it returns the performance of each sector in the universe. "
        "Table schema for sector performance level: "
        "sector: string, return: float,weight: float, weighted-return: float"
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
    weights = universe_holdings_df["Weight"].values
    if args.performance_level == "stock":
        # get the stock performance for the date range
        stock_performance = await get_stock_performance_for_date_range(
            gbi_ids=gbi_ids,
            start_date=args.date_range.start_date,
            user_id=context.user_id,
        )
        # Create a DataFrame for the stock performance
        data = {
            STOCK_ID_COL_NAME_DEFAULT: await StockID.from_gbi_id_list(gbi_ids),
            "weight": universe_holdings_df["Weight"].values,
            "return": [stock.performance for stock in stock_performance.stock_performance_list],
        }
        df = pd.DataFrame(data)
        df["weight"] = df["weight"] * 100
        df["return"] = df["return"] * 100
        df["weighted-return"] = (df["return"] * df["weight"] / 100).values
        # sort the DataFrame by weighted-return
        df = df.sort_values(by="weighted-return", ascending=False)
        # create a Table
        table = StockTable.from_df_and_cols(
            data=df,
            columns=[
                TableColumnMetadata(
                    label=STOCK_ID_COL_NAME_DEFAULT, col_type=TableColumnType.STOCK
                ),
                TableColumnMetadata(label="weight", col_type=TableColumnType.FLOAT),
                TableColumnMetadata(label="return", col_type=TableColumnType.FLOAT),
                TableColumnMetadata(label="weighted-return", col_type=TableColumnType.FLOAT),
            ],
        )

    elif args.performance_level == "sector":

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
            "sector": [sector.sector_name for sector in sector_performance],  # type: ignore
            "weight": [sector.sector_weight for sector in sector_performance],
            "return": [sector.sector_performance for sector in sector_performance],
            "weighted-return": [
                sector.weighted_sector_performance for sector in sector_performance
            ],
        }

        df = pd.DataFrame(data)
        df["weight"] = df["weight"] * 100
        df["return"] = df["return"] * 100
        df["weighted-return"] = df["weighted-return"] * 100
        df = df.sort_values(by="weighted-return", ascending=False)
        # create a Table
        table = StockTable.from_df_and_cols(
            data=df,
            columns=[
                TableColumnMetadata(label="sector", col_type=TableColumnType.STRING),
                TableColumnMetadata(label="weight", col_type=TableColumnType.FLOAT),
                TableColumnMetadata(label="return", col_type=TableColumnType.FLOAT),
                TableColumnMetadata(label="weighted-return", col_type=TableColumnType.FLOAT),
            ],
        )
    elif args.performance_level == "overall":
        # we can treat the universe as a stock to get the performance
        # get the gbi_id for universe
        universe_stockid_obj: StockID = await stock_identifier_lookup(  # type: ignore
            StockIdentifierLookupInput(stock_name=args.universe_name),
            context,
        )
        # get the universe performance for the date range
        overall_performance = await get_stock_performance_for_date_range(
            gbi_ids=[universe_stockid_obj.gbi_id],
            start_date=args.date_range.start_date,
            user_id=context.user_id,
        )
        # Create a DataFrame for the universe performance
        data = {
            STOCK_ID_COL_NAME_DEFAULT: [universe_stockid_obj],
            "return": [
                universe.performance for universe in overall_performance.stock_performance_list
            ],
        }
        df = pd.DataFrame(data)
        df["return"] = df["return"] * 100
        # create a Table
        table = StockTable.from_df_and_cols(
            data=df,
            columns=[
                TableColumnMetadata(
                    label=STOCK_ID_COL_NAME_DEFAULT, col_type=TableColumnType.STOCK
                ),
                TableColumnMetadata(label="return", col_type=TableColumnType.FLOAT),
            ],
        )

    return table
