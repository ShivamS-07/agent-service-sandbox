import datetime
import logging
import unittest
from unittest import IsolatedAsyncioTestCase
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pandas as pd
from google.protobuf.timestamp_pb2 import Timestamp
from pa_portfolio_service_proto_v1.backtest_data_service_pb2 import StockPerformance
from pa_portfolio_service_proto_v1.well_known_types_pb2 import UUID
from pa_portfolio_service_proto_v1.workspace_pb2 import WorkspaceAuth, WorkspaceMetadata

from agent_service.io_types.stock import StockID
from agent_service.io_types.table import (
    STOCK_ID_COL_NAME_DEFAULT,
    StockTable,
    TableColumnMetadata,
    TableColumnType,
)
from agent_service.tools.dates import DateRange
from agent_service.tools.portfolio import (
    GetPortfolioHoldingsInput,
    GetPortfolioInput,
    GetPortfolioPerformanceInput,
    convert_portfolio_mention_to_portfolio_id,
    get_portfolio_holdings,
    get_portfolio_performance,
)
from agent_service.types import PlanRunContext

DATE_RANGE_TEST = DateRange(
    start_date=datetime.date.fromisoformat("2024-01-01"),
    end_date=datetime.date.fromisoformat("2024-06-01"),
)

AAPL = StockID(gbi_id=714, isin="", symbol="AAPL", company_name="Apple")
ERGB = StockID(
    gbi_id=434782, isin="", symbol="ERGB", company_name="ErgoBilt, Inc."
)  # ergonomic chairs
TRQ = StockID(
    gbi_id=19694, isin="", symbol="TRQ", company_name="Turquoise Hill Resources Ltd."
)  # mining company


# @unittest.skip("Skip until DB mocks are implemented")
class TestPortfolioTools(IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        logging.getLogger("asyncio").setLevel(logging.ERROR)

    async def asyncSetUp(self):
        self.context = PlanRunContext.get_dummy()

    @patch("agent_service.tools.portfolio.get_all_workspaces")
    async def test_convert_portfolio_mention_to_portfolio_id(
        self, mock_get_all_workspaces: MagicMock
    ):
        rows = self.create_dummy_workspaces_for_user(self.context.user_id)

        mock_get_all_workspaces.return_value = [
            WorkspaceMetadata(
                workspace_id=UUID(id=row["id"]),
                name=row["name"],
                user_auth_level=row["user_auth_level"],
                last_updated=row["last_updated"],
                created_at=self.to_timestamp(1600000000),
            )
            for row in rows
        ]

        args = GetPortfolioInput(portfolio_name="Portfolio")
        result = await convert_portfolio_mention_to_portfolio_id(args, self.context)
        self.assertEqual(result, rows[0]["id"])

        args = GetPortfolioInput(portfolio_name="my Portfolio")
        result = await convert_portfolio_mention_to_portfolio_id(args, self.context)
        self.assertEqual(result, rows[1]["id"])

        args = GetPortfolioInput(portfolio_name="NonExistant Portfolio")
        result = await convert_portfolio_mention_to_portfolio_id(args, self.context)
        self.assertEqual(result, rows[0]["id"])

    @patch("agent_service.tools.portfolio.get_all_holdings_in_workspace")
    @patch("agent_service.tools.portfolio.get_workspace_name")
    @patch("agent_service.tools.portfolio.get_latest_price")
    @patch("agent_service.tools.portfolio.get_stock_performance_for_date_range")
    async def test_get_portfolio_holdings(
        self,
        mock_get_stock_performance_for_date_range: MagicMock,
        mock_get_latest_price: MagicMock,
        mock_get_workspace_name: MagicMock,
        mock_get_all_holdings_in_workspace: MagicMock,
    ):
        valid_uuid = str(uuid4())
        mock_get_all_holdings_in_workspace_return = MagicMock(
            workspace_id=valid_uuid,
            holdings=[
                MagicMock(
                    gbi_id=AAPL.gbi_id,
                    date=None,
                    weight=0.6,
                ),
                MagicMock(
                    gbi_id=ERGB.gbi_id,
                    date=None,
                    weight=0.4,
                ),
            ],
        )
        mock_get_all_holdings_in_workspace.return_value = mock_get_all_holdings_in_workspace_return

        mock_get_workspace_name.return_value = "Test Workspace"

        mock_get_latest_price.return_value = {AAPL.gbi_id: 215, ERGB.gbi_id: 123}

        mock_get_stock_performance_for_date_range.return_value = MagicMock(
            stock_performance_list=[
                StockPerformance(gbi_id=AAPL.gbi_id, performance=0.23),
                StockPerformance(gbi_id=ERGB.gbi_id, performance=0.04),
            ]
        )

        args = GetPortfolioHoldingsInput(portfolio_id=valid_uuid)
        result = await get_portfolio_holdings(args, self.context)

        expected_portfolio_holdings = StockTable.from_df_and_cols(
            data=pd.DataFrame(
                {
                    STOCK_ID_COL_NAME_DEFAULT: [AAPL, ERGB],
                    "Weight": [60.0, 40.0],
                    "Price": [215, 123],
                    "Performance": [23.0, 4.0],
                }
            ),
            columns=[
                TableColumnMetadata(
                    label=STOCK_ID_COL_NAME_DEFAULT, col_type=TableColumnType.STOCK
                ),
                TableColumnMetadata(label="Weight", col_type=TableColumnType.FLOAT),
                TableColumnMetadata(label="Price", col_type=TableColumnType.FLOAT),
                TableColumnMetadata(label="Performance", col_type=TableColumnType.FLOAT),
            ],
        )
        pd.testing.assert_frame_equal(result.to_df(), expected_portfolio_holdings.to_df())

    @patch("agent_service.tools.portfolio.get_all_workspaces")
    @patch("agent_service.tools.portfolio.get_full_strategy_info")
    @patch("agent_service.tools.portfolio.get_psql")
    async def test_get_portfolio_performance_monthly(
        self,
        mock_get_psql: MagicMock,
        mock_get_full_strategy_info: MagicMock,
        mock_get_all_workspaces: MagicMock,
    ):
        rows = self.create_dummy_workspaces_for_user(self.context.user_id)

        mock_get_all_workspaces.return_value = [
            WorkspaceMetadata(
                workspace_id=UUID(id=row["id"]),
                name=row["name"],
                user_auth_level=row["user_auth_level"],
                last_updated=row["last_updated"],
                created_at=self.to_timestamp(1600000000),
            )
            for row in rows
        ]
        # Create a mock database and mock psql
        mock_db = MagicMock()
        mock_db.get_workspace_linked_id.return_value = str(uuid4())
        mock_get_psql.return_value = mock_db

        # Mock portfolio details response
        mock_portfolio_details = MagicMock()
        mock_performance_info = MagicMock()
        mock_performance_info.monthly_gains.headers = ["Jan", "Feb", "YTD"]
        mock_performance_info.monthly_gains.row_values = [
            MagicMock(
                values=[
                    MagicMock(float_val=1.0),
                    MagicMock(float_val=2.0),
                    MagicMock(float_val=2.0),
                ]
            )
        ]
        mock_performance_info.monthly_gains_v_benchmark.row_values = [
            MagicMock(
                values=[
                    MagicMock(float_val=0.5),
                    MagicMock(float_val=1.5),
                    MagicMock(float_val=1.5),
                ]
            )
        ]
        mock_portfolio_details.backtest_results.performance_info = mock_performance_info

        mock_get_full_strategy_info.return_value = mock_portfolio_details

        # Use a valid UUID for the test
        valid_uuid = str(uuid4())
        args = GetPortfolioPerformanceInput(
            portfolio_id=valid_uuid,
            performance_level="monthly",
            date_range=DATE_RANGE_TEST,
            sector_performance_horizon="1M",
        )

        result = await get_portfolio_performance(args, self.context)
        print(result.to_df())
        expected_data = {
            "month": [datetime.date(2024, 1, 1), datetime.date(2024, 2, 1)],
            "return": [100.0, 200.0],
            "return-vs-benchmark": [50.0, 150.0],
        }
        expected_df = pd.DataFrame(expected_data)
        expected_df = expected_df.melt(id_vars=["month"], var_name="field", value_name="value")

        pd.testing.assert_frame_equal(result.to_df(), expected_df)

    @patch("agent_service.tools.portfolio.get_all_workspaces")
    @patch("agent_service.tools.portfolio.get_portfolio_holdings")
    @patch("agent_service.tools.portfolio.get_stocks_sector_performance_for_date_range")
    async def test_get_portfolio_performance_sector(
        self,
        mock_get_stocks_sector_performance_for_date_range: MagicMock,
        mock_get_portfolio_holdings: MagicMock,
        mock_get_all_workspaces: MagicMock,
    ):
        rows = self.create_dummy_workspaces_for_user(self.context.user_id)

        mock_get_all_workspaces.return_value = [
            WorkspaceMetadata(
                workspace_id=UUID(id=row["id"]),
                name=row["name"],
                user_auth_level=row["user_auth_level"],
                last_updated=row["last_updated"],
                created_at=self.to_timestamp(1600000000),
            )
            for row in rows
        ]
        # Mock portfolio holdings
        mock_portfolio_holdings = StockTable.from_df_and_cols(
            data=pd.DataFrame({STOCK_ID_COL_NAME_DEFAULT: [AAPL, ERGB], "Weight": [60.0, 40.0]}),
            columns=[
                TableColumnMetadata(
                    label=STOCK_ID_COL_NAME_DEFAULT, col_type=TableColumnType.STOCK
                ),
                TableColumnMetadata(label="Weight", col_type=TableColumnType.FLOAT),
            ],
        )
        mock_get_portfolio_holdings.return_value = mock_portfolio_holdings

        # Mock sector performance data
        mock_sector_performance_data = [
            MagicMock(
                sector_name="Technology",
                sector_performance=0.05,
                sector_weight=60.0,
                weighted_sector_performance=3.0,
            ),
            MagicMock(
                sector_name="Healthcare",
                sector_performance=0.02,
                sector_weight=40.0,
                weighted_sector_performance=0.8,
            ),
        ]
        mock_get_stocks_sector_performance_for_date_range.return_value = (
            mock_sector_performance_data
        )

        # Use a valid UUID for the test
        valid_uuid = str(uuid4())
        args = GetPortfolioPerformanceInput(
            portfolio_id=valid_uuid,
            performance_level="sector",
            date_range=DATE_RANGE_TEST,
            sector_performance_horizon="1M",
        )

        result = await get_portfolio_performance(args, self.context)
        expected_data = {
            "sector": ["Technology", "Healthcare"],
            "return": [5.0, 2.0],
            "weight": [60.0, 40.0],
            "weighted-return": [3.0, 0.8],
        }
        expected_df = pd.DataFrame(expected_data)

        pd.testing.assert_frame_equal(result.to_df(), expected_df)

    @patch("agent_service.tools.portfolio.get_all_workspaces")
    @patch("agent_service.tools.portfolio.get_portfolio_holdings")
    @patch("agent_service.tools.portfolio.get_stock_performance_for_date_range")
    async def test_get_portfolio_performance_stock(
        self,
        mock_get_portfolio_stock_performance_for_date_range: MagicMock,
        mock_get_portfolio_holdings: MagicMock,
        mock_get_all_workspaces: MagicMock,
    ):
        rows = self.create_dummy_workspaces_for_user(self.context.user_id)

        mock_get_all_workspaces.return_value = [
            WorkspaceMetadata(
                workspace_id=UUID(id=row["id"]),
                name=row["name"],
                user_auth_level=row["user_auth_level"],
                last_updated=row["last_updated"],
                created_at=self.to_timestamp(1600000000),
            )
            for row in rows
        ]
        # Mock portfolio holdings
        mock_portfolio_holdings = StockTable.from_df_and_cols(
            data=pd.DataFrame({STOCK_ID_COL_NAME_DEFAULT: [AAPL, ERGB], "Weight": [60.0, 40.0]}),
            columns=[
                TableColumnMetadata(
                    label=STOCK_ID_COL_NAME_DEFAULT, col_type=TableColumnType.STOCK
                ),
                TableColumnMetadata(label="Weight", col_type=TableColumnType.FLOAT),
            ],
        )
        mock_get_portfolio_holdings.return_value = mock_portfolio_holdings

        # Mock sector performance data
        mock_stock_performance_data = MagicMock(
            stock_performance_list=[
                MagicMock(
                    gbi_id=AAPL.gbi_id,
                    performance=0.05,
                ),
                MagicMock(
                    gbi_id=ERGB.gbi_id,
                    performance=0.02,
                ),
            ]
        )
        mock_get_portfolio_stock_performance_for_date_range.return_value = (
            mock_stock_performance_data
        )

        # Use a valid UUID for the test
        valid_uuid = str(uuid4())
        args = GetPortfolioPerformanceInput(
            portfolio_id=valid_uuid,
            performance_level="stock",
            date_range=DATE_RANGE_TEST,
            sector_performance_horizon="1M",
        )

        result = await get_portfolio_performance(args, self.context)

        # create the expected Table
        expected = StockTable.from_df_and_cols(
            data=pd.DataFrame(
                {
                    STOCK_ID_COL_NAME_DEFAULT: [AAPL, ERGB],
                    "return": [5.0, 2.0],
                    "portfolio-weight": [60.0, 40.0],
                    "weighted-return": [3.0, 0.8],
                }
            ),
            columns=[
                TableColumnMetadata(
                    label=STOCK_ID_COL_NAME_DEFAULT, col_type=TableColumnType.STOCK
                ),
                TableColumnMetadata(label="return", col_type=TableColumnType.FLOAT),
                TableColumnMetadata(label="portfolio-weight", col_type=TableColumnType.FLOAT),
                TableColumnMetadata(label="weighted-return", col_type=TableColumnType.FLOAT),
            ],
        )

        pd.testing.assert_frame_equal(result.to_df(), expected.to_df())

    def to_timestamp(self, seconds):
        timestamp = Timestamp()
        timestamp.seconds = seconds
        return timestamp

    def create_dummy_workspaces_for_user(self, user_id: str):
        rows = [
            {
                "id": str(uuid4()),
                "name": "test portfolio",
                "user_auth_level": WorkspaceAuth.WORKSPACE_AUTH_OWNER,
                "last_updated": self.to_timestamp(1600000000),
                "created_at": self.to_timestamp(1600000000),
            },
            {
                "id": str(uuid4()),
                "name": "my portfolio",
                "user_auth_level": WorkspaceAuth.WORKSPACE_AUTH_UNSPECIFIED,
                "last_updated": self.to_timestamp(1600000000),
                "created_at": self.to_timestamp(1600000000),
            },
            {
                "id": str(uuid4()),
                "name": "best portfolio",
                "user_auth_level": WorkspaceAuth.WORKSPACE_AUTH_UNSPECIFIED,
                "last_updated": self.to_timestamp(1600000000),
                "created_at": self.to_timestamp(1600000000),
            },
        ]
        return rows


if __name__ == "__main__":
    unittest.main()
