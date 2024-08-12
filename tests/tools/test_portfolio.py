import datetime
import logging
import unittest
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pandas as pd
from google.protobuf.timestamp_pb2 import Timestamp
from pa_portfolio_service_proto_v1.well_known_types_pb2 import UUID
from pa_portfolio_service_proto_v1.workspace_pb2 import WorkspaceAuth, WorkspaceMetadata

from agent_service.external.dal_svc_client import (
    DALServiceClient,
    PreviousTradesMetadata,
)
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
    GetPortfolioTradesInput,
    convert_portfolio_mention_to_portfolio_id,
    get_portfolio_holdings,
    get_portfolio_performance,
    get_portfolio_trades,
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
    @patch("agent_service.tools.portfolio.get_return_for_stocks")
    async def test_get_portfolio_holdings(
        self,
        mock_get_return_for_stocks: MagicMock,
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

        # Mock get_return_for_stocks
        mock_get_return_for_stocks.return_value = {AAPL.gbi_id: 0.23, ERGB.gbi_id: 0.04}

        args = GetPortfolioHoldingsInput(portfolio_id=valid_uuid)
        result = await get_portfolio_holdings(args, self.context)

        expected_portfolio_holdings = StockTable.from_df_and_cols(
            data=pd.DataFrame(
                {
                    STOCK_ID_COL_NAME_DEFAULT: [AAPL, ERGB],
                    "Weight": [0.6, 0.4],
                    "Price": [215, 123],
                    "Return": [0.23, 0.04],
                }
            ),
            columns=[
                TableColumnMetadata(
                    label=STOCK_ID_COL_NAME_DEFAULT, col_type=TableColumnType.STOCK
                ),
                TableColumnMetadata(label="Weight", col_type=TableColumnType.PERCENT),
                TableColumnMetadata(label="Price", col_type=TableColumnType.FLOAT),
                TableColumnMetadata(label="Return", col_type=TableColumnType.PERCENT),
            ],
        )
        pd.testing.assert_frame_equal(result.to_df(), expected_portfolio_holdings.to_df())

    @patch("agent_service.tools.portfolio.get_all_workspaces")
    @patch("agent_service.tools.portfolio.get_portfolio_holdings")
    @patch("agent_service.tools.portfolio.get_return_for_stocks")
    async def test_get_portfolio_performance_sector(
        self,
        mock_get_return_for_stocks: MagicMock,
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
            data=pd.DataFrame({STOCK_ID_COL_NAME_DEFAULT: [AAPL, ERGB], "Weight": [0.6, 0.4]}),
            columns=[
                TableColumnMetadata(
                    label=STOCK_ID_COL_NAME_DEFAULT, col_type=TableColumnType.STOCK
                ),
                TableColumnMetadata(label="Weight", col_type=TableColumnType.PERCENT),
            ],
        )
        mock_get_portfolio_holdings.return_value = mock_portfolio_holdings

        # Mock stock performance data
        mock_get_return_for_stocks.return_value = {AAPL.gbi_id: 0.1, ERGB.gbi_id: 0.05}

        # Use a valid UUID for the test
        valid_uuid = str(uuid4())
        args = GetPortfolioPerformanceInput(
            portfolio_id=valid_uuid,
            performance_level="sector",
            date_range=DATE_RANGE_TEST,
        )

        result = await get_portfolio_performance(args, self.context)
        print("result sector: ", result.to_df())
        # create the expected Table
        expected_data = {
            "sector": ["Information Technology", "Industrials"],
            "weight": [0.6, 0.4],
            "weighted-return": [0.06, 0.02],
        }
        expected_df = pd.DataFrame(expected_data)
        expected_table = StockTable.from_df_and_cols(
            data=expected_df,
            columns=[
                TableColumnMetadata(label="sector", col_type=TableColumnType.STRING),
                TableColumnMetadata(label="weight", col_type=TableColumnType.PERCENT),
                TableColumnMetadata(label="weighted-return", col_type=TableColumnType.PERCENT),
            ],
        )

        pd.testing.assert_frame_equal(result.to_df(), expected_table.to_df())

    @patch("agent_service.tools.portfolio.get_all_workspaces")
    @patch("agent_service.tools.portfolio.get_portfolio_holdings")
    @patch("agent_service.tools.portfolio.get_return_for_stocks")
    async def test_get_portfolio_performance_stock(
        self,
        mock_get_return_for_stocks: MagicMock,
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
            data=pd.DataFrame({STOCK_ID_COL_NAME_DEFAULT: [AAPL, ERGB], "Weight": [0.6, 0.4]}),
            columns=[
                TableColumnMetadata(
                    label=STOCK_ID_COL_NAME_DEFAULT, col_type=TableColumnType.STOCK
                ),
                TableColumnMetadata(label="Weight", col_type=TableColumnType.PERCENT),
            ],
        )
        mock_get_portfolio_holdings.return_value = mock_portfolio_holdings

        # Mock get_return_for_stocks
        mock_get_return_for_stocks.return_value = {AAPL.gbi_id: 0.1, ERGB.gbi_id: 0.05}

        # Use a valid UUID for the test
        valid_uuid = str(uuid4())
        args = GetPortfolioPerformanceInput(
            portfolio_id=valid_uuid,
            performance_level="stock",
            date_range=DATE_RANGE_TEST,
        )

        result = await get_portfolio_performance(args, self.context)
        print("result stock: ", result.to_df())
        # create the expected Table
        expected = StockTable.from_df_and_cols(
            data=pd.DataFrame(
                {
                    STOCK_ID_COL_NAME_DEFAULT: [AAPL, ERGB],
                    "return": [0.1, 0.05],
                    "portfolio-weight": [0.6, 0.4],
                    "weighted-return": [0.06, 0.02],
                }
            ),
            columns=[
                TableColumnMetadata(
                    label=STOCK_ID_COL_NAME_DEFAULT, col_type=TableColumnType.STOCK
                ),
                TableColumnMetadata(label="return", col_type=TableColumnType.PERCENT),
                TableColumnMetadata(label="portfolio-weight", col_type=TableColumnType.PERCENT),
                TableColumnMetadata(label="weighted-return", col_type=TableColumnType.PERCENT),
            ],
        )

        pd.testing.assert_frame_equal(result.to_df(), expected.to_df())

    @patch("agent_service.tools.portfolio.get_all_workspaces")
    @patch.object(
        DALServiceClient,
        "fetch_previous_trades",
        AsyncMock(
            return_value=[
                PreviousTradesMetadata(
                    gbi_id=AAPL.gbi_id,
                    trade_date=datetime.date(2024, 7, 31),
                    action="BUY",
                    allocation_change=0.15,
                ),
                PreviousTradesMetadata(
                    gbi_id=ERGB.gbi_id,
                    trade_date=datetime.date(2024, 7, 20),
                    action="SELL",
                    allocation_change=-0.05,
                ),
            ]
        ),
    )
    async def test_get_portfolio_trades(
        self,
        mock_get_all_workspaces: MagicMock,
    ):
        row = self.create_dummy_workspaces_for_user(self.context.user_id)[0]

        mock_get_all_workspaces.return_value = [
            WorkspaceMetadata(
                workspace_id=UUID(id=row["id"]),
                name=row["name"],
                user_auth_level=row["user_auth_level"],
                last_updated=row["last_updated"],
                created_at=self.to_timestamp(1600000000),
            )
        ]

        args = GetPortfolioTradesInput(
            portfolio_id=row["id"],
        )

        result = await get_portfolio_trades(args, self.context)

        # Expected data
        expected_data = {
            STOCK_ID_COL_NAME_DEFAULT: [AAPL, ERGB],
            "Date": [datetime.date(2024, 7, 31), datetime.date(2024, 7, 20)],
            "Action": ["BUY", "SELL"],
            "Allocation Change": [0.15, -0.05],
        }
        expected_df = pd.DataFrame(expected_data)
        expected_table = StockTable.from_df_and_cols(
            data=expected_df,
            columns=[
                TableColumnMetadata(
                    label=STOCK_ID_COL_NAME_DEFAULT, col_type=TableColumnType.STOCK
                ),
                TableColumnMetadata(label="Date", col_type=TableColumnType.DATE),
                TableColumnMetadata(label="Action", col_type=TableColumnType.STRING),
                TableColumnMetadata(label="Allocation Change", col_type=TableColumnType.PERCENT),
            ],
        )

        # Assert the result
        pd.testing.assert_frame_equal(result.to_df(), expected_table.to_df())

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
