import logging
import unittest
from unittest import IsolatedAsyncioTestCase
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pandas as pd
from google.protobuf.timestamp_pb2 import Timestamp
from pa_portfolio_service_proto_v1.well_known_types_pb2 import UUID
from pa_portfolio_service_proto_v1.workspace_pb2 import WorkspaceAuth, WorkspaceMetadata

from agent_service.tools.portfolio import (
    GetPortfolioInput,
    GetPortfolioPerformanceInput,
    convert_portfolio_mention_to_portfolio_id,
    get_portfolio_performance,
)
from agent_service.types import PlanRunContext


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

    @patch("agent_service.tools.portfolio.get_full_strategy_info")
    @patch("agent_service.tools.portfolio.get_psql")
    async def test_get_portfolio_performance(
        self, mock_get_psql: MagicMock, mock_get_full_strategy_info: MagicMock
    ):
        # Create a mock database and mock psql
        mock_db = MagicMock()
        mock_db.get_workspace_linked_id.return_value = str(uuid4())
        mock_get_psql.return_value = mock_db

        # Mock portfolio details response
        mock_portfolio_details = MagicMock()
        mock_performance_info = MagicMock()
        mock_performance_info.monthly_gains.headers = ["2023-01", "2023-02"]
        mock_performance_info.monthly_gains.row_values = [
            MagicMock(values=[MagicMock(float_val=1.0), MagicMock(float_val=2.0)])
        ]
        mock_performance_info.monthly_gains_v_benchmark.row_values = [
            MagicMock(values=[MagicMock(float_val=0.5), MagicMock(float_val=1.5)])
        ]
        mock_portfolio_details.backtest_results.performance_info = mock_performance_info

        mock_get_full_strategy_info.return_value = mock_portfolio_details

        # Use a valid UUID for the test
        valid_uuid = str(uuid4())
        args = GetPortfolioPerformanceInput(portfolio_id=valid_uuid)

        result = await get_portfolio_performance(args, self.context)
        expected_data = {
            "month": ["2023-01", "2023-02"],
            "return": [1.0, 2.0],
            "return-vs-benchmark": [0.5, 1.5],
        }
        expected_df = pd.DataFrame(expected_data)

        pd.testing.assert_frame_equal(result.to_df(), expected_df)

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
