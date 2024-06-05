import logging
import unittest
from unittest import IsolatedAsyncioTestCase
from unittest.mock import MagicMock, patch
from uuid import uuid4

from google.protobuf.timestamp_pb2 import Timestamp
from pa_portfolio_service_proto_v1.well_known_types_pb2 import UUID
from pa_portfolio_service_proto_v1.workspace_pb2 import WorkspaceAuth, WorkspaceMetadata

from agent_service.tools.portfolio import (
    GetPortfolioInput,
    convert_portfolio_mention_to_portfolio_id,
)
from agent_service.types import PlanRunContext


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

        async def get_all_workspaces(user_id: str):
            workspaces = [
                WorkspaceMetadata(
                    workspace_id=UUID(id=row["id"]),
                    name=row["name"],
                    user_auth_level=row["user_auth_level"],
                    last_updated=row["last_updated"],
                    created_at=self.to_timestamp(1600000000),
                )
                for row in rows
            ]
            return workspaces

        mock_get_all_workspaces.side_effect = get_all_workspaces

        args = GetPortfolioInput(portfolio_name="Portfolio")
        result = await convert_portfolio_mention_to_portfolio_id(args, self.context)
        self.assertEqual(result, rows[0]["id"])

        args = GetPortfolioInput(portfolio_name="my Portfolio")
        result = await convert_portfolio_mention_to_portfolio_id(args, self.context)
        self.assertEqual(result, rows[1]["id"])

        args = GetPortfolioInput(portfolio_name="NonExistant Portfolio")
        result = await convert_portfolio_mention_to_portfolio_id(args, self.context)
        self.assertEqual(result, rows[0]["id"])

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
