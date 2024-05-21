import logging
import unittest
from unittest import IsolatedAsyncioTestCase
from unittest.mock import MagicMock, patch
from uuid import uuid4

from pa_portfolio_service_proto_v1.watchlist_pb2 import (
    GetAllWatchlistsResponse,
    WatchlistMetadata,
)
from pa_portfolio_service_proto_v1.well_known_types_pb2 import UUID

from agent_service.tools.watchlist import (
    GetUserWatchlistStocksInput,
    get_user_watchlist_stocks,
)
from agent_service.types import PlanRunContext
from agent_service.utils.postgres import get_psql


class TestWatchlistTools(IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        logging.getLogger("asyncio").setLevel(logging.ERROR)

    async def asyncSetUp(self):
        self.context = PlanRunContext.get_dummy()

    @patch("agent_service.tools.watchlist.get_watchlist_stocks")
    @patch("agent_service.tools.watchlist.get_all_watchlists")
    async def test_get_user_watchlist_stocks(
        self, mock_get_all_watchlists: MagicMock, mock_get_watchlist_stocks: MagicMock
    ):
        rows = self.create_dummy_watchlists_for_user(self.context.user_id)

        # we must mock the grpc response because the data doesn't exist in DB
        async def get_all_watchlists(user_id: str):
            return GetAllWatchlistsResponse(
                watchlists=[
                    WatchlistMetadata(watchlist_id=UUID(id=row["id"]), name=row["name"])
                    for row in rows
                ]
            )

        mock_get_all_watchlists.side_effect = get_all_watchlists

        async def get_watchlist_stocks(user_id: str, watchlist_id: str):
            return [1, 2, 3]

        mock_get_watchlist_stocks.side_effect = get_watchlist_stocks

        args = GetUserWatchlistStocksInput(watchlist_name="Helo")
        result = await get_user_watchlist_stocks(args, self.context)
        self.assertEqual(result, [1, 2, 3])

    def create_dummy_watchlists_for_user(self, user_id: str):
        rows = [
            {"id": str(uuid4()), "name": "Hello", "owner_user_id": user_id},
            {"id": str(uuid4()), "name": "Hey", "owner_user_id": user_id},
            {"id": str(uuid4()), "name": "Hi", "owner_user_id": user_id},
        ]
        pg = get_psql(skip_commit=True)
        pg.multi_row_insert("public.watchlist", rows)
        return rows


if __name__ == "__main__":
    unittest.main()
