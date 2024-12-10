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

from agent_service.io_types.stock import StockID
from agent_service.tools.watchlist import (
    GetUserWatchlistStocksInput,
    get_user_watchlist_stocks,
    watchlist_match_by_gpt,
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
            return [714, 6963]

        mock_get_watchlist_stocks.side_effect = get_watchlist_stocks

        args = GetUserWatchlistStocksInput(watchlist_name="Helo")
        result = await get_user_watchlist_stocks(args, self.context)
        self.assertEqual(
            sorted(result),
            sorted(
                [
                    StockID(gbi_id=714, symbol="AAPL", isin="US0378331005", company_name=""),
                    StockID(gbi_id=6963, symbol="MSFT", isin="US5949181045", company_name=""),
                ]
            ),
        )

    @unittest.skip("flaky")
    async def test_watchlist_match_by_gpt(self):
        # Test 1 - Validate it matches to SOME watchlist
        watchlist_name = "Luxury & Super Luxury Watchlist"
        watchlists_name_to_id = {
            "LUX": "some_id",
            "Global Portfolio": "some_id",
            "Small Cap Portfolio": "some_id",
            "Random": "some_id",
            "Default Portfolio": "some_id",
        }

        result_name, _ = await watchlist_match_by_gpt(
            context=self.context,
            watchlist_name=watchlist_name,
            watchlists_name_to_id=watchlists_name_to_id,
        )

        self.assertIn(result_name, [name for name in watchlists_name_to_id.keys()])

        # Test 2 - Validate it matches to SOME watchlist
        watchlist_name = "GCS"
        watchlists_name_to_id = {
            "GSC": "some_id",
            "Global Portfolio": "some_id",
            "Small Cap Portfolio": "some_id",
            "Random": "some_id",
            "Default Portfolio": "some_id",
        }

        result_name, _ = await watchlist_match_by_gpt(
            context=self.context,
            watchlist_name=watchlist_name,
            watchlists_name_to_id=watchlists_name_to_id,
        )

        self.assertIn(result_name, [name for name in watchlists_name_to_id.keys()])

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
