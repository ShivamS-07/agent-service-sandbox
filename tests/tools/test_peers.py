import unittest

from agent_service.GPT.requests import set_use_global_stub
from agent_service.io_types.stock import StockID
from agent_service.tools.peers import GeneralPeersForStockInput, get_general_peers
from agent_service.types import PlanRunContext

MSFT = StockID(gbi_id=6963, isin="", symbol="MSFT", company_name="Microsoft")
FORD = StockID(gbi_id=4579, isin="", symbol="F", company_name="Ford")


class TestPeers(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        set_use_global_stub(False)
        self.context = PlanRunContext.get_dummy()

        # uncomment for easier debugging
        from agent_service.utils.logs import init_test_logging

        init_test_logging()

    async def test_get_peers(self):
        args = GeneralPeersForStockInput(stock_id=MSFT)
        msft_peers = await get_general_peers(args, self.context)
        msft_peers_gbi_ids = [stock_id.gbi_id for stock_id in msft_peers]
        self.assertGreater(len(msft_peers_gbi_ids), 1)
        self.assertTrue(714 in msft_peers_gbi_ids)  # Apple
        self.assertTrue(10096 in msft_peers_gbi_ids)  # Google
        self.assertTrue(149 in msft_peers_gbi_ids)  # Amazon

        args = GeneralPeersForStockInput(stock_id=FORD)
        ford_peers = await get_general_peers(args, self.context)
        ford_peers_gbi_ids = [stock_id.gbi_id for stock_id in ford_peers]
        self.assertGreater(len(ford_peers_gbi_ids), 1)
        self.assertTrue(25508 in ford_peers_gbi_ids)  # Tesla
        self.assertTrue(25477 in ford_peers_gbi_ids)  # General Motors
        self.assertTrue(389721 in ford_peers_gbi_ids)  # Toyota
