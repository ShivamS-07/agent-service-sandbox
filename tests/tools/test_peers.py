import unittest

from agent_service.GPT.requests import set_use_global_stub
from agent_service.io_types.stock import StockID
from agent_service.tools.peers import PeersForStockInput, get_peers
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
        args = PeersForStockInput(stock_ids=[MSFT])
        msft_peers = await get_peers(args, self.context)
        self.assertGreater(len(msft_peers), 1)

        args = PeersForStockInput(stock_ids=[FORD])
        ford_peers = await get_peers(args, self.context)
        self.assertGreater(len(ford_peers), 1)

        args = PeersForStockInput(stock_ids=[FORD, MSFT])
        both_peers = await get_peers(args, self.context)
        self.assertGreater(len(both_peers), len(ford_peers))
        self.assertGreater(len(both_peers), len(msft_peers))
