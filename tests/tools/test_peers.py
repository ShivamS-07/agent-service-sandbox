import unittest

from agent_service.io_types.stock import StockID
from agent_service.tools.peers import GeneralPeersForStockInput, get_general_peers
from agent_service.types import PlanRunContext

MSFT = StockID(gbi_id=6963, isin="", symbol="MSFT", company_name="Microsoft")
FORD = StockID(gbi_id=4579, isin="", symbol="F", company_name="Ford")


@unittest.skip("The tool is disabled")
class TestPeers(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.context = PlanRunContext.get_dummy()

        # uncomment for easier debugging
        from agent_service.utils.logs import init_test_logging

        init_test_logging()

    async def test_get_peers(self):
        args = GeneralPeersForStockInput(stock_id=MSFT)
        msft_peers = await get_general_peers(args, self.context)
        self.assertGreater(len(msft_peers), 1)

        args = GeneralPeersForStockInput(stock_id=MSFT, category="Operating Systems")
        msft_os_peers = await get_general_peers(args, self.context)
        self.assertGreater(len(msft_os_peers), 1)
        self.assertGreater(len(msft_peers), len(msft_os_peers))

        args = GeneralPeersForStockInput(stock_id=FORD)
        ford_peers = await get_general_peers(args, self.context)
        self.assertGreater(len(ford_peers), 1)
