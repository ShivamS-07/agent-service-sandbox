import unittest

from agent_service.utils.stock_metadata import get_stock_metadata


class TestTextObjects(unittest.IsolatedAsyncioTestCase):
    async def test_stock_metadata(self):
        mds = await get_stock_metadata(gbi_ids=[714])
        self.assertEqual(1, len(mds))
