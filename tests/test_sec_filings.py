import datetime
import logging
import unittest

from agent_service.utils.sec.constants import FILE_10K, FILE_10Q
from agent_service.utils.sec.sec_api import SecFiling

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s.%(funcName)s:%(lineno)d - %(levelname)s - %(message)s",
    force=True,
)


class TestSecFilings(unittest.IsolatedAsyncioTestCase):
    async def test_sec_filings(self):
        filing_gbi_pairs, filing_to_db_id = await SecFiling.get_filings(
            gbi_ids=[714, 6963, 26794],
            form_types=[FILE_10K, FILE_10Q],
            start_date=datetime.datetime(1900, 1, 1),
            end_date=datetime.datetime(2025, 1, 1),
        )
        print(len(filing_gbi_pairs))
        print(len(filing_to_db_id))
