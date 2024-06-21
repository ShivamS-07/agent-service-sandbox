import datetime
import logging
from unittest import IsolatedAsyncioTestCase

from agent_service.io_types.stock import StockID
from agent_service.tools.custom_documents import (
    GetCustomDocsByTopicInput,
    GetCustomDocsInput,
    get_user_custom_documents,
    get_user_custom_documents_by_topic,
)
from agent_service.types import PlanRunContext

# This user is set up in a test team with access to the custom documents tool
# and a handful of sample documents.
CUSTOM_DOC_DEV_TEST_USER = "515b61f7-38af-4826-ad32-0900b3b1b7d4"
COST_STOCK_ID = StockID(gbi_id=2844, isin="", symbol="COST", company_name="")


class TestCustomDocuments(IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        logging.getLogger("asyncio").setLevel(logging.ERROR)

    async def asyncSetUp(self):
        self.context = PlanRunContext.get_dummy(
            user_id=CUSTOM_DOC_DEV_TEST_USER,
        )

    async def test_get_custom_doc_by_stock(self):
        # these are well covered in the service itself, so just test for
        # integration here.
        input = GetCustomDocsInput(
            stock_ids=[COST_STOCK_ID],
            limit=5,
            # end_date=datetime.date.today(),
        )
        all_data = await get_user_custom_documents(input, self.context)
        self.assertGreater(len(all_data), 0)

    async def test_get_custom_doc_by_topic(self):
        # these are well covered in the service itself, so just test for
        # integration here.
        input = GetCustomDocsByTopicInput(
            topic="major decrease in earnings",
            limit=5,
            start_date=datetime.date(2020, 1, 1),
        )
        all_data = await get_user_custom_documents_by_topic(input, self.context)
        self.assertGreater(len(all_data), 0)
