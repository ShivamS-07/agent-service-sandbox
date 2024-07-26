import logging
import unittest
from unittest import IsolatedAsyncioTestCase

from agent_service.io_types.stock import StockID
from agent_service.io_types.text import TextOutput
from agent_service.tools.custom_documents import (
    GetCustomDocsInput,
    get_user_custom_documents,
)
from agent_service.tools.hypothesis import (
    SummarizeCustomDocumentHypothesisInput,
    SummarizeHypothesisFromVariousSourcesInput,
    TestAndSummarizeCustomDocsHypothesisInput,
    TestCustomDocsHypothesisInput,
    summarize_hypothesis_from_custom_documents,
    summarize_hypothesis_from_various_sources,
    test_and_summarize_hypothesis_with_custom_documents,
    test_hypothesis_for_custom_documents,
)
from agent_service.types import PlanRunContext
from agent_service.utils.output_utils.output_construction import get_output_from_io_type
from agent_service.utils.postgres import get_psql

# This user is set up in a test team with access to the custom documents tool
# and a handful of sample documents.
CUSTOM_DOC_DEV_TEST_USER = "515b61f7-38af-4826-ad32-0900b3b1b7d4"
STZ_STOCK_ID = StockID(gbi_id=2202, isin="", symbol="STZ", company_name="")


@unittest.skip("The tool is disabled")
class TestHypothesisPipeline(IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        logging.getLogger("asyncio").setLevel(logging.ERROR)

    async def asyncSetUp(self):
        self.context = PlanRunContext.get_dummy(
            user_id=CUSTOM_DOC_DEV_TEST_USER,
        )

    async def test_get_custom_doc_by_stock(self):
        get_docs_input = GetCustomDocsInput(
            stock_ids=[STZ_STOCK_ID],
            limit=5,
        )
        all_docs = await get_user_custom_documents(get_docs_input, self.context)
        self.assertGreater(len(all_docs), 0)

        test_input = TestCustomDocsHypothesisInput(
            hypothesis="expected earnings growth",
            custom_document_list=all_docs,
        )
        hypothesis_text = await test_hypothesis_for_custom_documents(test_input, self.context)

        summarize_input = SummarizeCustomDocumentHypothesisInput(
            hypothesis="expected earnings growth",
            custom_documents=hypothesis_text,
        )
        summarize_output = await summarize_hypothesis_from_custom_documents(
            summarize_input, self.context
        )
        result_output = await get_output_from_io_type(summarize_output, pg=get_psql())
        self.assertIsInstance(result_output, TextOutput)
        self.assertGreater(len(result_output.val), 0)

    async def test_summarize_all(self):
        """
        Right now I've only added the custom doc branch to summarize-all.
        """

        get_docs_input = GetCustomDocsInput(
            stock_ids=[STZ_STOCK_ID],
            limit=5,
        )
        all_docs = await get_user_custom_documents(get_docs_input, self.context)
        self.assertGreater(len(all_docs), 0)

        custom_doc_summary = await test_and_summarize_hypothesis_with_custom_documents(
            TestAndSummarizeCustomDocsHypothesisInput(
                hypothesis="expected earnings growth", custom_documents=all_docs
            ),
            self.context,
        )

        summarize_output = await summarize_hypothesis_from_various_sources(
            SummarizeHypothesisFromVariousSourcesInput(
                hypothesis="expected earnings growth", hypothesis_summaries=[custom_doc_summary]
            ),
            self.context,
        )
        result_output = await get_output_from_io_type(summarize_output, pg=get_psql())
        self.assertIsInstance(result_output, TextOutput)
        self.assertGreater(len(result_output.val), 0)
