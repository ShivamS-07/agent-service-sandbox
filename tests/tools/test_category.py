import logging
from typing import List
from unittest import IsolatedAsyncioTestCase

from agent_service.io_types.stock import StockID
from agent_service.io_types.text import TextOutput
from agent_service.tools.category import (
    Category,
    CriteriaForCompetitiveAnalysis,
    get_criteria_for_competitive_analysis,
)
from agent_service.types import PlanRunContext
from agent_service.utils.postgres import SyncBoostedPG


class TestHypothesisPipeline(IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        logging.getLogger("asyncio").setLevel(logging.ERROR)

    async def asyncSetUp(self):
        self.context = PlanRunContext.get_dummy()
        self.category = Category(
            name="Test Category",
            explanation="This is a test category",
            justification="This is a test justification",
            weight=0.5,
        )

    async def test_category_to_markdown_string(self):
        category_str = self.category.to_markdown_string()

        expected_str = (
            "**Category: Test Category**\n"
            " - Explanation: This is a test category\n"
            " - Justification: This is a test justification\n"
            " - Weight: 0.5\n"
        )

        self.assertEqual(category_str, expected_str)

    async def test_category_to_rich_output(self):
        category_output = await self.category.to_rich_output(pg=SyncBoostedPG(skip_commit=True))
        self.assertIsInstance(category_output, TextOutput)

    async def test_category_to_gpt_input(self):
        gpt_input = await self.category.to_gpt_input()
        expected_input = (
            "- Criteria Name: Test Category. Explanation: This is a test category. "
            "Justification: This is a test justification."
        )
        self.assertEqual(gpt_input, expected_input)

    async def test_category_hash(self):
        h1 = hash(self.category)  # `hash` is random, so we can't predict the output
        h2 = hash(
            (
                self.category.name,
                self.category.explanation,
                self.category.justification,
                self.category.weight,
            )
        )
        self.assertEqual(h1, h2)

    async def test_category_multi_to_gpt_input(self):
        c1 = Category(
            name="Test Category 1",
            explanation="This is a test category 1",
            justification="This is a test justification 1",
            weight=0.5,
        )
        c2 = Category(
            name="Test Category 2",
            explanation="This is a test category 2",
            justification="This is a test justification 2",
            weight=0.5,
        )

        gpt_input = Category.multi_to_gpt_input([c1, c2])
        expected_input = (
            "- 0: Test Category 1\n"
            "Explanation: This is a test category 1\n"
            "Justification: This is a test justification 1\n"
            "Weight: 0.5\n"
            "- 1: Test Category 2\n"
            "Explanation: This is a test category 2\n"
            "Justification: This is a test justification 2\n"
            "Weight: 0.5"
        )
        self.assertEqual(gpt_input, expected_input)

    async def test_get_criteria_for_competitive_analysis(self):
        args = CriteriaForCompetitiveAnalysis(
            market="AI Chips",
            must_include_criteria=["Market Share"],
            target_stock=StockID(gbi_id=7555, symbol="NVDA", company_name="NVIDIA Corporation"),
            limit=5,
        )
        categories: List[Category] = await get_criteria_for_competitive_analysis(args, self.context)
        self.assertGreater(len(categories), 0)
        self.assertIn("Market Share", [c.name for c in categories])
