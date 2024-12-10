import unittest
from uuid import uuid4

from agent_service.io_types.text import TestText, Text, TextGroup
from agent_service.tools.LLM_analysis.utils import extract_citations_from_gpt_output
from agent_service.types import PlanRunContext

GPT_OUTPUT = "This is the first sentence.[aa,1] This is the second sentence.[ab,2]\n{'aa':[{'num':0}], 'ab':[{'num':1, 'snippet':'exact match'}, {'num':1, 'snippet':'I made 1.3B investing with Boosted'}]}"  # noqa: E501

TEXT1 = "Doesn't matter what this text is, no snippet matching"
TEXT2 = "Need to make an exact match here. I scored $1.3B investing with Boosted.ai. $1.3B valuation for Boosted is not impossible"  # noqa: E501


class TestCitations(unittest.IsolatedAsyncioTestCase):
    async def test_load_citations(self):
        context = PlanRunContext(
            agent_id=str(uuid4()),
            plan_id=str(uuid4()),
            user_id=str(uuid4()),
            plan_run_id=str(uuid4()),
        )

        text1 = TestText(val=TEXT1)
        text2 = TestText(val=TEXT2)

        text_group = TextGroup(val=(text1, text2))
        _ = await Text.get_all_strs(text_group)

        text, citations = await extract_citations_from_gpt_output(GPT_OUTPUT, text_group, context)

        self.assertEqual(
            text, GPT_OUTPUT.split("\n")[0].replace("[aa,1]", "").replace("[ab,2]", "")
        )

        self.assertEqual(len(citations), 3)

        self.assertEqual(citations[0].source_text, text1)
        self.assertEqual(citations[0].citation_text_offset, GPT_OUTPUT.find("[aa") - 1)
        self.assertIsNone(citations[0].citation_snippet)

        self.assertEqual(citations[1].source_text, text2)
        self.assertEqual(citations[1].citation_text_offset, GPT_OUTPUT.find("[ab") - 7)
        self.assertEqual(citations[1].citation_snippet, "exact match")
        self.assertIn(citations[1].citation_snippet, citations[1].citation_snippet_context)
        self.assertGreater(
            len(citations[1].citation_snippet_context), len(citations[1].citation_snippet)
        )
        self.assertIn(citations[1].citation_snippet_context, text2.val)

        self.assertEqual(citations[2].source_text, text2)
        self.assertEqual(citations[2].citation_text_offset, GPT_OUTPUT.find("[ab") - 7)
        self.assertIn("investing", citations[2].citation_snippet)
        self.assertIn(citations[2].citation_snippet, citations[2].citation_snippet_context)
        self.assertGreater(
            len(citations[2].citation_snippet_context), len(citations[2].citation_snippet)
        )
        self.assertIn(citations[2].citation_snippet_context, text2.val)
