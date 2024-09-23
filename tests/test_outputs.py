import unittest
from typing import List

from parameterized import param, parameterized

from agent_service.io_type_utils import HistoryEntry
from agent_service.io_types.stock import StockID
from agent_service.io_types.text import StockText, Text, TextCitation
from agent_service.utils.output_utils.output_construction import (
    combine_text_list,
    prepare_list_of_stock_texts,
)


class TestOutputConstruction(unittest.IsolatedAsyncioTestCase):
    @parameterized.expand(
        [
            # TEST CASE 1
            param(
                texts=[
                    Text(
                        val="Hello this is a test text.",
                        history=[
                            HistoryEntry(
                                citations=[
                                    TextCitation(source_text=Text(), citation_text_offset=4),
                                    TextCitation(source_text=Text(), citation_text_offset=9),
                                ]
                            ),
                            HistoryEntry(
                                citations=[
                                    TextCitation(source_text=Text(), citation_text_offset=12),
                                ]
                            ),
                        ],
                    ),
                    Text(
                        val="This is another testing text.",
                        history=[
                            HistoryEntry(
                                citations=[
                                    TextCitation(source_text=Text(), citation_text_offset=3),
                                    TextCitation(source_text=Text(), citation_text_offset=6),
                                ]
                            )
                        ],
                    ),
                ],
                per_line_prefix="- ",
                per_line_suffix="\n",
                overall_prefix="",
                overall_suffix="",
                expected_result=Text(
                    val="- Hello this is a test text.\n- This is another testing text.\n",
                    history=[
                        HistoryEntry(
                            citations=[
                                TextCitation(source_text=Text(), citation_text_offset=6),
                                TextCitation(source_text=Text(), citation_text_offset=11),
                                TextCitation(source_text=Text(), citation_text_offset=14),
                                TextCitation(source_text=Text(), citation_text_offset=34),
                                TextCitation(source_text=Text(), citation_text_offset=37),
                            ]
                        )
                    ],
                ),
            ),
            # TEST CASE 2
            param(
                texts=[
                    Text(
                        val="Hello this is a test text.",
                        history=[
                            HistoryEntry(
                                citations=[
                                    TextCitation(source_text=Text(), citation_text_offset=4),
                                    TextCitation(source_text=Text(), citation_text_offset=9),
                                ]
                            ),
                            HistoryEntry(
                                citations=[
                                    TextCitation(source_text=Text(), citation_text_offset=12),
                                ]
                            ),
                        ],
                    ),
                    Text(
                        val="This is another testing text.",
                        history=[
                            HistoryEntry(
                                citations=[
                                    TextCitation(source_text=Text(), citation_text_offset=3),
                                    TextCitation(source_text=Text(), citation_text_offset=6),
                                ]
                            )
                        ],
                    ),
                ],
                per_line_prefix="- ",
                per_line_suffix="\n",
                overall_prefix="Something something stocks\n",
                overall_suffix="\n\n\n",
                expected_result=Text(
                    val=(
                        "Something something stocks\n"
                        "- Hello this is a test text.\n"
                        "- This is another testing text.\n\n\n\n"
                    ),
                    history=[
                        HistoryEntry(
                            citations=[
                                TextCitation(
                                    source_text=Text(),
                                    citation_text_offset=6 + len("Something something stocks\n"),
                                ),
                                TextCitation(
                                    source_text=Text(),
                                    citation_text_offset=11 + len("Something something stocks\n"),
                                ),
                                TextCitation(
                                    source_text=Text(),
                                    citation_text_offset=14 + len("Something something stocks\n"),
                                ),
                                TextCitation(
                                    source_text=Text(),
                                    citation_text_offset=34 + len("Something something stocks\n"),
                                ),
                                TextCitation(
                                    source_text=Text(),
                                    citation_text_offset=37 + len("Something something stocks\n"),
                                ),
                            ]
                        )
                    ],
                ),
            ),
        ]
    )
    async def test__combine_text_list(
        self,
        texts: List[Text],
        per_line_prefix: str,
        per_line_suffix: str,
        overall_prefix: str,
        overall_suffix: str,
        expected_result: Text,
    ):
        output = combine_text_list(
            texts=texts,
            per_line_prefix=per_line_prefix,
            per_line_suffix=per_line_suffix,
            overall_prefix=overall_prefix,
            overall_suffix=overall_suffix,
        )
        self.assertEqual(output.val, expected_result.val)
        self.assertEqual(output.history, expected_result.history)
        for output_cit, expected_cit in zip(
            output.get_all_citations(), expected_result.get_all_citations()
        ):
            self.assertEqual(output_cit.citation_text_offset, expected_cit.citation_text_offset)  # type: ignore

    async def test_prepare_list_of_stock_texts(self):
        aapl = StockID(gbi_id=714, symbol="AAPL", isin="")
        googl = StockID(gbi_id=10096, symbol="GOOGL", isin="")
        output = await prepare_list_of_stock_texts(
            texts=[
                StockText(
                    stock_id=aapl,
                    val="This is a test",
                    history=[
                        HistoryEntry(
                            citations=[
                                TextCitation(source_text=Text(), citation_text_offset=3),
                                TextCitation(source_text=Text(), citation_text_offset=6),
                            ]
                        ),
                        HistoryEntry(
                            citations=[
                                TextCitation(source_text=Text(), citation_text_offset=8),
                            ]
                        ),
                    ],
                ),
                StockText(
                    stock_id=aapl,
                    val="Another test",
                ),
                StockText(
                    stock_id=googl,
                    val="One more",
                    history=[
                        HistoryEntry(
                            citations=[
                                TextCitation(source_text=Text(), citation_text_offset=2),
                                TextCitation(source_text=Text(), citation_text_offset=7),
                            ]
                        ),
                    ],
                ),
            ]
        )

        # 5 above, 1 auto inserted for text 2 because it has no citations
        self.assertEqual(len(output.get_all_citations()), 6)
