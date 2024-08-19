# flake8: noqa
import unittest
from typing import List

from parameterized import param, parameterized

from agent_service.io_types.stock import StockID
from agent_service.io_types.text_objects import (
    CitationTextObject,
    StockTextObject,
    TextObject,
)


class TestTextObjects(unittest.IsolatedAsyncioTestCase):
    @parameterized.expand(
        [
            param(text="", text_objects=[], expected_result=""),
            param(text="testing!", text_objects=[], expected_result="testing!"),
            param(
                text="This is some test text\nHello!",
                text_objects=[CitationTextObject(citation_id="cit1", index=3)],
                expected_result='This ```{"type": "citation", "citation_id": "cit1"}```  is some test text\nHello!',
            ),
            # Two citations in the same place
            param(
                text="This is some test text\nHello!",
                text_objects=[
                    CitationTextObject(citation_id="cit1", index=3),
                    CitationTextObject(citation_id="cit2", index=3),
                ],
                expected_result='This ```{"type": "citation", "citation_id": "cit1"}```  ```{"type": "citation", "citation_id": "cit2"}```  is some test text\nHello!',
            ),
            param(
                text="This is some test text\nHello!",
                text_objects=[
                    CitationTextObject(citation_id="cit1", index=3),
                    CitationTextObject(citation_id="cit2", index=28),
                ],
                expected_result='This ```{"type": "citation", "citation_id": "cit1"}```  is some test text\nHello! ```{"type": "citation", "citation_id": "cit2"}``` ',
            ),
            # Citation and a stock object
            param(
                text="This week AAPL increased",
                text_objects=[
                    CitationTextObject(citation_id="cit1", index=3),
                    StockTextObject(index=10, end_index=13, gbi_id=714, symbol="", company_name=""),
                ],
                expected_result='This ```{"type": "citation", "citation_id": "cit1"}```  week  ```{"type": "stock", "gbi_id": 714, "symbol": "", "company_name": "", "text": "AAPL"}```  increased',
            ),
        ]
    )
    async def test_render_text_objects(
        self, text: str, text_objects: List[TextObject], expected_result: str
    ):
        output = TextObject.render_text_objects(text=text, objects=text_objects)
        self.assertEqual(output, expected_result)

    @parameterized.expand(
        [
            param(
                text="This is a test of AAPL and MSFT, hello there!",
                stocks=[
                    StockID(gbi_id=714, symbol="AAPL", isin=""),
                    StockID(gbi_id=6963, symbol="MSFT", isin=""),
                    StockID(gbi_id=7555, symbol="NVDA", isin=""),
                ],
                expected_objects=[
                    StockTextObject(
                        index=18, end_index=21, gbi_id=714, symbol="AAPL", company_name="Apple Inc."
                    ),
                    StockTextObject(
                        index=27,
                        end_index=30,
                        gbi_id=6963,
                        symbol="MSFT",
                        company_name="Microsoft Corporation",
                    ),
                ],
            ),
            param(
                text="This is a test of AAPL Apple Inc. and MSFT, hello there!",
                stocks=[
                    StockID(gbi_id=714, symbol="AAPL", isin=""),
                    StockID(gbi_id=6963, symbol="MSFT", isin=""),
                    StockID(gbi_id=7555, symbol="NVDA", isin=""),
                ],
                expected_objects=[
                    StockTextObject(
                        index=18, end_index=32, gbi_id=714, symbol="AAPL", company_name="Apple Inc."
                    ),
                    StockTextObject(
                        index=38,
                        end_index=41,
                        gbi_id=6963,
                        symbol="MSFT",
                        company_name="Microsoft Corporation",
                    ),
                ],
            ),
        ]
    )
    async def test_find_stock_references_in_text(
        self, text: str, stocks: List[StockID], expected_objects: List[StockTextObject]
    ):
        actual = await StockTextObject.find_stock_references_in_text(text=text, stocks=stocks)
        self.assertEqual(actual, expected_objects)
