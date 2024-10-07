# flake8: noqa
import datetime
import unittest
from typing import List

from parameterized import param, parameterized

from agent_service.io_type_utils import HistoryEntry, TableColumnType
from agent_service.io_types.stock import StockID
from agent_service.io_types.text import Text
from agent_service.io_types.text_objects import (
    CitationTextObject,
    StockTextObject,
    TextObject,
    TextObjectType,
)
from agent_service.utils.postgres import SyncBoostedPG


class TestTextObjects(unittest.IsolatedAsyncioTestCase):
    @parameterized.expand(
        [
            param(text="", text_objects=[], expected_result=""),
            param(text="testing!", text_objects=[], expected_result="testing!"),
            param(
                text="This is some test text\nHello!",
                text_objects=[CitationTextObject(citation_id="cit1", index=3)],
                expected_result='This```{"type": "citation", "citation_id": "cit1"}``` is some test text\nHello!',
            ),
            # Two citations in the same place
            param(
                text="This is some test text\nHello!",
                text_objects=[
                    CitationTextObject(citation_id="cit1", index=3),
                    CitationTextObject(citation_id="cit2", index=3),
                ],
                expected_result='This```{"type": "citation", "citation_id": "cit1"}``` ```{"type": "citation", "citation_id": "cit2"}``` is some test text\nHello!',
            ),
            param(
                text="This is some test text\nHello!",
                text_objects=[
                    CitationTextObject(citation_id="cit1", index=3),
                    CitationTextObject(citation_id="cit2", index=28),
                ],
                expected_result='This```{"type": "citation", "citation_id": "cit1"}``` is some test text\nHello!```{"type": "citation", "citation_id": "cit2"}```',
            ),
            # Citation and a stock object
            param(
                text="This week AAPL increased",
                text_objects=[
                    CitationTextObject(citation_id="cit1", index=3),
                    StockTextObject(
                        index=10, end_index=13, gbi_id=714, symbol="", company_name="", isin=""
                    ),
                ],
                expected_result='This```{"type": "citation", "citation_id": "cit1"}``` week ```{"gbi_id": 714, "symbol": "", "company_name": "", "isin": "", "sector": null, "subindustry": null, "exchange": null, "type": "stock", "text": "AAPL"}``` increased',  # noqa
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
                text=(
                    "This is a test of AAPL and MSFT, hello there,"
                    " also Other test should not be hit!"
                ),
                stocks={
                    "Apple": StockID(gbi_id=714, symbol="AAPL", isin=""),
                    "Microsoft": StockID(gbi_id=6963, symbol="MSFT", isin=""),
                },
                expected_output=[
                    TextObject(type=TextObjectType.STOCK, index=18, end_index=21),
                    TextObject(type=TextObjectType.STOCK, index=27, end_index=30),
                ],
            ),
            param(
                text="Another test with Microsoft and Apple, but also NVDA and stuff.",
                stocks={
                    "Microsoft": StockID(gbi_id=6963, symbol="MSFT", isin=""),
                    "Apple": StockID(gbi_id=714, symbol="AAPL", isin=""),
                    "Nvidia": StockID(gbi_id=7555, symbol="NVDA", isin=""),
                },
                expected_output=[
                    TextObject(type=TextObjectType.STOCK, index=18, end_index=26),
                    TextObject(type=TextObjectType.STOCK, index=32, end_index=36),
                    TextObject(type=TextObjectType.STOCK, index=48, end_index=51),
                ],
            ),
            param(
                text="Another test Apple Inc. with Tesla, Inc and AAPL, but also Nvidia Corporation and stuff.",
                stocks={
                    "Apple Inc.": StockID(gbi_id=714, symbol="AAPL", isin=""),
                    "Tesla, Inc": StockID(gbi_id=25508, symbol="TSLA", isin=""),
                    "Nvidia Corporation": StockID(gbi_id=7555, symbol="NVDA", isin=""),
                },
                expected_output=[
                    TextObject(type=TextObjectType.STOCK, index=13, end_index=22),
                    TextObject(type=TextObjectType.STOCK, index=29, end_index=38),
                    TextObject(type=TextObjectType.STOCK, index=59, end_index=76),
                ],
            ),
            param(
                text="Recent news on Apple (AAPL) includes several significant updates. Apple has announced a new patent for the MacBook Pro, which suggests expanding touch functionality to nearly all surfaces around the keyboard and potentially reimagining the trackpad as a dynamic display. The company has also launched a web version of its Podcasts app, allowing users to search, browse, and listen to podcasts through various web browsers. Additionally, Apple has released a firmware update for the Beats Studio Pro headphones, introducing an audio sharing feature. The Apple Watch Series 10 will feature larger screen sizes, advanced LTPO technology, and new health sensors for hypertension and sleep apnea. Apple has also extended the AppleCare+ coverage extension period from 30 to 45 days. Furthermore, the company is set to begin assembling the iPhone 16 Pro models in India soon after their launch. Lastly, Apple has announced that Matt Fischer, the vice president in charge of the App Store, will leave the company in October.\nTesla (TSLA) has also been in the news with several key developments. The company is hiring employees to help accumulate data for its Optimus humanoid robot project, which involves wearing a motion capture suit and a virtual reality headset. Tesla has launched the Powerwall 3 in Australia and New Zealand, featuring a battery capacity of 13.5 kWh and a maximum charging and discharging capacity of 5 kW. Additionally, Tesla's China Megafactory, aimed at producing Megapacks for energy storage, is reported to be 45% complete, with plans to start production in Q1 2025. The company is also expanding its GigaFactory network globally, including plans for seven new factories and a specific focus on establishing a facility in Indonesia.",  # noqa
                stocks={
                    "Apple": StockID(gbi_id=714, symbol="AAPL", isin=""),
                    "Tesla": StockID(gbi_id=25508, symbol="TSLA", isin=""),
                },
                expected_output=[
                    TextObject(type=TextObjectType.STOCK, index=22, end_index=25),
                    TextObject(type=TextObjectType.STOCK, index=1024, end_index=1027),
                ],
            ),
            param(
                text="Recent news on Apple includes several significant updates. Apple (AAPL) has announced a new patent for the MacBook Pro,",
                stocks={
                    "Apple": StockID(gbi_id=714, symbol="AAPL", isin=""),
                },
                expected_output=[
                    TextObject(type=TextObjectType.STOCK, index=15, end_index=19),
                ],
            ),
            param(
                text="Recent news on TSMC includes several significant updates.",
                stocks={
                    "TSMC": StockID(gbi_id=714, symbol="TSM", isin=""),
                },
                expected_output=[
                    TextObject(type=TextObjectType.STOCK, index=15, end_index=18),
                ],
            ),
            param(
                text="Tata Consultancy Services (TCS) is a prominent IT services company. AT&T Inc. is a leading telecommunications and technology services provider.",
                stocks={
                    "Tata Consultancy Services": StockID(gbi_id=714, symbol="TCS", isin=""),
                    "AT&T Inc.": StockID(gbi_id=714, symbol="T", isin=""),
                },
                expected_output=[
                    TextObject(type=TextObjectType.STOCK, index=27, end_index=29),
                    TextObject(type=TextObjectType.STOCK, index=68, end_index=76),
                ],
            ),
        ]
    )
    async def test__extract_stock_tags_from_text(
        self,
        text: str,
        stocks: dict[str, StockID],
        expected_output: list[TextObject],
    ):
        stock_objects = await TextObject._extract_stock_tags_from_text(
            text=text,
            stocks=stocks,
            db=SyncBoostedPG(),
        )
        # Make sure that the locations of the stock objects' tickers are the
        # same as they are in the original text
        for i, stock_obj in enumerate(stock_objects):
            self.assertTrue(
                stock_obj.index == expected_output[i].index
                and stock_obj.end_index == expected_output[i].end_index
            )

    async def test_generic_text_objects_in_history(self):
        t = Text(
            history=[
                HistoryEntry(
                    title=TableColumnType.BOOLEAN,
                    entry_type=TableColumnType.BOOLEAN,
                    explanation=True,
                ),
                HistoryEntry(
                    title=TableColumnType.STRING,
                    entry_type=TableColumnType.STRING,
                    explanation="hello",
                ),
                HistoryEntry(
                    title=TableColumnType.FLOAT,
                    entry_type=TableColumnType.FLOAT,
                    explanation=123.5363059602538608436,
                ),
                HistoryEntry(
                    title=TableColumnType.INTEGER,
                    entry_type=TableColumnType.INTEGER,
                    explanation=123,
                ),
                HistoryEntry(
                    title=TableColumnType.CURRENCY,
                    entry_type=TableColumnType.CURRENCY,
                    explanation=123.0,
                    unit="USD",
                ),
                HistoryEntry(
                    title=TableColumnType.DATE,
                    entry_type=TableColumnType.DATE,
                    explanation=datetime.date(2024, 1, 1),
                ),
                HistoryEntry(
                    title=TableColumnType.DATETIME,
                    entry_type=TableColumnType.DATETIME,
                    explanation=datetime.datetime(2024, 1, 1),
                ),
                HistoryEntry(
                    title=TableColumnType.QUARTER,
                    entry_type=TableColumnType.QUARTER,
                    explanation="2024Q2",
                ),
                HistoryEntry(
                    title=TableColumnType.PERCENT,
                    entry_type=TableColumnType.PERCENT,
                    explanation=0.35,
                ),
                HistoryEntry(
                    title=TableColumnType.DELTA, entry_type=TableColumnType.DELTA, explanation=43.35
                ),
                HistoryEntry(
                    title=TableColumnType.PCT_DELTA,
                    entry_type=TableColumnType.PCT_DELTA,
                    explanation=0.35,
                ),
            ]
        )
        result = t.history_to_str_with_text_objects()
        expected = """- **boolean**: ```{"type": "boolean", "value": true, "unit": null}```
- **string**: hello
- **float**: ```{"type": "float", "value": 123.53630596025386, "unit": null}```
- **integer**: ```{"type": "integer", "value": 123, "unit": null}```
- **currency**: ```{"type": "currency", "value": 123.0, "unit": "USD"}```
- **date**: ```{"type": "date", "value": "2024-01-01", "unit": null}```
- **datetime**: ```{"type": "datetime", "value": "2024-01-01T00:00:00", "unit": null}```
- **quarter**: ```{"type": "quarter", "value": "2024Q2", "unit": null}```
- **percent**: ```{"type": "percent", "value": 0.35, "unit": null}```
- **delta**: ```{"type": "delta", "value": 43.35, "unit": null}```
- **pct_delta**: ```{"type": "pct_delta", "value": 0.35, "unit": null}```"""
        self.assertEqual(result, expected)
