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
                expected_result='This```{"type": "citation", "citation_id": "cit1"}``````{"type": "citation", "citation_id": "cit2"}``` is some test text\nHello!',
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
                original_text=(
                    "This is a test of AAPL and MSFT, hello there,"
                    " also Other test should not be hit!"
                ),
                tagged_text=(
                    "This is a test of [[AAPL]] and [[MSFT]],  hello there,"
                    " also [[Other]] test should not be hit!"
                ),
                stocks=[
                    StockID(gbi_id=714, symbol="AAPL", isin=""),
                    StockID(gbi_id=6963, symbol="MSFT", isin=""),
                    StockID(gbi_id=7555, symbol="NVDA", isin=""),
                ],
            ),
            param(
                original_text="Another test with TSLA and AAPL, but also NVDA and stuff.",
                # Random spaces simulating GPT weirdness
                tagged_text="Another [[test]] with [[TSLA]]    and [[AAPL]], but also [[NVDA]] and   stuff.",
                stocks=[
                    StockID(gbi_id=714, symbol="AAPL", isin=""),
                    StockID(gbi_id=6963, symbol="MSFT", isin=""),
                    StockID(gbi_id=7555, symbol="NVDA", isin=""),
                ],
            ),
            param(
                original_text="Another test Apple Inc with TSLA and AAPL, but also NVDA and stuff.",
                # Random spaces simulating GPT weirdness
                tagged_text="Another [[test]] [[Apple Inc]] with [[TSLA]]    and [[AAPL]], but also [[NVDA]] and   stuff.",
                stocks=[
                    StockID(gbi_id=714, symbol="AAPL", isin="", company_name="Apple Inc"),
                    StockID(gbi_id=6963, symbol="MSFT", isin=""),
                    StockID(gbi_id=7555, symbol="NVDA", isin=""),
                ],
            ),
            param(
                original_text="Recent news on Apple (AAPL) includes several significant updates. Apple has announced a new patent for the MacBook Pro, which suggests expanding touch functionality to nearly all surfaces around the keyboard and potentially reimagining the trackpad as a dynamic display. The company has also launched a web version of its Podcasts app, allowing users to search, browse, and listen to podcasts through various web browsers. Additionally, Apple has released a firmware update for the Beats Studio Pro headphones, introducing an audio sharing feature. The Apple Watch Series 10 will feature larger screen sizes, advanced LTPO technology, and new health sensors for hypertension and sleep apnea. Apple has also extended the AppleCare+ coverage extension period from 30 to 45 days. Furthermore, the company is set to begin assembling the iPhone 16 Pro models in India soon after their launch. Lastly, Apple has announced that Matt Fischer, the vice president in charge of the App Store, will leave the company in October.\nTesla (TSLA) has also been in the news with several key developments. The company is hiring employees to help accumulate data for its Optimus humanoid robot project, which involves wearing a motion capture suit and a virtual reality headset. Tesla has launched the Powerwall 3 in Australia and New Zealand, featuring a battery capacity of 13.5 kWh and a maximum charging and discharging capacity of 5 kW. Additionally, Tesla's China Megafactory, aimed at producing Megapacks for energy storage, is reported to be 45% complete, with plans to start production in Q1 2025. The company is also expanding its GigaFactory network globally, including plans for seven new factories and a specific focus on establishing a facility in Indonesia.",  # noqa
                tagged_text="Recent news on Apple ([[AAPL]]) includes several significant updates. Apple has announced a new patent for the MacBook Pro, which suggests expanding touch functionality to nearly all surfaces around the keyboard and potentially reimagining the trackpad as a dynamic display. The company has also launched a web version of its Podcasts app, allowing users to search, browse, and listen to podcasts through various web browsers. Additionally, Apple has released a firmware update for the Beats Studio Pro headphones, introducing an audio sharing feature. The Apple Watch Series 10 will feature larger screen sizes, advanced LTPO technology, and new health sensors for hypertension and sleep apnea. Apple has also extended the AppleCare+ coverage extension period from 30 to 45 days. Furthermore, the company is set to begin assembling the iPhone 16 Pro models in India soon after their launch. Lastly, Apple has announced that Matt Fischer, the vice president in charge of the App Store, will leave the company in October.\nTesla ([[TSLA]]) has also been in the news with several key developments. The company is hiring employees to help accumulate data for its Optimus humanoid robot project, which involves wearing a motion capture suit and a virtual reality headset. Tesla has launched the Powerwall 3 in Australia and New Zealand, featuring a battery capacity of 13.5 kWh and a maximum charging and discharging capacity of 5 kW. Additionally, Tesla's China Megafactory, aimed at producing Megapacks for energy storage, is reported to be 45% complete, with plans to start production in Q1 2025.       The company is also expanding its GigaFactory network globally, including plans for seven new factories and a specific focus on establishing a facility in Indonesia.",  # noqa
                stocks=[
                    StockID(gbi_id=714, symbol="AAPL", isin=""),
                    StockID(gbi_id=1234, symbol="TSLA", isin=""),
                ],
            ),
        ]
    )
    async def test__extract_stock_tags_from_text(
        self,
        original_text,
        tagged_text: str,
        stocks: List[StockID],
    ):
        stock_map = {}
        for stock in stocks:
            if stock.symbol:
                stock_map[stock.symbol] = stock
            if stock.company_name:
                stock_map[stock.company_name] = stock
        stock_objects = await TextObject._extract_stock_tags_from_text(
            original_text=original_text,
            tagged_text=tagged_text,
            symbol_to_stock_map=stock_map,
            db=SyncBoostedPG(),
        )
        # Make sure that the locations of the stock objects' tickers are the
        # same as the are in the original text
        for stock_obj in stock_objects:
            company_name_matches = False
            symbol_name_matches = False
            if stock_obj.company_name:
                start = original_text.index(stock_obj.company_name)
                end = start + len(stock_obj.company_name) - 1
                if stock_obj.index == start and stock_obj.end_index == end:
                    company_name_matches = True
            if stock_obj.symbol:
                start = original_text.index(stock_obj.symbol)
                end = start + len(stock_obj.symbol) - 1
                if stock_obj.index == start and stock_obj.end_index == end:
                    symbol_name_matches = True

            self.assertTrue(company_name_matches or symbol_name_matches)
