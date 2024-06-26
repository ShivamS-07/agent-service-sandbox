import datetime
import unittest

from agent_service.io_type_utils import HistoryEntry
from agent_service.io_types.output import CitationOutput, CitationType, OutputType
from agent_service.io_types.stock import StockID
from agent_service.io_types.text import (
    NewsPoolArticleText,
    StockNewsDevelopmentText,
    Text,
    TextCitation,
    TextOutput,
)
from agent_service.utils.postgres import SyncBoostedPG


class TestOutputs(unittest.IsolatedAsyncioTestCase):
    async def test_news_article_text_citation_outputs(self):
        t = NewsPoolArticleText(
            id="6e6fd9ed-a4a2-431c-a21c-3990bc191f35",
            stock_id=StockID(gbi_id=714, symbol="AAPL", isin="", company_name=""),
        )
        rich = await t.to_rich_output(pg=SyncBoostedPG())
        self.assertEqual(
            rich,
            TextOutput(
                output_type=OutputType.TEXT,
                title="",
                citations=[
                    CitationOutput(
                        id="6e6fd9ed-a4a2-431c-a21c-3990bc191f35",
                        citation_type=CitationType.LINK,
                        name="forbes.com",
                        link="https://www.forbes.com/sites/jenamcgregor/2024/06/10/former-google-exec-laszlo-bocks-next-startup-the-productivity-theater-problem-and-more/",  # noqa
                        summary="Former Google Exec Laszlo Bock’s Next ‘Startup,’ The Productivity Theater Problem And More",  # noqa
                        published_at=datetime.datetime(
                            2024, 6, 10, 18, 53, 57, tzinfo=datetime.timezone.utc
                        ),
                    )
                ],
                val="Former Google Exec Laszlo Bock’s Next ‘Startup,’ The Productivity Theater Problem And More:\nA report from software firm BambooHR has identified the \"green status effect\" where employees feel more pressured to appear as though they're working more than they are, rather than actual results. The study also found that 18% of HR professionals and 25% of C-Suite executives and vice presidents said that one of the goals of an office mandate was to prompt some voluntary layoffs. The Conference Board also found a significant increase in voluntary turnover at companies with full-time office presence requirements, which grew twice as fast (16%) as it did at companies without hybrid or fully remote workforces (each 8%). The survey also found those who let people choose their onsite work schedules less trouble retaining workers than those that either mandated or strongly encouraged an onsite schedule. The Labor Department's monthly jobs report showed non-farm U.S. payrolls expanding by 272,000, which will not help hopes for interest rate cuts. The unemployment rate was reported at 4%, up from 3.7% the previous year.",  # noqa
                score=None,
            ),
        )

    async def test_stock_dev_text_citation_outputs(self):
        t = StockNewsDevelopmentText(
            id="e604e22d-c4a0-49bb-95cb-22f26f59e0dc",
            stock_id=StockID(gbi_id=714, symbol="AAPL", isin="", company_name=""),
        )
        rich = await t.to_rich_output(pg=SyncBoostedPG())
        self.assertEqual(
            rich,
            TextOutput(
                output_type=OutputType.TEXT,
                title="",
                citations=[
                    CitationOutput(
                        id="faa3d193-1c91-4f5e-8a58-b6fcce3d6f2d",
                        citation_type=CitationType.LINK,
                        name="businessinsider.com",
                        link="https://www.businessinsider.com/ceo-harder-to-buy-apple-watch-christmas-2023-12",  # noqa
                        summary="The CEO who went head-to-head against Apple — and won",
                        published_at=datetime.datetime(
                            2023, 12, 27, 16, 18, 6, tzinfo=datetime.timezone.utc
                        ),
                    )
                ],
                val="Rising Tech Competition:\nApple Inc. is facing rising tech competition as it loses a patent dispute to Masimo, leading to the removal of certain Apple Watch models from its stores, and as a key design executive leaves for a new AI project. Additionally, MicroStrategy's stock performance has surpassed Apple's due to strategic investments in Bitcoin.",  # noqa
                score=None,
            ),
        )

    @unittest.skip("Failing for now, need to use mock data")
    async def test_multiple_text_citation_outputs(self):
        # A fake summarized text that has multiple citations of different types.
        t = Text(
            val="Test text",
            history=[
                HistoryEntry(
                    explanation="",
                    citations=[
                        TextCitation(
                            source_text=NewsPoolArticleText(
                                id="6e6fd9ed-a4a2-431c-a21c-3990bc191f35",
                                stock_id=StockID(
                                    gbi_id=714, symbol="AAPL", isin="", company_name=""
                                ),
                            )
                        ),
                        TextCitation(
                            source_text=StockNewsDevelopmentText(
                                id="d207aeaa-cd9d-4571-af64-8ddff6ce06a2",
                                stock_id=StockID(
                                    gbi_id=714, symbol="AAPL", isin="", company_name=""
                                ),
                            )
                        ),
                    ],
                )
            ],
        )
        rich = await t.to_rich_output(pg=SyncBoostedPG())
        self.assertEqual(
            rich,
            TextOutput(
                output_type=OutputType.TEXT,
                title="",
                citations=[
                    CitationOutput(
                        id="6e6fd9ed-a4a2-431c-a21c-3990bc191f35",
                        citation_type=CitationType.LINK,
                        name="forbes.com",
                        link="https://www.forbes.com/sites/jenamcgregor/2024/06/10/former-google-exec-laszlo-bocks-next-startup-the-productivity-theater-problem-and-more/",  # noqa
                    ),
                    CitationOutput(
                        id="b4060d44-a257-42ae-97d2-4cee3698b148",
                        citation_type=CitationType.LINK,
                        name="businessinsider.com",
                        link="https://www.businessinsider.com/apple-to-allow-iphone-text-scheduling-ios-software-update-2024-6",  # noqa
                    ),
                ],
                val="Test text",
                score=None,
            ),
        )
