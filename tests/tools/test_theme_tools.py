import unittest
from unittest import IsolatedAsyncioTestCase

from agent_service.io_types.text import ThemeNewsDevelopmentText, ThemeText
from agent_service.tools.themes import (
    GetThemeDevelopmentNewsArticlesInput,
    GetThemeDevelopmentNewsInput,
    GetTopNThemesInput,
    get_news_articles_for_theme_developments,
    get_news_developments_about_theme,
    get_top_N_themes,
)
from agent_service.types import PlanRunContext


class TestThemeDevelopmentNews(IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.context = PlanRunContext.get_dummy()

    async def test_get_news_developments_about_theme_raisinginterstrate(self):
        # theme: raising interest rate with id c5bd0897-5187-4df8-9abb-b5bf0bb7d090
        self.args = GetThemeDevelopmentNewsInput(
            themes=[ThemeText(id="c5bd0897-5187-4df8-9abb-b5bf0bb7d090")]
        )
        result = await get_news_developments_about_theme(self.args, self.context)
        # print(len(result))
        self.assertGreater(len(result), 0)


class TestThemeDevelopmentNewsArticles(IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.context = PlanRunContext.get_dummy()

    async def test_get_news_articles_for_theme_raisinginterstrate(self):
        # development: Lower Inflation Expectations Due to High Rates with id 8fce3d05-e9bc-4ecd-9d8e-227ce71a5e5c
        self.args = GetThemeDevelopmentNewsArticlesInput(
            developments_list=[ThemeNewsDevelopmentText(id="8fce3d05-e9bc-4ecd-9d8e-227ce71a5e5c")],
            start_date="2023-01-01 00:00:00",
        )
        result = await get_news_articles_for_theme_developments(self.args, self.context)
        self.assertGreater(len(result), 0)


@unittest.skip("Skipping this test.")
class TestGetTopNThemes(IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.context = PlanRunContext.get_dummy()

    async def test_get_top_N_themes(self):
        self.args = GetTopNThemesInput(
            date_range="1M",
            number_per_section=3,
        )
        result = await get_top_N_themes(self.args, self.context)
        self.assertIsInstance(result[0], ThemeText)
        self.assertEqual(len(result), 3)
