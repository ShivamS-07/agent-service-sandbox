from unittest import IsolatedAsyncioTestCase

from agent_service.io_types.text import ThemeNewsDevelopmentText, ThemeText
from agent_service.tools.themes import (
    GetThemeDevelopmentNewsArticlesInput,
    GetThemeDevelopmentNewsInput,
    get_news_articles_for_theme_developments,
    get_news_developments_about_theme,
)
from agent_service.types import PlanRunContext


class TestThemeDevelopmentNews(IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.context = PlanRunContext.get_dummy()

    async def test_get_news_developments_about_theme_raisinginterstrate(self):
        # theme: raising interest rate with id c5bd0897-5187-4df8-9abb-b5bf0bb7d090
        self.args = GetThemeDevelopmentNewsInput(
            theme=ThemeText(id="c5bd0897-5187-4df8-9abb-b5bf0bb7d090")
        )
        result = await get_news_developments_about_theme(self.args, self.context)
        # print(len(result))
        self.assertGreater(len(result), 0)


class TestThemeDevelopmentNewsArticles(IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.context = PlanRunContext.get_dummy()

    async def test_get_news_developments_about_theme_raisinginterstrate(self):
        # development: US Debt Inflection Point with id 332102ba-1fdb-4f83-9172-af3b734d1a61
        self.args = GetThemeDevelopmentNewsArticlesInput(
            development=ThemeNewsDevelopmentText(id="332102ba-1fdb-4f83-9172-af3b734d1a61")
        )
        result = await get_news_articles_for_theme_developments(self.args, self.context)
        # print(len(result))
        self.assertGreater(len(result), 0)
