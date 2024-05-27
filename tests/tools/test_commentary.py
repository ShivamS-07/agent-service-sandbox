import unittest
from unittest import IsolatedAsyncioTestCase

from agent_service.io_types.text import Text
from agent_service.tools.commentary import WriteCommentaryInput, write_commentary
from agent_service.tools.themes import (
    GetMacroeconomicThemeInput,
    GetThemeDevelopmentNewsArticlesInput,
    GetThemeDevelopmentNewsInput,
    get_macroeconomic_themes,
    get_news_articles_for_theme_developments,
    get_news_developments_about_theme,
)
from agent_service.types import ChatContext, Message, PlanRunContext
from agent_service.utils.date_utils import get_now_utc


@unittest.skip("Skipping this test class till mock is implemented.")
class TestWriteCommentary(IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        input_text = "Write a commentary on interest rates and gen ai."
        user_message = Message(message=input_text, is_user_message=True, message_time=get_now_utc())
        chat_context = ChatContext(messages=[user_message])

        self.context = PlanRunContext(
            agent_id="123",
            plan_id="123",
            user_id="123",
            plan_run_id="123",
            chat=chat_context,
            run_tasks_without_prefect=True,
        )

    async def test_write_commentary(self):
        topics = ["interest rates", "inflation"]
        themes = await get_macroeconomic_themes(
            GetMacroeconomicThemeInput(theme_refs=topics), self.context
        )
        print("Number of themes: ", len(themes))
        developments_list = await get_news_developments_about_theme(
            GetThemeDevelopmentNewsInput(themes=themes), self.context
        )
        print(
            "Number of developments: ",
            sum([len(developments) for developments in developments_list]),
        )
        articles_list = await get_news_articles_for_theme_developments(
            GetThemeDevelopmentNewsArticlesInput(developments_list=developments_list), self.context
        )
        print("Number of articles: ", sum([len(articles) for articles in articles_list]))

        self.args = WriteCommentaryInput(
            topics=topics, themes=themes, developments=developments_list, articles=articles_list
        )
        result = await write_commentary(self.args, self.context)
        print(result)
        self.assertIsInstance(result, Text)
