import datetime
import unittest
from unittest import IsolatedAsyncioTestCase
from uuid import uuid4

from agent_service.io_types.text import Text
from agent_service.tools.commentary.tools import (
    GetCommentaryTextsInput,
    WriteCommentaryInput,
    get_commentary_texts,
    write_commentary,
)
from agent_service.types import ChatContext, Message, PlanRunContext
from agent_service.utils.date_utils import get_now_utc


class TestCommentary(IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        input_text = (
            "Write a commentary on impact of cloud computing on military industrial complex."
        )
        user_message = Message(message=input_text, is_user_message=True, message_time=get_now_utc())
        chat_context = ChatContext(messages=[user_message])

        self.context = PlanRunContext(
            agent_id="7cb9fb8f-690e-4535-8b48-f6e63494c366",
            plan_id="b3330500-9870-480d-bcb1-cf6fe6b487e3",
            user_id=str(uuid4()),
            plan_run_id=str(uuid4()),
            chat=chat_context,
            skip_db_commit=True,
            skip_task_cache=True,
            run_tasks_without_prefect=True,
        )

    @unittest.skip("Skipping for now until we can mock GPT output easily")
    async def test_write_commentary(self):

        texts = await get_commentary_texts(
            GetCommentaryTextsInput(
                topics=["cloud computing", "military industrial complex"],
                start_date=datetime.date(2024, 4, 1),
            ),
            self.context,
        )
        print("Length of texts: ", len(texts))  # type: ignore

        self.args = WriteCommentaryInput(
            texts=texts,  # type: ignore
        )
        result = await write_commentary(self.args, self.context)
        self.assertIsInstance(result, Text)
