import datetime
import unittest
from unittest import IsolatedAsyncioTestCase
from uuid import uuid4

from agent_service.io_types.text import Text
from agent_service.tools.commentary import (
    GetCommentaryTextsInput,
    WriteCommentaryInput,
    get_commentary_texts,
    write_commentary,
)
from agent_service.types import ChatContext, Message, PlanRunContext
from agent_service.utils.date_utils import get_now_utc


@unittest.skip("Skipping this test class till mock is implemented.")
class TestWriteCommentary(IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        input_text = (
            "Write a commentary on impact of cloud computing on military industrial complex."
        )
        user_message = Message(message=input_text, is_user_message=True, message_time=get_now_utc())
        chat_context = ChatContext(messages=[user_message])

        self.context = PlanRunContext(
            agent_id=str(uuid4()),
            plan_id=str(uuid4()),
            user_id=str(uuid4()),
            plan_run_id=str(uuid4()),
            chat=chat_context,
            run_tasks_without_prefect=True,
        )

    async def test_write_commentary(self):
        texts = await get_commentary_texts(
            GetCommentaryTextsInput(
                topics=["cloud computing", "military industrial complex"],
                start_date=datetime.date(2024, 4, 1),
                general_commentary_flag=False,
            ),
            self.context,
        )
        print("Length of texts: ", len(texts))
        self.args = WriteCommentaryInput(
            texts=texts,
        )
        result = await write_commentary(self.args, self.context)
        print(result)
        self.assertIsInstance(result, Text)
