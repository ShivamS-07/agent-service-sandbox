import unittest
import warnings
from unittest import IsolatedAsyncioTestCase

from agent_service.planner.action_decide import FirstActionDecider
from agent_service.types import ChatContext, Message
from agent_service.utils.date_utils import get_now_utc
from agent_service.utils.logs import init_stdout_logging
from agent_service.utils.postgres import get_psql


class TestActionDecider(IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        warnings.filterwarnings(
            "ignore", category=ResourceWarning, message="unclosed.*<ssl.SSLSocket.*>"
        )
        warnings.filterwarnings(
            "ignore", category=ResourceWarning, message="unclosed.*<socket.socket.*>"
        )
        warnings.filterwarnings(
            "ignore",
            category=DeprecationWarning,
            message="The loop argument is deprecated since Python 3.8, and scheduled for removal in Python 3.10",  # noqa
        )

        init_stdout_logging()

    @unittest.skip("Takes too long to run")
    async def test_all_sample_plans_as_first_query(self) -> None:
        action_decider = FirstActionDecider("123")

        print("#####testing Plan related inputs")
        db = get_psql(skip_commit=True)
        all_sample_plans_obj = db.get_all_sample_plans()
        all_sample_plans_input = [plan.input for plan in all_sample_plans_obj]
        for new_input in all_sample_plans_input:
            user_message = Message(
                message=new_input, is_user_message=True, message_time=get_now_utc()
            )
            chat_context = ChatContext(messages=[user_message])
            result = await action_decider.decide_action(chat_context)  # type: ignore
            if result.lower() != "plan":
                print("=" * 100)
                print(result)
                print(new_input)
            chat_context.messages.pop()

        print("#####testing None related inputs")
        none_related_inputs = [
            "Hey how are you?",
            "good morning!",
        ]
        for new_input in none_related_inputs:
            user_message = Message(
                message=new_input, is_user_message=True, message_time=get_now_utc()
            )
            chat_context = ChatContext(messages=[user_message])
            result = await action_decider.decide_action(chat_context)
            if result.lower() != "none":
                print("=" * 100)
                print(result)
                print(new_input)

        print("#####testing Refer related inputs")
        refer_related_inputs = [
            "which data source do you use?",
            "how can I use your tools?",
            "How to sey up a notification?",
        ]
        for new_input in refer_related_inputs:
            user_message = Message(
                message=new_input, is_user_message=True, message_time=get_now_utc()
            )
            chat_context = ChatContext(messages=[user_message])
            result = await action_decider.decide_action(chat_context)
            if result.lower() != "refer":
                print("=" * 100)
                print(result)
                print(new_input)
