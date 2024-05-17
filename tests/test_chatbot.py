import unittest
import warnings
from unittest import IsolatedAsyncioTestCase

from agent_service.chatbot.chatbot import Chatbot
from agent_service.planner.planner_types import (
    ExecutionPlan,
    ToolExecutionNode,
    Variable,
)
from agent_service.types import ChatContext, Message
from agent_service.utils.logs import init_stdout_logging


class TestPlans(IsolatedAsyncioTestCase):
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
    async def test_initial_chat(self) -> None:
        input_text = (
            "Can you give me a single summary of news published in the last month "
            "about machine learning at Meta, Apple, and Microsoft?"
        )
        user_message = Message(message=input_text, is_user_message=True)
        chat_context = ChatContext(messages=[user_message])
        chatbot = Chatbot("123")
        preplan_response = await chatbot.generate_initial_preplan_response(chat_context)
        chat_context.messages.append(Message(message=preplan_response, is_user_message=False))
        plan_nodes = [
            ToolExecutionNode(
                tool_name="get_date_from_date_str",
                args={"time_str": "1 month ago"},
                description='Convert "1 month ago" to a date to use as the start date for news search',
                output_variable_name="start_date",
                is_output_node=False,
            ),
            ToolExecutionNode(
                tool_name="stock_identifier_lookup_multi",
                args={"stock_names": ["Meta", "Apple", "Microsoft"]},
                description="Convert company names to stock identifiers",
                output_variable_name="stock_ids",
                is_output_node=False,
            ),
            ToolExecutionNode(
                tool_name="get_news_developments_about_companies",
                args={
                    "stock_ids": Variable(var_name="stock_ids"),
                    "start_date": Variable(var_name="start_date"),
                },
                description="Get news developments for Meta, Apple, and Microsoft since last month",
                output_variable_name="news_developments",
                is_output_node=False,
            ),
            ToolExecutionNode(
                tool_name="collapse_lists",
                args={"lists_of_lists": Variable(var_name="news_developments")},
                description="Collapse the list of lists of news developments into a single list",
                output_variable_name="collapsed_news_ids",
                is_output_node=False,
            ),
            ToolExecutionNode(
                tool_name="get_news_development_descriptions",
                args={"development_ids": Variable(var_name="collapsed_news_ids")},
                description="Retrieve the text descriptions of the news developments",
                output_variable_name="news_descriptions",
                is_output_node=False,
            ),
            ToolExecutionNode(
                tool_name="filter_texts_by_topic",
                args={"topic": "machine_learning", "texts": Variable(var_name="news_descriptions")},
                description="Filter news descriptions to only those related to machine learning",
                output_variable_name="filtered_texts",
                is_output_node=True,
            ),
            ToolExecutionNode(
                tool_name="summarize_texts",
                args={"texts": Variable(var_name="filtered_texts")},
                description="Summarize the news descriptions into a single summary text",
                output_variable_name="summary",
                is_output_node=True,
            ),
        ]
        execution_plan = ExecutionPlan(nodes=plan_nodes)
        postplan_response = await chatbot.generate_initial_postplan_response(
            chat_context, execution_plan
        )
        chat_context.messages.append(Message(message=postplan_response, is_user_message=False))
        fake_output = "Meta released a new open source model Llama 3 while Microsoft continues to milk OpenAI for all their worth. By comparison, all the news about Apple is how they are getting left in dust by everyone else. Losers!"  # noqa: E501
        complete_response = await chatbot.generate_execution_complete_response(
            chat_context, execution_plan, fake_output
        )
        chat_context.messages.append(Message(message=complete_response, is_user_message=False))
        print(chat_context.get_gpt_input())
