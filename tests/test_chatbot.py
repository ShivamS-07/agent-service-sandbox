import unittest
import warnings
from unittest import IsolatedAsyncioTestCase

from agent_service.chatbot.chatbot import Chatbot
from agent_service.planner.planner_types import (
    ErrorInfo,
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
    async def test_full_chat(self) -> None:
        input_text = (
            "Can you give me a single summary of news published in the last month "
            "about Generative AI at Meta, Apple, and Microsoft?"
        )
        user_message = Message(message=input_text, is_user_message=True)
        chat_context = ChatContext(messages=[user_message])
        chatbot = Chatbot("123")
        preplan_response = await chatbot.generate_initial_preplan_response(chat_context)
        print(preplan_response)
        chat_context.messages.append(Message(message=preplan_response, is_user_message=False))
        midplan_response = await chatbot.generate_initial_midplan_response(chat_context)
        print(midplan_response)
        chat_context.messages.append(Message(message=midplan_response, is_user_message=False))
        failed_plan_response = await chatbot.generate_initial_plan_failed_response_suggestions(
            chat_context
        )
        print(failed_plan_response)
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
                args={"topic": "Generative AI", "texts": Variable(var_name="news_descriptions")},
                description="Filter news descriptions to only those related to Generative AI",
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
        fake_output = [
            "Meta released a new open source model Llama 3 while Microsoft continues to milk OpenAI for all their worth. By comparison, all the news about Apple is how they are getting left in dust by everyone else. Losers!"  # noqa: E501
        ]
        complete_response = await chatbot.generate_execution_complete_response(
            chat_context, execution_plan, fake_output
        )
        chat_context.messages.append(Message(message=complete_response, is_user_message=False))
        no_action_input = "Oh, that's interesting, thanks!"
        chat_context.messages.append(Message(message=no_action_input, is_user_message=True))
        no_action_response = await chatbot.generate_input_update_no_action_response(chat_context)
        print(no_action_response)
        chat_context.messages.pop()
        rerun_input = "Okay, but can you rewrite the summary in a more casual style?"
        chat_context.messages.append(Message(message=rerun_input, is_user_message=True))
        rerun_response = await chatbot.generate_input_update_rerun_response(
            chat_context, execution_plan, "summarize_texts"
        )
        print(rerun_response)
        chat_context.messages.pop()
        replan_input = "Can you include Amazon in the summary?"
        chat_context.messages.append(Message(message=replan_input, is_user_message=True))
        replan_preplan_response = await chatbot.generate_input_update_replan_preplan_response(
            chat_context
        )
        print(replan_preplan_response)
        new_nodes = plan_nodes[:]
        new_nodes[1] = ToolExecutionNode(
            tool_name="stock_identifier_lookup_multi",
            args={"stock_names": ["Meta", "Apple", "Microsoft", "Amazon"]},
            description="Convert company names to stock identifiers",
            output_variable_name="stock_ids",
            is_output_node=False,
        )
        new_nodes[2] = ToolExecutionNode(
            tool_name="get_news_developments_about_companies",
            args={
                "stock_ids": Variable(var_name="stock_ids"),
                "start_date": Variable(var_name="start_date"),
            },
            description="Get news developments for Meta, Apple, Microsoft, and Amazon since last month",
            output_variable_name="news_developments",
            is_output_node=False,
        )
        new_execution_plan = ExecutionPlan(nodes=new_nodes)
        replan_postplan_response = await chatbot.generate_input_update_replan_postplan_response(
            chat_context, execution_plan, new_execution_plan
        )
        print(replan_postplan_response)
        chat_context.messages.pop()
        chat_context.messages.pop()
        error_info = ErrorInfo(
            error="Exception: index error",
            step=plan_nodes[5],
            change="Include all AI news in the filter, not just Generative AI news",
        )
        error_preplan_response = await chatbot.generate_error_replan_preplan_response(
            chat_context, execution_plan, error_info
        )
        print(error_preplan_response)
        chat_context.messages.append(Message(message=error_preplan_response, is_user_message=False))
        new_node = ToolExecutionNode(
            tool_name="filter_texts_by_topic",
            args={
                "topic": "Artificial Intelligence",
                "texts": Variable(var_name="news_descriptions"),
            },
            description="Filter news descriptions to only those related to artificial intelligences",
            output_variable_name="filtered_texts",
            is_output_node=True,
        )
        new_nodes = plan_nodes[:]
        new_nodes[5] = new_node
        new_execution_plan = ExecutionPlan(nodes=new_nodes)
        error_postplan_response = await chatbot.generate_error_replan_postplan_response(
            chat_context, execution_plan, new_execution_plan
        )
        print(error_postplan_response)
