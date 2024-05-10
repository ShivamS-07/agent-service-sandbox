import datetime
import unittest
import warnings
from typing import Any, List, Type, Union
from unittest import IsolatedAsyncioTestCase
from unittest.case import TestCase

from agent_service.planner.planner import Planner
from agent_service.planner.planner_types import ParsedStep, ToolExecutionNode
from agent_service.tool import ToolArgs, ToolCategory, ToolRegistry, Variable, tool
from agent_service.types import ChatContext, Message, PlanRunContext
from agent_service.utils.date_utils import get_now_utc


def get_test_registry() -> Type[ToolRegistry]:
    class TestRegistry(ToolRegistry):
        pass

    class SummarizeTextInput(ToolArgs):
        texts: List[str]

    @tool(
        description="This function take a list of texts and use an LLM to summarize them into a single text",
        category=ToolCategory.LLM_ANALYSIS,
        tool_registry=TestRegistry,
    )
    async def summarize_texts(args: SummarizeTextInput, context: PlanRunContext) -> str:
        return ""

    class FilterByTopicRelevanceInput(ToolArgs):
        topic: str
        texts: List[str]

    @tool(
        description=(
            "This function takes any list of texts"
            " and uses machine learning to filter texts to only those relevant to the provided topic"
        ),
        category=ToolCategory.LLM_ANALYSIS,
        tool_registry=TestRegistry,
    )
    async def filter_by_topic_relevance(
        args: SummarizeTextInput, context: PlanRunContext
    ) -> List[str]:
        return []

    class GetNewsaboutCompaniesInput(ToolArgs):
        topic: str
        company_identifiers: List[int]
        start_date: datetime.date
        end_date: datetime.date = datetime.date.today()

    @tool(
        description=(
            "This function calls an interal API which provides all the news articles that were"
            "published between the start date and the end date that are relevant to the companies,"
            "the output is a list of list of articles (strings) in plain text,"
            "each list of articles corresponds to one of the companies"
        ),
        category=ToolCategory.DATA_RETRIEVAL_INT,
        tool_registry=TestRegistry,
    )
    async def get_news_about_companies(
        args: GetNewsaboutCompaniesInput, context: PlanRunContext
    ) -> List[List[int]]:
        return [[]]

    class CollapseListsInput(ToolArgs):
        lists_of_lists: List[List[Union[str, int, float, bool]]]

    @tool(
        description="This function collapses a list of lists into a list",
        category=ToolCategory.LIST,
        tool_registry=TestRegistry,
    )
    async def collapse_lists(
        args: CollapseListsInput, context: PlanRunContext
    ) -> List[Union[str, int, float, bool]]:
        return []

    class GetDateFromDateStrInput(ToolArgs):
        time_str: str

    @tool(
        description=(
            "This function takes a string which refers to a time,"
            " either absolute or relative to the current time, and converts it to a Python date"
        ),
        category=ToolCategory.DATES,
        tool_registry=TestRegistry,
    )
    async def get_date_from_date_str(
        args: GetDateFromDateStrInput, context: PlanRunContext
    ) -> datetime.date:
        return datetime.date.today()

    class StockIdentifierLookupInput(ToolArgs):
        stock_str: str

    @tool(
        description="This function takes a string which refers to a stock, and converts it to an integer identifier",
        category=ToolCategory.STOCK,
        tool_registry=TestRegistry,
    )
    async def stock_identifier_lookup(
        args: StockIdentifierLookupInput, context: PlanRunContext
    ) -> int:
        return 0

    class StockIdentifierLookupMultiInput(ToolArgs):
        stock_strs: List[str]

    @tool(
        description=(
            "This function takes a list of strings each of which refers to a stock, "
            "and converts them to integer identifiers"
        ),
        category=ToolCategory.STOCK,
        tool_registry=TestRegistry,
    )
    async def stock_identifier_lookup_multi(
        args: StockIdentifierLookupMultiInput, context: PlanRunContext
    ) -> List[int]:
        return [0]

    return TestRegistry


class TestPlanner(IsolatedAsyncioTestCase):
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

    def setUp(self) -> None:
        self.tool_registry = get_test_registry()

    async def test_tool_registry_works(self) -> None:
        get_test_registry()

    @unittest.skip("Takes too long to run")
    async def test_planner(self) -> None:
        input_text = (
            "Can you give me a single summary of news published in the last month "
            "about machine learning at Meta, Apple, and Microsoft?"
        )
        user_message = Message(content=input_text, is_user=True, timestamp=get_now_utc())
        chat_context = ChatContext(messages=[user_message])
        planner = Planner("", tool_registry=self.tool_registry)
        await planner.create_initial_plan(chat_context)


class TestPlanConstructionValidation(TestCase):
    def setUp(self) -> None:
        self.tool_registry = get_test_registry()

    def test_literal_parsing(self):
        planner = Planner(agent_id="TEST")
        cases = [
            ("True", True),
            ("False", False),
            ("'Test one'", "Test one"),
            ('"Test"', "Test"),
            ("32520", 32520),
            ("3252.42", 3252.42),
            ("-32520", -32520),
            ("-3252.42", -3252.42),
        ]
        for in_str, expected in cases:
            self.assertEqual(
                expected, planner._try_parse_primitive_literal(in_str, expected_type=Any)
            )

    def test_list_parsing(self):
        planner = Planner(agent_id="TEST")
        variable_lookup = {"test1": int, "test2": str}
        cases = [
            ("[True]", [True], List[bool]),
            ("[True, False]", [True, False], List[bool]),
            ("[True,False]", [True, False], List[bool]),
            ("[True, 123,123.1]", [True, 123, 123.1], List[Union[bool, int, float]]),
            (
                "[True, 123,123.1, 'Hi']",
                [True, 123, 123.1, "Hi"],
                List[Union[bool, int, float, str]],
            ),
            ("[1, 2, test1]", [1, 2, Variable(var_name="test1")], List[int]),
            ("['1', '2', test2]", ["1", "2", Variable(var_name="test2")], List[str]),
        ]
        for in_str, expected, typ in cases:
            self.assertEqual(
                expected,
                planner._try_parse_list_literal(
                    in_str, expected_type=typ, variable_lookup=variable_lookup
                ),
            )

    def test_validate_construct_plan(self):
        planner = Planner(agent_id="TEST")
        example_input = [
            ParsedStep(
                output_var="start_date",
                function="get_date_from_date_str",
                arguments={"time_str": '"1 month ago"'},
                description='Convert the string "1 month ago" to a date representing one month ago',
            ),
            ParsedStep(
                output_var="company_ids",
                function="stock_identifier_lookup_multi",
                arguments={"stock_strs": '["Meta", "Apple", "Microsoft"]'},
                description="Convert the company names Meta, Apple, and Microsoft into their integer identifiers",
            ),
            ParsedStep(
                output_var="news_articles",
                function="get_news_about_companies",
                arguments={
                    "topic": '"machine learning"',
                    "company_identifiers": "company_ids",
                    "start_date": "start_date",
                },
                description=(
                    "Retrieve news articles published in the last month about "
                    "machine learning related to Meta, Apple, and Microsoft"
                ),
            ),
            ParsedStep(
                output_var="collapsed_news",
                function="collapse_lists",
                arguments={"lists_of_lists": "news_articles"},
                description="Collapse the list of lists of news articles into a single list",
            ),
            ParsedStep(
                output_var="summary",
                function="summarize_texts",
                arguments={"texts": "collapsed_news"},
                description="Summarize the collapsed list of news articles into a single summary",
            ),
        ]
        execution_plan = planner._validate_and_construct_plan(example_input)

        expected_output = [
            ToolExecutionNode(
                tool_name="get_date_from_date_str",
                args={"time_str": "1 month ago"},
                output_variable_name="start_date",
                is_output_node=False,
            ),
            ToolExecutionNode(
                tool_name="stock_identifier_lookup_multi",
                args={"stock_strs": ["Meta", "Apple", "Microsoft"]},
                output_variable_name="company_ids",
                is_output_node=False,
            ),
            ToolExecutionNode(
                tool_name="get_news_about_companies",
                args={
                    "topic": "machine learning",
                    "company_identifiers": Variable(var_name="company_ids"),
                    "start_date": Variable(var_name="start_date"),
                },
                output_variable_name="news_articles",
                is_output_node=False,
            ),
            ToolExecutionNode(
                tool_name="collapse_lists",
                args={"lists_of_lists": Variable(var_name="news_articles")},
                output_variable_name="collapsed_news",
                is_output_node=False,
            ),
            ToolExecutionNode(
                tool_name="summarize_texts",
                args={"texts": Variable(var_name="collapsed_news")},
                output_variable_name="summary",
                is_output_node=True,
            ),
        ]

        self.assertEqual(execution_plan.nodes, expected_output)
