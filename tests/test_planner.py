import logging
import unittest
import warnings
from typing import Any, Dict, List, Union
from unittest import IsolatedAsyncioTestCase, TestCase

from agent_service.GPT.requests import set_use_global_stub
from agent_service.planner.planner import Planner
from agent_service.planner.planner_types import (
    ExecutionPlan,
    ExecutionPlanParsingError,
    ParsedStep,
    ToolExecutionNode,
    Variable,
    accumulate_type_from_list,
)
from agent_service.types import ChatContext, Message
from agent_service.utils.logs import init_stdout_logging
from tests.planner_utils import get_test_registry

logger = logging.getLogger(__name__)


class TestPlans(IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        set_use_global_stub(False)
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
        # input_text = (
        #     "I need a summary of all the earnings calls from yesterday of companies"
        #     "that might impact stocks in my portfolio"
        # )
        # input_text = (
        #     "I want to see the profit margins of Microsoft for the last year graphed"
        #     "against the average of its peers."
        # )
        # input_text = (
        #     "We are looking for low-to-mid cap, high growth healthcare companies"
        #     " with innovative technologies that can become the standard of care for"
        #     " unmet medical needs"
        # )
        # input_text = (
        #     "I want to find good stocks to short so that if a recession happens I'm protected."
        # )
        user_message = Message(message=input_text, is_user_message=True)
        chat_context = ChatContext(messages=[user_message])
        planner = Planner("", tool_registry=self.tool_registry)
        await planner.create_initial_plan(chat_context)


class TestPlanConstructionValidation(IsolatedAsyncioTestCase):
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

    def test_dict_parsing(self):
        planner = Planner(agent_id="TEST")
        variable_lookup = {"test1": int, "test2": str}
        cases = [
            ('{"test": 1}', {"test": 1}, Dict[str, int]),
            ('{"test": True}', {"test": True}, Dict[str, bool]),
            ('{"test": "hello"}', {"test": "hello"}, Dict[str, str]),
            ('{"test": test1}', {"test": Variable(var_name="test1")}, Dict[str, int]),
        ]
        for in_str, expected, typ in cases:
            self.assertEqual(
                expected,
                planner._try_parse_dict_literal(
                    in_str, expected_type=typ, variable_lookup=variable_lookup
                ),
            )

    def test_step_parse(self):
        planner = Planner(agent_id="TEST")
        example_input = """start_date = get_date_from_date_str(time_str="1 month ago")  # Convert "1 month ago" to a date to use as the start date for news search
stock_ids = stock_identifier_lookup_multi(stock_names=["Meta", "Apple", "Microsoft)"])  # Look up stock identifiers for Meta, Apple, and Microsoft
news_developments = get_news_developments_about_companies(stock_ids=stock_ids, start_date=start_date)  # Get news developments in the last month for Meta, Apple, and Microsoft
collapsed_news_developments = collapse_lists(lists_of_lists=news_developments)  # Collapse the list of lists of news development IDs into a single list
filtered_news = filter_texts_by_topic(topic="machine learning", texts=collapsed_news_developments)  # Filter news descriptions to only those related to machine learning
summary = summarize_texts(texts=filtered_news)  # Summarize the machine learning related news into a single text"""  # noqa: E501
        steps = planner._parse_plan_str(example_input)

        expected_output = [
            ParsedStep(
                original_str='start_date = get_date_from_date_str(time_str="1 month ago")  # Convert "1 month ago" to a date to use as the start date for news search',  # noqa: E501
                output_var="start_date",
                function="get_date_from_date_str",
                arguments={"time_str": '"1 month ago"'},
                description='Convert "1 month ago" to a date to use as the start date for news search',
            ),
            ParsedStep(
                original_str='stock_ids = stock_identifier_lookup_multi(stock_names=["Meta", "Apple", "Microsoft)"])  # Look up stock identifiers for Meta, Apple, and Microsoft',  # noqa: E501
                output_var="stock_ids",
                function="stock_identifier_lookup_multi",
                # Add a parenthesis to test the regex
                arguments={"stock_names": '["Meta", "Apple", "Microsoft)"]'},
                description="Look up stock identifiers for Meta, Apple, and Microsoft",
            ),
            ParsedStep(
                original_str="news_developments = get_news_developments_about_companies(stock_ids=stock_ids, start_date=start_date)  # Get news developments in the last month for Meta, Apple, and Microsoft",  # noqa: E501
                output_var="news_developments",
                function="get_news_developments_about_companies",
                arguments={"stock_ids": "stock_ids", "start_date": "start_date"},
                description="Get news developments in the last month for Meta, Apple, and Microsoft",
            ),
            ParsedStep(
                original_str="collapsed_news_developments = collapse_lists(lists_of_lists=news_developments)  # Collapse the list of lists of news development IDs into a single list",  # noqa: E501
                output_var="collapsed_news_developments",
                function="collapse_lists",
                arguments={"lists_of_lists": "news_developments"},
                description="Collapse the list of lists of news development IDs into a single list",
            ),
            ParsedStep(
                original_str='filtered_news = filter_texts_by_topic(topic="machine learning", texts=collapsed_news_developments)  # Filter news descriptions to only those related to machine learning',  # noqa: E501
                output_var="filtered_news",
                function="filter_texts_by_topic",
                arguments={"topic": '"machine learning"', "texts": "collapsed_news_developments"},
                description="Filter news descriptions to only those related to machine learning",
            ),
            ParsedStep(
                original_str="summary = summarize_texts(texts=filtered_news)  # Summarize the machine learning related news into a single text",  # noqa: E501
                output_var="summary",
                function="summarize_texts",
                arguments={"texts": "filtered_news"},
                description="Summarize the machine learning related news into a single text",
            ),
        ]
        self.assertEqual(steps, expected_output)

    def test_validate_generic_types(self):
        planner = Planner(agent_id="TEST", tool_registry=get_test_registry())

        with self.subTest("should succeed"):
            example_input = [
                ParsedStep(
                    output_var="stock_ids_1",
                    function="stock_identifier_lookup_multi",
                    arguments={"stock_names": '["Meta", "Apple,1", "Microsoft"]'},
                    description="Convert company names to stock identifiers",
                    original_str="",
                ),
                ParsedStep(
                    output_var="stock_ids_2",
                    function="stock_identifier_lookup_multi",
                    arguments={"stock_names": '["Meta", "Apple,1", "Microsoft"]'},
                    description="Convert company names to stock identifiers",
                    original_str="",
                ),
                ParsedStep(
                    output_var="combined",
                    function="add_lists",
                    arguments={"list1": "stock_ids_1", "list2": "stock_ids_2"},
                    description="Merge lists",
                    original_str="",
                ),
                ParsedStep(
                    output_var="result",
                    function="prepare_output",
                    arguments={"object_to_output": "combined", "title": '"test"'},
                    description="Output the result",
                    original_str="",
                ),
            ]
            _ = planner._validate_and_construct_plan(steps=example_input)

        with self.subTest("should fail"):
            example_input = [
                ParsedStep(
                    output_var="start_date",
                    function="get_date_from_date_str",
                    arguments={"time_str": '"1 month ago"'},
                    description='Convert "1 month ago" to a date to use as the start date for news search',
                    original_str="",
                ),
                ParsedStep(
                    output_var="stock_ids_1",
                    function="stock_identifier_lookup_multi",
                    arguments={"stock_names": '["Meta", "Apple,1", "Microsoft"]'},
                    description="Convert company names to stock identifiers",
                    original_str="",
                ),
                ParsedStep(
                    output_var="combined",
                    function="add_lists",
                    arguments={"list1": "stock_ids_1", "list2": '["Meta", "Apple,1", "Microsoft"]'},
                    description="Merge lists",
                    original_str="",
                ),
                ParsedStep(
                    output_var="news_developments",
                    function="get_news_developments_about_companies",
                    arguments={"stock_ids": "combined", "start_date": "start_date"},
                    description="Get news developments for Meta, Apple, and Microsoft since last month",
                    original_str="",
                ),
                ParsedStep(
                    output_var="result",
                    function="prepare_output",
                    arguments={"object_to_output": "combined", "title": '"test"'},
                    description="Output the result",
                    original_str="",
                ),
            ]
            with self.assertRaises(ExecutionPlanParsingError):
                _ = planner._validate_and_construct_plan(steps=example_input)

    def test_validate_construct_plan(self):
        planner = Planner(agent_id="TEST", tool_registry=get_test_registry())
        example_input = [
            ParsedStep(
                original_str="",
                output_var="start_date",
                function="get_date_from_date_str",
                arguments={"time_str": '"1 month ago"'},
                description='Convert "1 month ago" to a date to use as the start date for news search',
            ),
            ParsedStep(
                original_str="",
                output_var="stock_ids",
                function="stock_identifier_lookup_multi",
                arguments={"stock_names": '["Meta", "Apple,1", "Microsoft"]'},
                description="Convert company names to stock identifiers",
            ),
            ParsedStep(
                original_str="",
                output_var="news_developments",
                function="get_news_developments_about_companies",
                arguments={"stock_ids": "stock_ids", "start_date": "start_date"},
                description="Get news developments for Meta, Apple, and Microsoft since last month",
            ),
            ParsedStep(
                original_str="",
                output_var="collapsed_news_developments",
                function="collapse_lists",
                arguments={"lists_of_lists": "news_developments"},
                description="Collapse the list of lists of news developments into a single list",
            ),
            ParsedStep(
                original_str="",
                output_var="filtered_news",
                function="filter_texts_by_topic",
                arguments={"topic": '"machine learning"', "texts": "collapsed_news_developments"},
                description="Filter news descriptions to only those related to machine learning",
            ),
            ParsedStep(
                original_str="",
                output_var="summary",
                function="summarize_texts",
                arguments={"texts": "filtered_news"},
                description="Summarize the news descriptions into a single summary text",
            ),
            ParsedStep(
                original_str="",
                output_var="unused",
                function="get_news_developments_about_companies",
                arguments={"stock_ids": "[stock_ids[0]]", "start_date": "start_date"},
                description="Get news developments for only the first stock",
            ),
            ParsedStep(
                original_str="",
                output_var="unused2",
                function="get_name_of_single_stock",
                arguments={"stock_id": "stock_ids[0]"},
                description="Gets the name of the stock indicated by the stock_id",
            ),
            ParsedStep(
                original_str="",
                output_var="result",
                function="prepare_output",
                arguments={"object_to_output": "summary", "title": '"test"'},
                description="Output the result",
            ),
        ]
        execution_plan = planner._validate_and_construct_plan(example_input)

        expected_output = [
            ToolExecutionNode(
                tool_name="get_date_from_date_str",
                args={"time_str": "1 month ago"},
                description='Convert "1 month ago" to a date to use as the start date for news search',
                output_variable_name="start_date",
                is_output_node=False,
            ),
            ToolExecutionNode(
                tool_name="stock_identifier_lookup_multi",
                args={"stock_names": ["Meta", "Apple,1", "Microsoft"]},
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
                output_variable_name="collapsed_news_developments",
                is_output_node=False,
            ),
            ToolExecutionNode(
                tool_name="filter_texts_by_topic",
                args={
                    "topic": "machine learning",
                    "texts": Variable(var_name="collapsed_news_developments"),
                },
                description="Filter news descriptions to only those related to machine learning",
                output_variable_name="filtered_news",
                is_output_node=False,
            ),
            ToolExecutionNode(
                tool_name="summarize_texts",
                args={"texts": Variable(var_name="filtered_news")},
                description="Summarize the news descriptions into a single summary text",
                output_variable_name="summary",
                is_output_node=False,
            ),
            ToolExecutionNode(
                tool_name="get_news_developments_about_companies",
                args={
                    "stock_ids": [Variable(var_name="stock_ids", index=0)],
                    "start_date": Variable(var_name="start_date"),
                },
                description="Get news developments for only the first stock",
                output_variable_name="unused",
                is_output_node=False,
            ),
            ToolExecutionNode(
                tool_name="get_name_of_single_stock",
                args={
                    "stock_id": Variable(var_name="stock_ids", index=0),
                },
                description="Gets the name of the stock indicated by the stock_id",
                output_variable_name="unused2",
                is_output_node=False,
            ),
            ToolExecutionNode(
                tool_name="prepare_output",
                args={"object_to_output": Variable(var_name="summary"), "title": "test"},
                description="Output the result",
                output_variable_name="result",
                is_output_node=True,
            ),
        ]

        for in_node, out_node in zip(execution_plan.nodes, expected_output):
            self.assertEqual(in_node.tool_name, out_node.tool_name)
            self.assertEqual(in_node.args, out_node.args)
            self.assertEqual(in_node.is_output_node, out_node.is_output_node)
            self.assertEqual(in_node.output_variable_name, out_node.output_variable_name)


class TestPlannerTypes(TestCase):
    def test_accumulate_type_from_list(self):
        var_type_lookup = {"test_int": int, "test_int_str": Union[str, int]}
        cases = [
            ([1, 2], List[int]),
            ([1, 2, Variable(var_name="test_int")], List[int]),
            ([Variable(var_name="test_int")], List[int]),
            ([True, Variable(var_name="test_int_str")], List[Union[bool, str, int]]),
            (
                [[1, "test"], Variable(var_name="test_int_str")],
                List[Union[List[Union[int, str]], str, int]],
            ),
        ]
        for list_val, expected in cases:
            with self.subTest(str(list_val)):
                self.assertIs(
                    accumulate_type_from_list(list_val=list_val, var_type_lookup=var_type_lookup),
                    expected,
                )

    def test_get_node_dependency_map(self):
        # define nodes
        test1 = ToolExecutionNode(
            tool_name="test_1",
            args={"name": "constant"},
            description="",
            output_variable_name="test_1_out",
        )
        test2 = ToolExecutionNode(
            tool_name="test_2",
            args={"name": "constant", "name2": Variable(var_name="test_1_out")},
            description="",
            output_variable_name="test_2_out",
        )
        test3 = ToolExecutionNode(
            tool_name="test_3",
            args={
                "name": "constant",
                "name2": Variable(var_name="test_1_out"),
                "name3": Variable(var_name="test_2_out"),
            },
            description="",
            output_variable_name="test_3_out",
        )
        test4 = ToolExecutionNode(
            tool_name="test_4",
            args={
                "name": Variable(var_name="test_1_out"),
            },
            description="",
            output_variable_name="test_4_out",
        )
        test5 = ToolExecutionNode(
            tool_name="test_5",
            args={
                "name": "constant",
                "name2": Variable(var_name="test_4_out"),
                "name3": Variable(var_name="test_3_out"),
            },
            description="",
            output_variable_name="test_5_out",
        )

        test_plan = ExecutionPlan(nodes=[test1, test2, test3, test4, test5])
        expected_tree = {
            test1: {test2, test3, test4},
            test2: {test3},
            test3: {test5},
            test4: {test5},
            test5: set(),
        }
        actual = test_plan.get_node_dependency_map()
        self.assertEqual(expected_tree, actual)

    def test_get_pruned_plan(self):
        # define nodes
        test1 = ToolExecutionNode(
            tool_name="test_1",
            args={"name": "constant"},
            description="",
            output_variable_name="test_1_out",
            tool_task_id="test_1",
        )
        test2 = ToolExecutionNode(
            tool_name="test_2",
            args={"name": "constant", "name2": Variable(var_name="test_1_out")},
            description="",
            output_variable_name="test_2_out",
        )
        test3 = ToolExecutionNode(
            tool_name="test_3",
            args={
                "name": "constant",
                "name2": Variable(var_name="test_1_out"),
                "name3": Variable(var_name="test_2_out"),
            },
            description="",
            output_variable_name="test_3_out",
        )
        test4 = ToolExecutionNode(
            tool_name="test_4",
            args={
                "name": Variable(var_name="test_1_out"),
            },
            description="",
            output_variable_name="test_4_out",
        )
        test5 = ToolExecutionNode(
            tool_name="test_5",
            args={
                "name": "constant",
                "name2": Variable(var_name="test_4_out"),
                "name3": Variable(var_name="test_3_out"),
            },
            description="",
            output_variable_name="test_5_out",
        )
        test_out_1 = ToolExecutionNode(
            tool_name="prepare_output",
            args={"name": Variable(var_name="test_2_out")},
            description="",
            is_output_node=True,
            tool_task_id="output1",
        )
        test_out_2 = ToolExecutionNode(
            tool_name="prepare_output",
            args={"name": Variable(var_name="test_5_out")},
            description="",
            is_output_node=True,
            tool_task_id="output2",
        )

        test_plan = ExecutionPlan(nodes=[test1, test2, test3, test4, test5, test_out_1, test_out_2])
        # First, remove output 1. Since another node depends on "test_2_out", it should have no effect
        expected_plan = ExecutionPlan(nodes=[test1, test2, test3, test4, test5, test_out_2])
        pruned = test_plan.get_pruned_plan(task_ids_to_remove={"output1"})
        with self.subTest(msg="No change expected"):
            self.assertEqual(pruned, expected_plan)

        # Now, remove output 2. Nodes should be removed in this case
        expected_plan = ExecutionPlan(nodes=[test1, test2, test_out_1])
        pruned = test_plan.get_pruned_plan(task_ids_to_remove={"output2"})
        with self.subTest(msg="Pruning expected"):
            self.assertEqual(pruned, expected_plan)

        # Now, remove both outputs. Plan should now be empty.
        expected_plan = ExecutionPlan(nodes=[])
        pruned = test_plan.get_pruned_plan(task_ids_to_remove={"output1", "output2"})
        with self.subTest(msg="All nodes removed"):
            self.assertEqual(pruned, expected_plan)

        # Now, remove nothing. Plan should be unchanged.
        expected_plan = ExecutionPlan(
            nodes=[test1, test2, test3, test4, test5, test_out_1, test_out_2]
        )
        pruned = test_plan.get_pruned_plan(task_ids_to_remove=set())
        with self.subTest(msg="No nodes removed"):
            self.assertEqual(pruned, expected_plan)

        with self.subTest(msg="Expect error removing non-output"):
            with self.assertRaises(RuntimeError):
                pruned = test_plan.get_pruned_plan(task_ids_to_remove={"test_1"})

    def test_reorder_plan_with_output_task_ordering(self):
        # define nodes
        test1 = ToolExecutionNode(
            tool_name="test_1",
            args={"name": "constant"},
            description="",
            output_variable_name="test_1_out",
            tool_task_id="test_1",
        )
        test2 = ToolExecutionNode(
            tool_name="test_2",
            args={"name": "constant", "name2": Variable(var_name="test_1_out")},
            description="",
            output_variable_name="test_2_out",
        )
        test3 = ToolExecutionNode(
            tool_name="test_3",
            args={
                "name": "constant",
                "name2": Variable(var_name="test_1_out"),
                "name3": Variable(var_name="test_2_out"),
            },
            description="",
            output_variable_name="test_3_out",
        )
        test4 = ToolExecutionNode(
            tool_name="test_4",
            args={
                "name": Variable(var_name="test_1_out"),
            },
            description="",
            output_variable_name="test_4_out",
        )
        test5 = ToolExecutionNode(
            tool_name="test_5",
            args={
                "name": "constant",
                "name2": Variable(var_name="test_4_out"),
                "name3": Variable(var_name="test_3_out"),
            },
            description="",
            output_variable_name="test_5_out",
        )
        test_out_1 = ToolExecutionNode(
            tool_name="prepare_output",
            args={"name": Variable(var_name="test_2_out")},
            description="",
            is_output_node=True,
            tool_task_id="output1",
        )
        test_out_2 = ToolExecutionNode(
            tool_name="prepare_output",
            args={"name": Variable(var_name="test_5_out")},
            description="",
            is_output_node=True,
            tool_task_id="output2",
        )
        test_out_3 = ToolExecutionNode(
            tool_name="prepare_output",
            args={"name": Variable(var_name="test_1_out")},
            description="",
            is_output_node=True,
            tool_task_id="output3",
        )

        test_plan = ExecutionPlan(nodes=[test1, test2, test_out_1, test3, test4, test5, test_out_2])
        # First, swap the ordering of outputs 1 and 2
        expected_plan = ExecutionPlan(
            nodes=[test1, test2, test3, test4, test5, test_out_2, test_out_1]
        )
        reordered = test_plan.reorder_plan_with_output_task_ordering([test_out_2, test_out_1])
        with self.subTest(msg="First reordering"):
            self.assertEqual(reordered, expected_plan)

        test_plan = ExecutionPlan(
            nodes=[test1, test2, test_out_1, test3, test4, test5, test_out_2, test_out_3]
        )
        expected_plan = ExecutionPlan(
            nodes=[test1, test_out_3, test2, test3, test4, test5, test_out_2, test_out_1]
        )
        reordered = test_plan.reorder_plan_with_output_task_ordering(
            [test_out_3, test_out_2, test_out_1]
        )
        with self.subTest(msg="Second reordering"):
            self.assertEqual(reordered, expected_plan)

        test_plan = ExecutionPlan(nodes=[test1, test2, test_out_1, test3, test4, test5, test_out_2])
        with self.subTest(msg="Expect error reordering with non-output"):
            with self.assertRaises(RuntimeError):
                _ = test_plan.reorder_plan_with_output_task_ordering([test_out_1, test2])

    def test_build_dependency_graph(self):
        # Copy from plan_id = `def785e4-c62a-41cb-9d8e-a29dbbca9942`
        plan = ExecutionPlan.model_validate(
            {
                "nodes": [
                    {
                        "args": {"stock_name": "NVDA"},
                        "tool_name": "stock_identifier_lookup",
                        "description": "Look up NVDA identifier",
                        "store_output": True,
                        "tool_task_id": "87674424-d5db-4cb1-9adc-448b6dcb0acd",
                        "is_output_node": False,
                        "output_variable_name": "nvda_id",
                    },
                    {
                        "args": {"universe_name": "S&P 500"},
                        "tool_name": "get_stock_universe",
                        "description": "Get stocks from S&P 500",
                        "store_output": True,
                        "tool_task_id": "1cbb27a1-5baf-45ca-a9d6-0cebe70b809c",
                        "is_output_node": False,
                        "output_variable_name": "sp500_stocks",
                    },
                    {
                        "args": {"stock_ids": {"index": None, "var_name": "sp500_stocks"}},
                        "tool_name": "get_company_descriptions",
                        "description": "Get company descriptions for stocks in the S&P 500",
                        "store_output": True,
                        "tool_task_id": "ae7910e6-91be-460a-8a1a-29c97dbd4d8e",
                        "is_output_node": False,
                        "output_variable_name": "company_descriptions",
                    },
                    {
                        "args": {
                            "texts": {"index": None, "var_name": "company_descriptions"},
                            "stock_ids": {"index": None, "var_name": "sp500_stocks"},
                            "filter_only": True,
                            "max_results": 10,
                            "product_str": "AI chips",
                            "must_include_stocks": [{"index": None, "var_name": "nvda_id"}],
                        },
                        "tool_name": "filter_stocks_by_product_or_service",
                        "description": "Filter stocks to those in the AI chips market, including NVDA, with a maximum of 10 results",
                        "store_output": True,
                        "tool_task_id": "b7985bdd-aa42-49b4-8fd6-e0d59d386931",
                        "is_output_node": False,
                        "output_variable_name": "ai_chip_stocks",
                    },
                    {
                        "args": {"stock_ids": {"index": None, "var_name": "ai_chip_stocks"}},
                        "tool_name": "get_default_text_data_for_stocks",
                        "description": "Get default text data for all stocks in the AI chips market",
                        "store_output": False,
                        "tool_task_id": "f06deb86-8b11-4ea6-98b1-9558058bcd73",
                        "is_output_node": False,
                        "output_variable_name": "ai_chip_texts",
                    },
                    {
                        "args": {
                            "market": "AI chips",
                            "target_stock": {"index": None, "var_name": "nvda_id"},
                        },
                        "tool_name": "get_criteria_for_competitive_analysis",
                        "description": "Generate criteria for a competitive analysis of the AI chips market with NVDA as the target stock",
                        "store_output": True,
                        "tool_task_id": "29805fdb-b3b7-44a3-a99c-149f871377c0",
                        "is_output_node": False,
                        "output_variable_name": "criteria",
                    },
                    {
                        "args": {
                            "prompt": "Is NVDA the leader in AI chips space",
                            "stocks": {"index": None, "var_name": "ai_chip_stocks"},
                            "criteria": {"index": None, "var_name": "criteria"},
                            "target_stock": {"index": None, "var_name": "nvda_id"},
                            "all_text_data": {"index": None, "var_name": "ai_chip_texts"},
                        },
                        "tool_name": "do_competitive_analysis",
                        "description": "Perform a competitive analysis to evaluate if NVDA is the leader in the AI chips space based on the criteria",
                        "store_output": True,
                        "tool_task_id": "11287882-3e8e-4763-a37e-f3ab5c41268d",
                        "is_output_node": False,
                        "output_variable_name": "competitive_analysis",
                    },
                    {
                        "args": {
                            "prompt": "Is NVDA the leader in AI chips space",
                            "competitive_analysis": {
                                "index": None,
                                "var_name": "competitive_analysis",
                            },
                        },
                        "tool_name": "generate_summary_for_competitive_analysis",
                        "description": "Generate a text summary of NVDA's position in the AI chips market based on the competitive analysis",
                        "store_output": True,
                        "tool_task_id": "cb2f690b-67ec-4ab1-a31a-6b052a319fae",
                        "is_output_node": False,
                        "output_variable_name": "summary_text",
                    },
                    {
                        "args": {
                            "title": "Summary of NVDA's Position in AI Chips Market",
                            "object_to_output": {"index": None, "var_name": "summary_text"},
                        },
                        "tool_name": "prepare_output",
                        "description": "Output the summary of NVDA's position in the AI chips market",
                        "store_output": False,
                        "tool_task_id": "134a7e56-788f-4cb4-ba84-680b313f18c0",
                        "is_output_node": True,
                        "output_variable_name": "output1",
                    },
                    {
                        "args": {
                            "title": "Detailed Competitive Analysis of AI Chips Market",
                            "object_to_output": {"index": None, "var_name": "competitive_analysis"},
                        },
                        "tool_name": "prepare_output",
                        "description": "Output the detailed competitive analysis of the AI chips market",
                        "store_output": False,
                        "tool_task_id": "9bb1b14b-0d58-4412-bcab-f486a3820dc6",
                        "is_output_node": True,
                        "output_variable_name": "output2",
                    },
                    {
                        "args": {
                            "title": "Criteria Used for Competitive Analysis",
                            "object_to_output": {"index": None, "var_name": "criteria"},
                        },
                        "tool_name": "prepare_output",
                        "description": "Output the criteria used for the competitive analysis",
                        "store_output": False,
                        "tool_task_id": "111fdfca-9b83-49e0-a00a-6f8e65c5a9f6",
                        "is_output_node": True,
                        "output_variable_name": "output3",
                    },
                ],
                "locked_task_ids": [],
                "deleted_task_ids": [],
            }
        )

        parent_to_children, child_to_parents, node_id_to_indegree, _ = plan.build_dependency_graph()

        # Assert Parent - Children Relationship
        node0 = plan.nodes[0]  # `stock_identifier_lookup`
        self.assertNotIn(node0.tool_task_id, child_to_parents)
        self.assertEqual(node_id_to_indegree[node0.tool_task_id], 0)
        self.assertSetEqual(
            {node.tool_name for node in parent_to_children[node0.tool_task_id]},
            {
                "filter_stocks_by_product_or_service",
                "get_criteria_for_competitive_analysis",
                "do_competitive_analysis",
            },
        )

        node1 = plan.nodes[1]  # `get_stock_universe`
        self.assertNotIn(node1.tool_task_id, child_to_parents)
        self.assertEqual(node_id_to_indegree[node1.tool_task_id], 0)
        self.assertSetEqual(
            {node.tool_name for node in parent_to_children[node1.tool_task_id]},
            {"get_company_descriptions", "filter_stocks_by_product_or_service"},
        )

        node2 = plan.nodes[2]  # `get_company_descriptions`
        self.assertEqual(node_id_to_indegree[node2.tool_task_id], 1)
        self.assertSetEqual(
            {node.tool_name for node in child_to_parents[node2.tool_task_id]},
            {"get_stock_universe"},
        )
        self.assertSetEqual(
            {node.tool_name for node in parent_to_children[node2.tool_task_id]},
            {"filter_stocks_by_product_or_service"},
        )

        node3 = plan.nodes[3]  # `filter_stocks_by_product_or_service`
        self.assertEqual(node_id_to_indegree[node3.tool_task_id], 3)
        self.assertSetEqual(
            {node.tool_name for node in child_to_parents[node3.tool_task_id]},
            {
                "get_company_descriptions",
                "get_stock_universe",
                "stock_identifier_lookup",
            },
        )
        self.assertSetEqual(
            {node.tool_name for node in parent_to_children[node3.tool_task_id]},
            {"get_default_text_data_for_stocks", "do_competitive_analysis"},
        )

        node4 = plan.nodes[4]  # `get_default_text_data_for_stocks`
        self.assertEqual(node_id_to_indegree[node4.tool_task_id], 1)
        self.assertSetEqual(
            {node.tool_name for node in child_to_parents[node4.tool_task_id]},
            {"filter_stocks_by_product_or_service"},
        )
        self.assertSetEqual(
            {node.tool_name for node in parent_to_children[node4.tool_task_id]},
            {"do_competitive_analysis"},
        )

        node5 = plan.nodes[5]  # `get_criteria_for_competitive_analysis`
        self.assertEqual(node_id_to_indegree[node5.tool_task_id], 1)
        self.assertSetEqual(
            {node.tool_name for node in child_to_parents[node5.tool_task_id]},
            {"stock_identifier_lookup"},
        )
        self.assertSetEqual(
            {node.tool_name for node in parent_to_children[node5.tool_task_id]},
            {"do_competitive_analysis", "prepare_output"},
        )

        node6 = plan.nodes[6]  # `do_competitive_analysis`
        self.assertEqual(node_id_to_indegree[node6.tool_task_id], 4)
        self.assertSetEqual(
            {node.tool_name for node in child_to_parents[node6.tool_task_id]},
            {
                "filter_stocks_by_product_or_service",
                "get_criteria_for_competitive_analysis",
                "stock_identifier_lookup",
                "get_default_text_data_for_stocks",
            },
        )
        self.assertSetEqual(
            {node.tool_name for node in parent_to_children[node6.tool_task_id]},
            {"generate_summary_for_competitive_analysis", "prepare_output"},
        )

        node7 = plan.nodes[7]  # `generate_summary_for_competitive_analysis`
        self.assertEqual(node_id_to_indegree[node7.tool_task_id], 1)
        self.assertSetEqual(
            {node.tool_name for node in child_to_parents[node7.tool_task_id]},
            {"do_competitive_analysis"},
        )
        self.assertSetEqual(
            {node.tool_name for node in parent_to_children[node7.tool_task_id]},
            {"prepare_output"},
        )

        node8 = plan.nodes[8]  # `prepare_output`
        self.assertEqual(node_id_to_indegree[node8.tool_task_id], 1)
        self.assertSetEqual(
            {node.tool_name for node in child_to_parents[node8.tool_task_id]},
            {"generate_summary_for_competitive_analysis"},
        )
        self.assertSetEqual(
            {node.tool_name for node in parent_to_children[node8.tool_task_id]},
            {"prepare_output"},
        )
        self.assertSetEqual(
            {node.tool_task_id for node in parent_to_children[node8.tool_task_id]},
            {"9bb1b14b-0d58-4412-bcab-f486a3820dc6"},
        )

        node9 = plan.nodes[9]  # `prepare_output`
        self.assertEqual(node_id_to_indegree[node9.tool_task_id], 2)
        self.assertSetEqual(
            {node.tool_name for node in child_to_parents[node9.tool_task_id]},
            {"do_competitive_analysis", "prepare_output"},
        )
        self.assertSetEqual(
            {node.tool_task_id for node in parent_to_children[node9.tool_task_id]},
            {"111fdfca-9b83-49e0-a00a-6f8e65c5a9f6"},
        )

        node10 = plan.nodes[10]  # `prepare_output`
        self.assertEqual(node_id_to_indegree[node10.tool_task_id], 2)
        self.assertSetEqual(
            {node.tool_name for node in child_to_parents[node10.tool_task_id]},
            {"get_criteria_for_competitive_analysis", "prepare_output"},
        )
        self.assertNotIn(node10.tool_task_id, parent_to_children)
