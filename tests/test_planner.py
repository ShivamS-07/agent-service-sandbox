import unittest
import warnings
from typing import Any, List, Union
from unittest import IsolatedAsyncioTestCase, TestCase

from agent_service.GPT.requests import set_use_global_stub
from agent_service.planner.planner import Planner
from agent_service.planner.planner_types import (
    ExecutionPlan,
    ParsedStep,
    ToolExecutionNode,
    Variable,
)
from agent_service.types import ChatContext, Message
from agent_service.utils.logs import init_stdout_logging
from tests.planner_utils import get_test_registry


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
                output_var="start_date",
                function="get_date_from_date_str",
                arguments={"time_str": '"1 month ago"'},
                description='Convert "1 month ago" to a date to use as the start date for news search',
            ),
            ParsedStep(
                output_var="stock_ids",
                function="stock_identifier_lookup_multi",
                # Add a parenthesis to test the regex
                arguments={"stock_names": '["Meta", "Apple", "Microsoft)"]'},
                description="Look up stock identifiers for Meta, Apple, and Microsoft",
            ),
            ParsedStep(
                output_var="news_developments",
                function="get_news_developments_about_companies",
                arguments={"stock_ids": "stock_ids", "start_date": "start_date"},
                description="Get news developments in the last month for Meta, Apple, and Microsoft",
            ),
            ParsedStep(
                output_var="collapsed_news_developments",
                function="collapse_lists",
                arguments={"lists_of_lists": "news_developments"},
                description="Collapse the list of lists of news development IDs into a single list",
            ),
            ParsedStep(
                output_var="filtered_news",
                function="filter_texts_by_topic",
                arguments={"topic": '"machine learning"', "texts": "collapsed_news_developments"},
                description="Filter news descriptions to only those related to machine learning",
            ),
            ParsedStep(
                output_var="summary",
                function="summarize_texts",
                arguments={"texts": "filtered_news"},
                description="Summarize the machine learning related news into a single text",
            ),
        ]
        self.assertEqual(steps, expected_output)

    def test_validate_construct_plan(self):
        planner = Planner(agent_id="TEST", tool_registry=get_test_registry())
        example_input = [
            ParsedStep(
                output_var="start_date",
                function="get_date_from_date_str",
                arguments={"time_str": '"1 month ago"'},
                description='Convert "1 month ago" to a date to use as the start date for news search',
            ),
            ParsedStep(
                output_var="stock_ids",
                function="stock_identifier_lookup_multi",
                arguments={"stock_names": '["Meta", "Apple,1", "Microsoft"]'},
                description="Convert company names to stock identifiers",
            ),
            ParsedStep(
                output_var="news_developments",
                function="get_news_developments_about_companies",
                arguments={"stock_ids": "stock_ids", "start_date": "start_date"},
                description="Get news developments for Meta, Apple, and Microsoft since last month",
            ),
            ParsedStep(
                output_var="collapsed_news_developments",
                function="collapse_lists",
                arguments={"lists_of_lists": "news_developments"},
                description="Collapse the list of lists of news developments into a single list",
            ),
            ParsedStep(
                output_var="filtered_news",
                function="filter_texts_by_topic",
                arguments={"topic": '"machine learning"', "texts": "collapsed_news_developments"},
                description="Filter news descriptions to only those related to machine learning",
            ),
            ParsedStep(
                output_var="summary",
                function="summarize_texts",
                arguments={"texts": "filtered_news"},
                description="Summarize the news descriptions into a single summary text",
            ),
            ParsedStep(
                output_var="unused",
                function="get_news_developments_about_companies",
                arguments={"stock_ids": "[stock_ids[0]]", "start_date": "start_date"},
                description="Get news developments for only the first stock",
            ),
            ParsedStep(
                output_var="unused2",
                function="get_name_of_single_stock",
                arguments={"stock_id": "stock_ids[0]"},
                description="Gets the name of the stock indicated by the stock_id",
            ),
            ParsedStep(
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
