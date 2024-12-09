import unittest
import warnings
from unittest.mock import patch

from agent_service.endpoints.models import AgentInfo
from agent_service.GPT.requests import set_use_global_stub
from agent_service.planner.executor import run_execution_plan_local
from agent_service.planner.planner_types import (
    ExecutionPlan,
    ToolExecutionNode,
    Variable,
)
from agent_service.types import PlanRunContext
from agent_service.utils.date_utils import get_now_utc
from agent_service.utils.logs import init_stdout_logging
from agent_service.utils.postgres import DEFAULT_AGENT_NAME, get_psql
from tests.planner_utils import get_test_registry


class TestExecutor(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        # Start patching
        self.patcher = patch(
            "agent_service.planner.executor.default_tool_registry", get_test_registry
        )
        self.mock_function = self.patcher.start()

    def tearDown(self):
        # Stop patching
        self.patcher.stop()

    @classmethod
    def setUpClass(cls) -> None:
        set_use_global_stub(False)
        warnings.filterwarnings(
            "ignore", category=ResourceWarning, message="unclosed.*<ssl.SSLSocket.*>"
        )
        warnings.filterwarnings(
            "ignore", category=ResourceWarning, message="unclosed.*<socket.socket.*>"
        )

        init_stdout_logging()

    async def test_executor(self):
        plan = ExecutionPlan(
            nodes=[
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
                    output_variable_name="filtered_texts",
                    is_output_node=False,
                ),
                ToolExecutionNode(
                    tool_name="summarize_texts",
                    args={"texts": Variable(var_name="filtered_texts")},
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
                    args={"object_to_output": [Variable(var_name="summary")], "title": "test"},
                    description="Output the result",
                    output_variable_name="result",
                    is_output_node=True,
                ),
            ]
        )
        plan_run_context = PlanRunContext.get_dummy()
        agent = AgentInfo(
            agent_id=plan_run_context.agent_id,
            user_id=plan_run_context.user_id,
            agent_name=DEFAULT_AGENT_NAME,
            created_at=get_now_utc(),
            last_updated=get_now_utc(),
            deleted=False,
        )
        with self.subTest("Old executor"):
            db = get_psql(skip_commit=True)
            db.insert_agent(agent)
            db.write_execution_plan(
                plan_id=plan_run_context.plan_id, agent_id=plan_run_context.agent_id, plan=plan
            )
            result, _ = await run_execution_plan_local(
                plan,
                plan_run_context,
                do_chat=False,
            )
            self.assertIsNotNone(result)
            db.close()
        with self.subTest("New executor"):
            db = get_psql(skip_commit=True)
            db.insert_agent(agent)
            db.write_execution_plan(
                plan_id=plan_run_context.plan_id, agent_id=plan_run_context.agent_id, plan=plan
            )
            result, _ = await run_execution_plan_local(
                plan,
                plan_run_context,
                do_chat=False,
                use_new_executor_impl=True,
            )
            self.assertIsNotNone(result)
