# type: ignore
import asyncio
import datetime
import json
import traceback
import unittest
import uuid
import warnings
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, Optional, Tuple
from uuid import uuid4

from gbi_common_py_utils.utils.clickhouse_base import ClickhouseBase
from gbi_common_py_utils.utils.environment import DEV_TAG
from gbi_common_py_utils.utils.event_logging import log_event

from agent_service.endpoints.models import AgentMetadata
from agent_service.GPT.constants import DEFAULT_CHEAP_MODEL
from agent_service.GPT.requests import GPT
from agent_service.io_type_utils import IOType, TableColumnType, dump_io_type
from agent_service.planner.executor import (
    create_execution_plan_local,
    run_execution_plan_local,
)
from agent_service.planner.planner import plan_to_json
from agent_service.planner.planner_types import ExecutionPlan
from agent_service.types import ChatContext, Message, PlanRunContext
from agent_service.utils.date_utils import get_now_utc
from agent_service.utils.output_utils.output_construction import PreparedOutput
from agent_service.utils.postgres import DEFAULT_AGENT_NAME, get_psql
from regression_test.util import (
    validate_and_compare_text,
    validate_line_graph,
    validate_table_and_get_columns,
)

CH = ClickhouseBase(environment=DEV_TAG)
SERVICE_VERSION = "374053208103.dkr.ecr.us-west-2.amazonaws.com/agent-service:afabe4a61d2205b91d22c64218ee1579dec317e2"


class PlanGenerationError(Exception):
    pass


@dataclass
class EventLog:
    event_name: str
    event_data: Optional[Dict[str, Any]] = None


def get_expected_output(prompt: str) -> Tuple[str, str]:
    res = CH.generic_read(
        "select execution_plan from agent.regression_test where prompt = %(prompt)s and "
        "service_version = %(service_version)s",
        {"prompt": prompt, "service_version": SERVICE_VERSION},
    )
    if not res:
        return None
    return res[0]["execution_plan"]


def plan_to_simple_json(plan: ExecutionPlan) -> str:
    json_list = []
    for node in plan.nodes:
        json_list.append({"tool_name": node.tool_name})
    return json.dumps(json_list)


def compare_plan(actual: ExecutionPlan, expected: ExecutionPlan):
    warn_msg = f"""
    actual plan is {plan_to_simple_json(actual)} and
    expected plan is {plan_to_simple_json(expected)}
    """
    if len(actual.nodes) == len(expected.nodes):
        for i, node in enumerate(actual.nodes):
            if node.tool_name != expected.nodes[i].tool_name:
                warnings.warn(
                    f"""
                    The tool name {node.tool_name} at step {i} does not match with {expected.nodes[i].tool_name}
                    {warn_msg}
                    """
                )
    else:
        warnings.warn(
            f"""
            The length of plans is not same.
            Actual has length {len(actual.nodes)} and expected length is {len(expected.nodes)}.
            {warn_msg}
            """
        )


def get_output(output: IOType) -> IOType:
    if isinstance(output, PreparedOutput):
        return output.val
    if isinstance(output, list) and isinstance(output[0], PreparedOutput):
        return get_output(output[0])
    return output


def validate_plan(prompt: str, plan: Optional[ExecutionPlan]) -> None:
    expected_plan_json = get_expected_output(prompt=prompt)
    if expected_plan_json:
        expected_plan = ExecutionPlan(**({"nodes": json.loads(expected_plan_json)}))
        compare_plan(plan, expected_plan)
    else:
        warnings.warn("Could not find expected plan")


class TestExecutionPlanner(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.loop = asyncio.get_event_loop()  # type: ignore[assignment]
        cls.test_suite_id = str(uuid.uuid4())
        cls.llm = GPT(model=DEFAULT_CHEAP_MODEL)

    def prompt_test(self, prompt: str, validate_plan: Callable, validate_output: Callable):
        user_id = str(uuid.uuid4())
        shared_log_data = {
            "user_id": user_id,
            "test_suite_id": self.test_suite_id,
            "prompt": prompt,
        }
        plan_generated_log = EventLog(
            event_name="agent-regression-plan-generated",
            event_data={
                **shared_log_data,
            },
        )
        output_generated_log = EventLog(
            event_name="agent-regression-output-generated",
            event_data={
                **shared_log_data,
            },
        )
        try:
            plan, output = self.run_regression(
                prompt=prompt,
                plan_generated_log=plan_generated_log,
                output_generated_log=output_generated_log,
                user_id=user_id,
            )

            validate_plan(prompt=prompt, plan=plan)
            validate_output(prompt=prompt, output=output)
            log_event(**asdict(plan_generated_log))
            log_event(**asdict(output_generated_log))
        except PlanGenerationError as e:
            plan_generated_log.event_data["error_msg"] = traceback.format_exc()
            log_event(**asdict(plan_generated_log))
            raise e
        except Exception as e:
            output_generated_log.event_data["error_msg"] = traceback.format_exc()
            log_event(**asdict(plan_generated_log))
            log_event(**asdict(output_generated_log))
            raise e

    def run_regression(
        self,
        prompt: str,
        plan_generated_log: EventLog,
        output_generated_log: EventLog,
        user_id: str,
        do_chat: bool = False,
    ):
        agent_id = str(uuid.uuid4())
        agent = AgentMetadata(
            agent_id=agent_id,
            user_id=user_id,
            agent_name=DEFAULT_AGENT_NAME,
            created_at=get_now_utc(),
            last_updated=get_now_utc(),
        )
        user_msg = Message(
            agent_id=agent.agent_id,
            message=prompt,
            is_user_message=True,
            message_time=get_now_utc(),
        )

        db = get_psql(skip_commit=True)
        db.insert_agent(agent)
        db.insert_chat_messages([user_msg])
        plan_id = str(uuid4())
        chat = ChatContext(messages=[Message(message=prompt, is_user_message=True)])
        execution_plan_start = datetime.datetime.utcnow().isoformat()
        try:
            plan = self.loop.run_until_complete(
                create_execution_plan_local(
                    agent_id=agent_id,
                    plan_id=plan_id,
                    user_id=user_id,
                    skip_db_commit=True,
                    skip_task_cache=True,
                    run_plan_in_prefect_immediately=False,
                    run_tasks_without_prefect=True,
                    chat_context=chat,
                    do_chat=do_chat,
                )
            )
            execution_plan_finished_at = datetime.datetime.utcnow().isoformat()
            plan_generated_log.event_data.update(
                {
                    "started_at_utc": execution_plan_start,
                    "finished_at_utc": execution_plan_finished_at,
                    "execution_plan": plan_to_json(plan=plan),
                    "plan_id": plan_id,
                }
            )
        except Exception as e:
            raise PlanGenerationError(e)
        context = PlanRunContext(
            agent_id=agent_id,
            plan_id=plan_id,
            user_id=user_id,
            plan_run_id=str(uuid4()),
            skip_db_commit=True,
            skip_task_cache=True,
            run_tasks_without_prefect=True,
        )
        context.chat = chat
        execution_started_at = datetime.datetime.utcnow().isoformat()
        output = self.loop.run_until_complete(
            run_execution_plan_local(
                plan=plan,
                context=context,
                do_chat=do_chat,
                log_all_outputs=False,
                replan_execution_error=False,
            )
        )
        output_generated_log.event_data.update(
            {
                "execution_plan_started_at_utc": execution_plan_start,
                "execution_plan_finished_at_utc": execution_plan_finished_at,
                "execution_start_at_utc": execution_started_at,
                "execution_finished_at_utc": datetime.datetime.utcnow().isoformat(),
                "execution_plan": plan_to_json(plan=plan),
                "execution_plan_simple": plan_to_simple_json(plan=plan),
                "output": dump_io_type(output),
                "plan_id": plan_id,
            }
        )
        return plan, output

    def test_machine_learning_news_summary(self):
        prompt = (
            "Can you give me a single summary of news published in the last week about machine "
            "learning at Meta, Apple, and Microsoft?"
        )

        def validate_output(prompt: str, output: IOType):
            output_text = get_output(output)
            self.loop.run_until_complete(
                validate_and_compare_text(llm=self.llm, output_text=output_text, prompt=prompt)
            )

        self.prompt_test(
            prompt=prompt, validate_plan=validate_plan, validate_output=validate_output
        )

    def test_open_close_spread(self):
        prompt = "Calculate the spread between open and close for AAPL over the month of Jan 2024"

        def validate_output(prompt: str, output: IOType):
            output_stock_table = get_output(output)
            date_column = validate_table_and_get_columns(
                output_stock_table=output_stock_table, column_types=[TableColumnType.DATE]
            )[0]
            self.assertGreaterEqual(len(date_column.data), 10)
            self.assertLessEqual(len(date_column.data), 50)

        self.prompt_test(
            prompt=prompt, validate_plan=validate_plan, validate_output=validate_output
        )

    def test_main_kpi_compare(self):
        prompt = "Compare how the main KPI for Microsoft have been discussed in the last 2 earning's calls"

        def validate_output(prompt: str, output: IOType):
            output_text = get_output(output)
            self.loop.run_until_complete(
                validate_and_compare_text(llm=self.llm, output_text=output_text, prompt=prompt)
            )

        self.prompt_test(
            prompt=prompt, validate_plan=validate_plan, validate_output=validate_output
        )

    def test_graph_pe(self):
        prompt = "Graph the PE of health care stocks in QQQ over the past year"

        def validate_output(prompt: str, output: IOType):
            output_line_graph = get_output(output)
            validate_line_graph(output_line_graph=output_line_graph)
            self.assertGreaterEqual(len(output_line_graph.data[0].points), 150)
            self.assertLessEqual(len(output_line_graph.data[0].points), 400)

        self.prompt_test(
            prompt=prompt, validate_plan=validate_plan, validate_output=validate_output
        )

    def test_mcap_nvda(self):
        prompt = "Show me the market cap of NVDA?"

        def validate_output(prompt: str, output: IOType):
            output_stock_table = get_output(output)
            date_column, mcap_column = validate_table_and_get_columns(
                output_stock_table=output_stock_table,
                column_types=[TableColumnType.DATE, TableColumnType.CURRENCY],
            )
            self.assertGreater(len(mcap_column.data), 0)

        self.prompt_test(
            prompt=prompt, validate_plan=validate_plan, validate_output=validate_output
        )

    def test_pe_nvda(self):
        prompt = "Show me the PE of NVDA?"

        def validate_output(prompt: str, output: IOType):
            output_stock_table = get_output(output)
            date_column, pe_column = validate_table_and_get_columns(
                output_stock_table=output_stock_table,
                column_types=[TableColumnType.DATE, TableColumnType.FLOAT],
            )
            self.assertGreater(len(date_column.data), 0)
            self.assertGreater(len(pe_column.data), 0)

        self.prompt_test(
            prompt=prompt, validate_plan=validate_plan, validate_output=validate_output
        )

    def test_market_commentary(self):
        prompt = "Write a market commentary of everything that has happened over the past week?"

        def validate_output(prompt: str, output: IOType):
            output_text = get_output(output=output)
            self.loop.run_until_complete(
                validate_and_compare_text(llm=self.llm, output_text=output_text, prompt=prompt)
            )

        self.prompt_test(
            prompt=prompt, validate_plan=validate_plan, validate_output=validate_output
        )

    def test_stock_screener_spy(self):
        prompt = (
            "Can you find good buying opportunities in SPY? Make sure PE > 20 "
            "and PE < 30, news is positive, and their earnings mention Generative AI."
        )

        def validate_output(prompt: str, output: IOType):
            output_stock_ids = get_output(output=output)
            self.assertGreater(len(output_stock_ids), 0)

        self.prompt_test(
            prompt=prompt, validate_plan=validate_plan, validate_output=validate_output
        )

    def test_sector_stocks(self):
        prompt = "Find stocks in the technology sector"

        def validate_output(prompt: str, output: IOType):
            self.assertGreater(len(output), 0)

        self.prompt_test(
            prompt=prompt, validate_plan=validate_plan, validate_output=validate_output
        )

    def test_relative_strength(self):
        prompt = "Show me Relative Strength Index for NVDA, AMD, INTL and GOOG over the past year"

        def validate_output(prompt: str, output: IOType):
            output_line_graph = get_output(output=output)
            validate_line_graph(output_line_graph=output_line_graph)
            self.assertGreaterEqual(len(output_line_graph.data[0].points), 150)
            self.assertLessEqual(len(output_line_graph.data[0].points), 400)
            self.assertEqual(len(output_line_graph.data), 4)

        self.prompt_test(
            prompt=prompt, validate_plan=validate_plan, validate_output=validate_output
        )

    def test_exploration_expense(self):
        prompt = "Show Exploration Expense for XOM"

        def validate_output(prompt: str, output: IOType):
            output_stock_table = get_output(output=output)
            validate_table_and_get_columns(output_stock_table=output_stock_table, column_types=[])

        self.prompt_test(
            prompt=prompt, validate_plan=validate_plan, validate_output=validate_output
        )

    def test_plot_tsla_price(self):
        prompt = "plot tsla price"

        def validate_output(prompt: str, output: IOType):
            output_line_graph = get_output(output)
            validate_line_graph(output_line_graph=output_line_graph)

        self.prompt_test(
            prompt=prompt, validate_plan=validate_plan, validate_output=validate_output
        )

    def test_top_mcap(self):
        prompt = "top 10 by market cap today, and then graph their market caps over the last month"

        def validate_output(prompt: str, output: IOType):
            output_line_graph = get_output(output=output)
            validate_line_graph(output_line_graph=output_line_graph)
            self.assertEqual(len(output_line_graph.data), 10)
            self.assertGreaterEqual(len(output_line_graph.data[0].points), 10)

        self.prompt_test(
            prompt=prompt, validate_plan=validate_plan, validate_output=validate_output
        )

    def test_intersection_of_qqq_xlv(self):
        prompt = "Find the intersection of QQQ and XLV"

        def validate_output(prompt: str, output: IOType):
            output_stock_ids = get_output(output=output)
            self.assertGreater(len(output_stock_ids), 0)

        self.prompt_test(
            prompt=prompt, validate_plan=validate_plan, validate_output=validate_output
        )
