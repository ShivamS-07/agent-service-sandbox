# type: ignore
import asyncio
import inspect
import json
import logging
import os
import traceback
import unittest
import uuid
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
from uuid import uuid4

from gbi_common_py_utils.utils.clickhouse_base import ClickhouseBase
from gbi_common_py_utils.utils.environment import DEV_TAG

from agent_service.endpoints.models import AgentMetadata
from agent_service.GPT.constants import CLIENT_NAMESPACE, DEFAULT_CHEAP_MODEL
from agent_service.GPT.requests import GPT
from agent_service.io_type_utils import IOType, dump_io_type
from agent_service.planner.executor import (
    create_execution_plan_local,
    run_execution_plan_local,
)
from agent_service.planner.planner import plan_to_json
from agent_service.planner.planner_types import ExecutionPlan, SamplePlan
from agent_service.planner.utils import get_similar_sample_plans
from agent_service.types import ChatContext, Message, PlanRunContext
from agent_service.utils.date_utils import get_now_utc
from agent_service.utils.output_utils.output_construction import PreparedOutput
from agent_service.utils.postgres import DEFAULT_AGENT_NAME, get_psql

logger = logging.getLogger(__name__)


CH = ClickhouseBase(environment=DEV_TAG)
SERVICE_VERSION = "374053208103.dkr.ecr.us-west-2.amazonaws.com/agent-service:405d39f6fb15ad617fae2584cd812ae51ca033a9"


class PlanGenerationError(Exception):
    pass


def skip_in_ci(test_func):
    def wrapper(*args, **kwargs):
        if os.getenv("RUN_IN_CI") == "true":
            raise unittest.SkipTest("Skipping test, don't run in CI")
        return test_func(*args, **kwargs)

    return wrapper


@dataclass
class EventLog:
    event_name: str
    event_data: Optional[Dict[str, Any]] = field(default_factory=dict)


def force_log_bulk(events: List[EventLog]):
    ch = ClickhouseBase()
    agent_service_version = os.getenv("AGENT_SERVICE_VERSION")
    if not agent_service_version:
        raise Exception("Missing agent service version")
    event_metadata = {"kubernetes": {"container_image": agent_service_version}}
    event_dicts = []
    for event in events:
        event_dicts.append(
            {
                "event_data": json.dumps(event.event_data),
                "event_name": event.event_name,
                "timestamp": get_now_utc(),
                "event_metadata": json.dumps(event_metadata),
            }
        )
    ch.multi_row_insert(table_name="events", rows=event_dicts)


def plan_to_simple_json(plan: ExecutionPlan) -> str:
    json_list = []
    for node in plan.nodes:
        json_list.append({"tool_name": node.tool_name})
    return json.dumps(json_list, indent=4)


def compare_plan(actual: ExecutionPlan, expected: ExecutionPlan):
    actual_tools_called = set([step.tool_name for step in actual.nodes])
    expected_tools_called = set([step.tool_name for step in expected.nodes])

    if actual_tools_called != expected_tools_called:
        return False
    return True


def get_output(output: IOType) -> IOType:
    if isinstance(output, PreparedOutput):
        return output.val
    if isinstance(output, list) and isinstance(output[0], PreparedOutput):
        return get_output(output[0])
    return output


def validate_tools_used(
    prompt: str,
    plan: Optional[ExecutionPlan],
    required_tools: List[str],
    disallowed_tools: List[str],
) -> None:
    actual_tools = set((step.tool_name for step in plan.nodes))
    assert len(required_tools) > 0, "No required tools provided"
    missing_required_tools = set(required_tools).difference(set(actual_tools))
    err_msg = (
        "The required tools are not called.\n"
        f"Missing required tools are {missing_required_tools}\n"
        f"Actual plan is {plan_to_simple_json(plan)}"
    )
    assert set(required_tools).issubset(set(actual_tools)), err_msg

    disallowed_tools_used = set(disallowed_tools).intersection(set(actual_tools))

    err_msg = f"""
    Some of the disallowed tools are called.\n
    disallowed tools that were used are {disallowed_tools_used}\n
    actual plan is {plan_to_simple_json(plan)}
    """

    assert len(disallowed_tools_used) == 0, err_msg


def validate_required_sample_plans(
    sample_plans: List[SamplePlan], required_sample_plans: List[SamplePlan]
) -> None:
    assert set(required_sample_plans).issubset(
        set([sample_plan.id for sample_plan in sample_plans])
    )


class TestExecutionPlanner(unittest.TestCase):
    test_suite_id = str(uuid.uuid4())

    @classmethod
    def setUpClass(cls) -> None:
        cls.loop = asyncio.get_event_loop()  # type: ignore[assignment]
        cls.llm = GPT(model=DEFAULT_CHEAP_MODEL)
        cls.logs = []

    def prompt_test(
        self,
        prompt: str,
        validate_output: Callable,
        required_tools: List[str] = [],
        disallowed_tools: List[str] = [],
        raise_plan_validation_error: Optional[bool] = False,
        raise_output_validation_error: Optional[bool] = False,
        user_id: Optional[str] = None,
        required_sample_plans: List[str] = [],
        validate_tool_args: Callable = lambda execution_log: None,
        only_validate_plan: bool = os.getenv("RUN_IN_CI", False),
    ):
        test_name = inspect.stack()[1].function
        user_id = user_id or "6c14fe54-de50-4d05-9533-57541715064f"
        agent_id = str(uuid.uuid4())
        shared_log_data = {
            "user_id": user_id,
            "test_suite_id": self.test_suite_id,
            "prompt": prompt,
            "test_name": test_name,
            "agent_id": agent_id,
        }
        regression_test_log = EventLog(
            event_name="agent-regression-output-generated",
            event_data={
                **shared_log_data,
            },
        )

        warning_msg = ""
        try:
            sample_plans, plan, output, execution_log = self.run_regression(
                prompt=prompt,
                regression_test_log=regression_test_log,
                user_id=user_id,
                agent_id=agent_id,
                skip_output=only_validate_plan,
            )
            try:
                if required_sample_plans and not only_validate_plan:
                    validate_required_sample_plans(
                        sample_plans=sample_plans, required_sample_plans=required_sample_plans
                    )
                validate_tools_used(
                    prompt=prompt,
                    plan=plan,
                    required_tools=required_tools,
                    disallowed_tools=disallowed_tools,
                )
            except AssertionError as e:
                if raise_plan_validation_error or only_validate_plan:
                    raise e
                else:
                    warning_msg += f"Plan validation warning -\n{e}\n"
            try:
                if not only_validate_plan:
                    validate_output(prompt=prompt, output=output)
                    validate_tool_args(execution_log=execution_log)
            except AssertionError as e:
                if raise_output_validation_error:
                    raise e
                else:
                    warning_msg += f"Output validation warning -\n{e}\n"
            if warning_msg:
                regression_test_log.event_data["warning_msg"] = warning_msg.strip()
                warnings.warn(warning_msg)

            self.logs.append(regression_test_log)
        except Exception as e:
            regression_test_log.event_data["error_msg"] = traceback.format_exc()
            self.logs.append(regression_test_log)
            raise e

    def run_regression(
        self,
        prompt: str,
        regression_test_log: EventLog,
        user_id: str,
        agent_id: str,
        skip_output: bool,
        do_chat: bool = False,
    ):
        agent = AgentMetadata(
            agent_id=agent_id,
            user_id=user_id,
            agent_name=DEFAULT_AGENT_NAME,
            created_at=get_now_utc(),
            last_updated=get_now_utc(),
            deleted=False,
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
        chat = ChatContext(
            messages=[Message(message=prompt, is_user_message=True, agent_id=agent_id)]
        )
        logger.warning(f"test started {plan_id=}, {prompt=}")
        try:
            sample_plans = self.loop.run_until_complete(
                get_similar_sample_plans(input=chat.get_gpt_input(client_only=True))
            )
            execution_plan_start = get_now_utc().isoformat()
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
            execution_plan_finished_at = get_now_utc().isoformat()
            regression_test_log.event_data.update(
                {
                    "execution_plan_started_at_utc": execution_plan_start,
                    "execution_plan_finished_at_utc": execution_plan_finished_at,
                    "execution_plan": plan_to_json(plan=plan),
                    "plan_id": plan_id,
                    "execution_plan_simple": plan_to_simple_json(plan=plan),
                    "sample_plans": json.dumps(
                        [sample_plan.model_dump() for sample_plan in sample_plans]
                    ),
                }
            )
        except Exception as e:
            raise PlanGenerationError(e)
        if skip_output:
            return sample_plans, plan, None, None
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
        execution_started_at = get_now_utc().isoformat()
        execution_log = defaultdict(list)
        output = self.loop.run_until_complete(
            run_execution_plan_local(
                plan=plan,
                context=context,
                do_chat=do_chat,
                log_all_outputs=False,
                replan_execution_error=False,
                execution_log=execution_log,
            )
        )
        regression_test_log.event_data.update(
            {
                "execution_start_at_utc": execution_started_at,
                "execution_finished_at_utc": get_now_utc().isoformat(),
                "output": dump_io_type(output),
            }
        )
        logger.warning(f"test completed {plan_id=}, {prompt=}")
        return sample_plans, plan, output, execution_log

    @classmethod
    def tearDownClass(cls) -> None:
        if CLIENT_NAMESPACE != "LOCAL":
            force_log_bulk(cls.logs)
