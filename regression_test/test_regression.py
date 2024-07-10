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
from agent_service.io_type_utils import IOType, dump_io_type
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

    def prompt_test(
        self,
        prompt: str,
        validate_plan: Callable,
        validate_output: Callable,
        user_id: Optional[str] = None,
    ):
        user_id = user_id or str(uuid.uuid4())
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
