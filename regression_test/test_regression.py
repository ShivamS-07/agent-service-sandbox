# type: ignore
import asyncio
import datetime
import inspect
import json
import traceback
import unittest
import uuid
import warnings
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List, Optional
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
SERVICE_VERSION = "374053208103.dkr.ecr.us-west-2.amazonaws.com/agent-service:405d39f6fb15ad617fae2584cd812ae51ca033a9"


class PlanGenerationError(Exception):
    pass


class ToolsValidationError(Exception):
    pass


class OutputValidationError(Exception):
    pass


@dataclass
class EventLog:
    event_name: str
    event_data: Optional[Dict[str, Any]] = None


def plan_to_simple_json(plan: ExecutionPlan) -> str:
    json_list = []
    for node in plan.nodes:
        json_list.append({"tool_name": node.tool_name})
    return json.dumps(json_list)


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
    prompt: str, plan: Optional[ExecutionPlan], required_tools: List[str]
) -> None:
    actual_tools = set((step.tool_name for step in plan.nodes))

    if len(required_tools) == 0:
        error_msg = "Could not find required tool groups"
        raise ToolsValidationError(error_msg)
    if set(required_tools).issubset(set(actual_tools)):
        return
    err_msg = f"""
        The required tools are not called.
        actual plan is {plan_to_simple_json(plan)} and
        required tools are {required_tools}
        """
    raise ToolsValidationError(f"\n{err_msg}")


class TestExecutionPlanner(unittest.TestCase):
    test_suite_id = str(uuid.uuid4())

    @classmethod
    def setUpClass(cls) -> None:
        cls.loop = asyncio.get_event_loop()  # type: ignore[assignment]
        cls.llm = GPT(model=DEFAULT_CHEAP_MODEL)

    def prompt_test(
        self,
        prompt: str,
        validate_output: Callable,
        required_tools: List[str] = [],
        raise_plan_validation_error: Optional[bool] = False,
        raise_output_validation_error: Optional[bool] = False,
        user_id: Optional[str] = None,
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
            plan, output = self.run_regression(
                prompt=prompt,
                regression_test_log=regression_test_log,
                user_id=user_id,
                agent_id=agent_id,
            )
            try:
                validate_tools_used(prompt=prompt, plan=plan, required_tools=required_tools)
            except ToolsValidationError as e:
                if raise_plan_validation_error:
                    raise e
                else:
                    warning_msg += f"Plan validation warning: {e}\n"
            try:
                validate_output(prompt=prompt, output=output)
            except OutputValidationError as e:
                if raise_output_validation_error:
                    raise e
                else:
                    warning_msg += f"Output validation warning: {e}\n"
            if warning_msg:
                regression_test_log.event_data["warning_msg"] = warning_msg.strip()
                warnings.warn(warning_msg)
            log_event(**asdict(regression_test_log))
        except (PlanGenerationError, ToolsValidationError) as e:
            regression_test_log.event_data["error_msg"] = traceback.format_exc()
            log_event(**asdict(regression_test_log))
            raise e
        except Exception as e:
            regression_test_log.event_data["error_msg"] = traceback.format_exc()
            log_event(**asdict(regression_test_log))
            raise e

    def run_regression(
        self,
        prompt: str,
        regression_test_log: EventLog,
        user_id: str,
        agent_id: str,
        do_chat: bool = False,
    ):
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
        chat = ChatContext(
            messages=[Message(message=prompt, is_user_message=True, agent_id=agent_id)]
        )
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
            regression_test_log.event_data.update(
                {
                    "execution_plan_started_at_utc": execution_plan_start,
                    "execution_plan_finished_at_utc": execution_plan_finished_at,
                    "execution_plan": plan_to_json(plan=plan),
                    "plan_id": plan_id,
                    "execution_plan_simple": plan_to_simple_json(plan=plan),
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
        regression_test_log.event_data.update(
            {
                "execution_start_at_utc": execution_started_at,
                "execution_finished_at_utc": datetime.datetime.utcnow().isoformat(),
                "output": dump_io_type(output),
            }
        )
        return plan, output
