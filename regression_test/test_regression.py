# type: ignore
import asyncio
import datetime
import json
import traceback
import unittest
import uuid
from typing import Optional
from uuid import uuid4

from gbi_common_py_utils.utils.event_logging import log_event

from agent_service.endpoints.models import AgentMetadata
from agent_service.io_type_utils import dump_io_type
from agent_service.planner.executor import (
    create_execution_plan_local,
    run_execution_plan_local,
)
from agent_service.planner.planner import plan_to_json
from agent_service.planner.planner_types import ExecutionPlan
from agent_service.types import ChatContext, Message, PlanRunContext
from agent_service.utils.date_utils import get_now_utc
from agent_service.utils.postgres import DEFAULT_AGENT_NAME, get_psql


def plan_to_simple_json(plan: ExecutionPlan) -> str:
    json_list = []
    for node in plan.nodes:
        json_list.append({"tool_name": node.tool_name})
    return json.dumps(json_list)


class TestExecutionPlanner(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.loop = asyncio.get_event_loop()  # type: ignore[assignment]
        cls.test_suite_id = str(uuid.uuid4())

    def run_regression(self, prompt: str, user_id: Optional[str] = None, do_chat: bool = False):
        agent_id = str(uuid.uuid4())
        shared_log_data = {
            "user_id": user_id,
            "test_suite_id": self.test_suite_id,
            "prompt": prompt,
        }
        user_id = user_id if user_id else str(uuid.uuid4())
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
            log_event(
                event_name="agent-regression-plan-generated",
                event_data={
                    "started_at_utc": execution_plan_start,
                    "finished_at_utc": execution_plan_finished_at,
                    "execution_plan": plan_to_json(plan=plan),
                    "plan_id": plan_id,
                    **shared_log_data,
                },
            )
        except Exception as e:
            log_event(
                event_name="agent-regression-plan-generated",
                event_data={
                    "started_at_utc": execution_plan_start,
                    "finished_at_utc": datetime.datetime.utcnow().isoformat(),
                    "error_msg": traceback.format_exc(),
                    "plan_id": plan_id,
                    **shared_log_data,
                },
            )
            raise e
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
        try:
            output = self.loop.run_until_complete(
                run_execution_plan_local(
                    plan=plan,
                    context=context,
                    do_chat=do_chat,
                    log_all_outputs=False,
                    replan_execution_error=False,
                )
            )
            log_event(
                event_name="agent-regression-output-generated",
                event_data={
                    "execution_plan_started_at_utc": execution_plan_start,
                    "execution_plan_finished_at_utc": execution_plan_finished_at,
                    "execution_start_at_utc": execution_started_at,
                    "execution_finished_at_utc": datetime.datetime.utcnow().isoformat(),
                    "execution_plan": plan_to_json(plan=plan),
                    "execution_plan_simple": plan_to_simple_json(plan=plan),
                    "output": dump_io_type(output),
                    "plan_id": plan_id,
                    **shared_log_data,
                },
            )
        except Exception as e:
            log_event(
                event_name="agent-regression-output-generated",
                event_data={
                    "execution_plan_started_at_utc": execution_plan_start,
                    "execution_plan_finished_at_utc": execution_plan_finished_at,
                    "execution_start_at_utc": execution_started_at,
                    "execution_finished_at_utc": datetime.datetime.utcnow().isoformat(),
                    "execution_plan": plan_to_json(plan=plan),
                    "execution_plan_simple": plan_to_simple_json(plan=plan),
                    "error_msg": traceback.format_exc(),
                    "plan_id": plan_id,
                    **shared_log_data,
                },
            )
            raise e

    def test_machine_learning_news_summary(self):
        prompt = (
            "Can you give me a single summary of news published in the last week about machine "
            "learning at Meta, Apple, and Microsoft?"
        )
        self.run_regression(prompt=prompt)

    def test_open_close_spread(self):
        prompt = "Calculate the spread between open and close for AAPL over the month of Jan 2024"
        self.run_regression(prompt=prompt)

    def test_main_kpi_compare(self):
        prompt = "Compare how the main KPI for Microsoft have been discussed in the last 2 earning's calls"
        self.run_regression(prompt=prompt)
