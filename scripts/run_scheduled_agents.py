import argparse
import json
import uuid
from typing import List

import boto3
from gbi_common_py_utils.utils.event_logging import json_serial

from agent_service.planner.planner_types import ExecutionPlan
from agent_service.tools import *  # noqa
from agent_service.types import PlanRunContext
from agent_service.utils.constants import AGENT_AUTOMATION_WORKER_QUEUE
from agent_service.utils.date_utils import get_now_utc
from agent_service.utils.logs import init_stdout_logging
from agent_service.utils.postgres import get_psql


def prefect_automated_run_execution_plan(plan: ExecutionPlan, context: PlanRunContext) -> None:
    sqs = boto3.resource("sqs", region_name="us-west-2")
    arguments = {
        "plan": plan.model_dump(),
        "context": context.model_dump(),
        "do_chat": False,
        "replan_execution_error": False,
        "scheduled_by_automation": True,
    }
    message = {
        "method": "run_execution_plan",
        "arguments": arguments,
        "send_time_utc": get_now_utc().isoformat(),
    }
    queue = sqs.get_queue_by_name(QueueName=AGENT_AUTOMATION_WORKER_QUEUE)
    queue.send_message(MessageBody=json.dumps(message, default=json_serial))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--agent-ids", type=str, nargs="*", required=False, default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    init_stdout_logging()
    db = get_psql()
    agent_ids: List[str] = args.agent_ids
    if not agent_ids:
        agent_ids = db.get_scheduled_agents()
    agents_info = db.get_live_agents_info(agent_ids=agent_ids)
    chat_contexts = db.get_chat_contexts(agent_ids=agent_ids)
    for agent_info in agents_info:
        prefect_automated_run_execution_plan(
            plan=ExecutionPlan(**agent_info["plan"]),
            context=PlanRunContext(
                agent_id=agent_info["agent_id"],
                plan_id=agent_info["plan_id"],
                user_id=agent_info["user_id"],
                plan_run_id=str(uuid.uuid4()),
                chat=chat_contexts.get(agent_info["agent_id"]),
            ),
        )


if __name__ == "__main__":
    main()
