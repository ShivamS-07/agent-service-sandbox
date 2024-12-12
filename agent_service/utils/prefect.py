import enum
import json
import logging
from logging import Logger, LoggerAdapter
from typing import Dict, Optional, Union

import boto3
from gbi_common_py_utils.utils.event_logging import json_serial

from agent_service.planner.constants import FollowupAction
from agent_service.planner.planner_types import ErrorInfo, ExecutionPlan
from agent_service.types import PlanRunContext
from agent_service.utils.constants import AGENT_WORKER_QUEUE, BOOSTED_DAG_QUEUE
from agent_service.utils.date_utils import get_now_utc
from agent_service.utils.feature_flags import use_boosted_dag_for_run_execution_plan
from agent_service.utils.logs import async_perf_logger
from agent_service.utils.s3_upload import upload_string_to_s3

logger = logging.getLogger(__name__)


@async_perf_logger
async def kick_off_create_execution_plan(
    agent_id: str,
    plan_id: str,
    user_id: str,
    action: FollowupAction = FollowupAction.CREATE,
    error_info: Optional[ErrorInfo] = None,
    skip_db_commit: bool = False,
    skip_task_cache: bool = False,
    run_plan_in_prefect_immediately: bool = True,
) -> None:
    sqs = boto3.resource("sqs", region_name="us-west-2")
    arguments = {
        "agent_id": agent_id,
        "plan_id": plan_id,
        "user_id": user_id,
        "skip_db_commit": skip_db_commit,
        "skip_task_cache": skip_task_cache,
        "run_plan_in_prefect_immediately": run_plan_in_prefect_immediately,
        "action": action,
        "error_info": error_info.model_dump() if error_info else error_info,
    }
    message = {
        "method": "create_execution_plan",
        "arguments": arguments,
        "send_time_utc": get_now_utc().isoformat(),
    }
    queue = sqs.get_queue_by_name(QueueName=AGENT_WORKER_QUEUE)
    queue.send_message(MessageBody=json.dumps(message, default=json_serial))


@async_perf_logger
async def kick_off_run_execution_plan(
    plan: ExecutionPlan,
    context: PlanRunContext,
    do_chat: bool = True,
    override_task_output_id_lookup: Optional[Dict[str, str]] = None,
    override_task_work_log_id_lookup: Optional[Dict[str, str]] = None,
    replan_execution_error: bool = False,
) -> None:
    sqs = boto3.resource("sqs", region_name="us-west-2")
    arguments = {
        "plan": plan.model_dump(),
        "context": context.model_dump(),
        "do_chat": do_chat,
        "override_task_output_id_lookup": override_task_output_id_lookup,
        "override_task_work_log_id_lookup": override_task_work_log_id_lookup,
        "replan_execution_error": replan_execution_error,
    }
    message_contents = {
        "method": "run_execution_plan",
        "arguments": arguments,
        "send_time_utc": get_now_utc().isoformat(),
    }
    message = {
        "s3_path": upload_string_to_s3(data=json.dumps(message_contents, default=json_serial)),
        "method": "run_execution_plan",
        "agent_id": context.agent_id,
        "plan_id": context.plan_id,
        "plan_run_id": context.plan_run_id,
        "user_id": context.user_id,
    }
    queue_name = (
        BOOSTED_DAG_QUEUE if use_boosted_dag_for_run_execution_plan() else AGENT_WORKER_QUEUE
    )
    queue = sqs.get_queue_by_name(QueueName=queue_name)
    queue.send_message(MessageBody=json.dumps(message, default=json_serial))


def get_task_run_name(ctx: PlanRunContext) -> str:
    return f"{ctx.plan_run_id}:{ctx.task_id}"


def get_prefect_logger(name: str) -> Union[Logger, LoggerAdapter]:
    return logging.getLogger(name)


class ExecutionRunType(enum.Enum):
    PLAN_EXECUTION = 1
    PLAN_CREATION = 2
