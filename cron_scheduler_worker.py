import argparse
import datetime
import hashlib
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass
from typing import List, Optional

import boto3
import redis
from apscheduler.executors.pool import ThreadPoolExecutor
from apscheduler.jobstores.memory import MemoryJobStore
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
from apscheduler.schedulers.background import BackgroundScheduler
from gbi_common_py_utils.config import get_config
from gbi_common_py_utils.utils.event_logging import json_serial

from agent_service.planner.planner_types import ExecutionPlan
from agent_service.types import PlanRunContext
from agent_service.utils.constants import AGENT_AUTOMATION_WORKER_QUEUE
from agent_service.utils.date_utils import get_now_utc
from agent_service.utils.environment import EnvironmentUtils
from agent_service.utils.logs import init_stdout_logging
from agent_service.utils.postgres import get_psql
from agent_service.utils.s3_upload import upload_string_to_s3
from agent_service.utils.scheduling import AgentSchedule
from agent_service.utils.sentry_utils import init_sentry

SCHEDULE_UPDATE_DELAY = 60 * 10  # Check for new schedules every 10 min
# If the scheduler was disabled, kick off jobs that are up to 10 minutes late
MISFIRE_GRACE_PERIOD = 60 * 45  # allow runs up to 45 minutes late
AGENT_LOCK_PERIOD = 60 * 50  # prevent duplicate runs for 50 min

logger = logging.getLogger("cron_scheduler_worker")

_REDIS_HOST = os.getenv("REDIS_QUEUE_HOST")
_REDIS_PORT = os.getenv("REDIS_PORT", "6379")

_REDIS_OPERATION_TIMEOUT = 1.0  # 1s
_REDIS_CONNECT_TIMEOUT = 5.0  # 5s

_REDIS = None

TEST_MODE: bool = False


def is_redis_available() -> bool:
    if _REDIS_HOST and _REDIS_PORT:
        return True
    return False


def get_redis_client() -> Optional[redis.Redis]:
    """
    Returns a global Redis instance if one is available, otherwise returns None.
    """
    global _REDIS
    # is_redis_available checks these, but needed for mypy
    if _REDIS is None and _REDIS_PORT and _REDIS_HOST and is_redis_available():
        logger.info(f"Initializing redis connection: {_REDIS_HOST}:{_REDIS_PORT}")
        _REDIS = redis.Redis(
            host=_REDIS_HOST,
            port=int(_REDIS_PORT),
            decode_responses=False,
            socket_timeout=_REDIS_OPERATION_TIMEOUT,
            socket_connect_timeout=_REDIS_CONNECT_TIMEOUT,
        )

    return _REDIS


REDIS_KEY_TEMPLATE = "agent-svc-cron-scheduler:{agent_id}"


def get_db_url() -> str:
    db = get_config().app_db
    return f"postgresql+psycopg://{db.username}:{db.password}@{db.host}:{db.port}/{db.database}"


@dataclass(frozen=True)
class AgentInfo:
    plan: ExecutionPlan
    context: PlanRunContext
    schedule: AgentSchedule


def get_live_agent_infos() -> List[AgentInfo]:
    db = get_psql()
    agent_ids = db.get_scheduled_agents()
    agents_info = db.get_live_agents_info(agent_ids=agent_ids)
    chat_contexts = db.get_chat_contexts(agent_ids=agent_ids)
    outputs = []
    for agent_info in agents_info:
        # Handle cases without schedule for backwards compatibility
        schedule = (
            AgentSchedule.model_validate(agent_info["schedule"])
            if agent_info["schedule"]
            else AgentSchedule.default()
        )
        chat_context = chat_contexts.get(agent_info["agent_id"])
        if chat_context:
            chat_context.messages = [
                message
                for message in chat_context.messages
                if message.message_time < agent_info["plan_created_at"]
            ]
        outputs.append(
            AgentInfo(
                plan=ExecutionPlan(**agent_info["plan"]),
                context=PlanRunContext(
                    agent_id=agent_info["agent_id"],
                    plan_id=agent_info["plan_id"],
                    user_id=agent_info["user_id"],
                    plan_run_id=str(uuid.uuid4()),
                    chat=chat_context,
                ),
                schedule=schedule,
            ),
        )
    return outputs


def start_agent_run(agent: AgentInfo) -> None:
    agent_key = REDIS_KEY_TEMPLATE.format(agent_id=agent.context.agent_id)
    redis = get_redis_client()
    if redis:
        # There seems to be a weird edge case where an agent run is scheduled
        # multiple times in quick succession. We'll prevent that by using redis
        # to store recently run plan_run_id's
        duplicate = redis.get(agent_key)
        if duplicate:
            logger.warning(f"Skipping duplicate scheduled run for {agent=}")
            return
        redis.set(
            name=agent_key,
            value="t",  # value doesn't matter
            ex=AGENT_LOCK_PERIOD,
        )
    logger.info(f"Running agent: {agent.context.agent_id}, schedule is: {agent.schedule}")
    sqs = boto3.resource("sqs", region_name="us-west-2")
    arguments = {
        "plan": agent.plan.model_dump(),
        "context": agent.context.model_dump(),
        "do_chat": False,
        "replan_execution_error": False,
        "scheduled_by_automation": True,
    }
    message_contents = {
        "method": "run_execution_plan",
        "arguments": arguments,
        "send_time_utc": get_now_utc().isoformat(),
    }
    message = {
        "s3_path": upload_string_to_s3(data=json.dumps(message_contents, default=json_serial)),
        "method": "run_execution_plan",
        "agent_id": agent.context.agent_id,
        "plan_id": agent.context.plan_id,
        "plan_run_id": agent.context.plan_run_id,
        "user_id": agent.context.user_id,
        "scheduled_by_automation": True,
    }
    queue = sqs.get_queue_by_name(QueueName=AGENT_AUTOMATION_WORKER_QUEUE)
    if not TEST_MODE:
        queue.send_message(MessageBody=json.dumps(message, default=json_serial))
    else:
        logger.info("Would have sent: " + json.dumps(message, default=json_serial))


def run_schedule_with_recurring_agent_updates(scheduler: BackgroundScheduler) -> None:
    scheduler.start()
    while True:
        # On first startup, make sure there are no lingering jobs
        scheduler.remove_all_jobs()
        scheduler.pause()
        logger.info("Scheduling agent cronjobs...")
        agent_infos = get_live_agent_infos()
        logger.info(f"Got {len(agent_infos)} agents to process...")
        for agent_info in agent_infos:
            jitter_minutes = 0
            scheduled_hours, scheduled_minutes = agent_info.schedule.get_schedule_hours_minutes()
            if scheduled_hours == 8 and scheduled_minutes == 0:
                # Add jitter to agents scheduled at the default time.  Convert
                # the agent_id hash to a number of minutes between -120 and
                # 0. This makes sure that agents don't overwhelm our system, but
                # also are consistent each day and won't be missed by the
                # scheduler.
                hashed_id = int(hashlib.md5(agent_info.context.agent_id.encode()).hexdigest(), 16)
                jitter_minutes = (hashed_id % 120) * -1
                logger.info(f"Agent {agent_info.context.agent_id} scheduled with {jitter_minutes=}")
            scheduler.add_job(
                func=start_agent_run,
                trigger=agent_info.schedule.to_cron_trigger(jitter_minutes=jitter_minutes),
                kwargs={"agent": agent_info},
                id=agent_info.context.agent_id,
                name=f"start_agent_run_{agent_info.context.agent_id}",
                replace_existing=True,
                max_instances=1,
            )
        logger.info("Agent cronjobs scheduled.")
        scheduler.resume()
        time.sleep(SCHEDULE_UPDATE_DELAY)
        logger.info(f"Waited {SCHEDULE_UPDATE_DELAY} seconds, refreshing agent cronjobs...")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test",
        action="store_true",
        help=(
            "Runs the scheduler with an in memory job store instead of Postgres, "
            "and logs instead of writing an actual sqs"
        ),
        default=False,
    )
    return parser.parse_args()


def main() -> None:
    global TEST_MODE
    args = parse_args()
    TEST_MODE = args.test
    job_store = SQLAlchemyJobStore(
        url=get_db_url(), tablename="scheduled_jobs", tableschema="agent"
    )
    if TEST_MODE:
        logger.info("STARTING IN TEST MODE")
        logging.getLogger("apscheduler").setLevel(logging.DEBUG)
        job_store = MemoryJobStore()
    jobstores = {
        "default": job_store,
    }
    job_defaults = {
        "coalesce": True,
        "max_instances": 1,
        "misfire_grace_time": MISFIRE_GRACE_PERIOD,
    }
    executors = {"default": ThreadPoolExecutor(max_workers=1)}
    scheduler = BackgroundScheduler(
        executors=executors,
        jobstores=jobstores,
        job_defaults=job_defaults,
        timezone=datetime.timezone.utc,
        logger=logger,
    )
    run_schedule_with_recurring_agent_updates(scheduler=scheduler)


if __name__ == "__main__":
    init_stdout_logging()
    init_sentry(disable_sentry=not EnvironmentUtils.is_deployed)
    main()
