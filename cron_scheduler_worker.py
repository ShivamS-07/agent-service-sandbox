import asyncio
import datetime
import json
import logging
import uuid
from dataclasses import dataclass
from typing import List

import boto3
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
from apscheduler.schedulers.asyncio import AsyncIOScheduler
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

SCHEDULE_UPDATE_DELAY = 60 * 5  # Check for new schedules every 5 min
# If the scheduler was disabled, kick off jobs that are up to 10 minutes late
MISFIRE_GRACE_PERIOD = 60 * 10

logger = logging.getLogger("cron_scheduler_worker")


def get_db_url() -> str:
    db = get_config().app_db
    return f"postgresql+psycopg://{db.username}:{db.password}@{db.host}:{db.port}/{db.database}"


@dataclass(frozen=True)
class AgentInfo:
    plan: ExecutionPlan
    context: PlanRunContext
    schedule: AgentSchedule


async def get_live_agent_infos() -> List[AgentInfo]:
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


async def start_agent_run(agent: AgentInfo) -> None:
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
    }
    queue = sqs.get_queue_by_name(QueueName=AGENT_AUTOMATION_WORKER_QUEUE)
    queue.send_message(MessageBody=json.dumps(message, default=json_serial))


async def run_schedule_with_recurring_agent_updates(scheduler: AsyncIOScheduler) -> None:
    scheduler.start()
    while True:
        # On first startup, make sure there are no lingering jobs
        scheduler.remove_all_jobs()
        scheduler.pause()
        logger.info("Scheduling agent cronjobs...")
        agent_infos = await get_live_agent_infos()
        logger.info(f"Got {len(agent_infos)} agents to process...")
        for agent_info in agent_infos:
            scheduler.add_job(
                func=start_agent_run,
                trigger=agent_info.schedule.to_cron_trigger(),
                kwargs={"agent": agent_info},
                id=agent_info.context.agent_id,
                name=f"start_agent_run_{agent_info.context.agent_id}",
            )
        logger.info("Agent cronjobs scheduled.")
        scheduler.resume()
        await asyncio.sleep(SCHEDULE_UPDATE_DELAY)
        logger.info(f"Waited {SCHEDULE_UPDATE_DELAY} seconds, refreshing agent cronjobs...")


async def main() -> None:
    jobstores = {
        "default": SQLAlchemyJobStore(
            url=get_db_url(), tablename="scheduled_jobs", tableschema="agent"
        ),
    }
    job_defaults = {
        "coalesce": True,
        "max_instances": 1,
        "misfire_grace_time": MISFIRE_GRACE_PERIOD,
    }
    scheduler = AsyncIOScheduler(
        jobstores=jobstores,
        job_defaults=job_defaults,
        timezone=datetime.timezone.utc,
        logger=logger,
    )
    await run_schedule_with_recurring_agent_updates(scheduler=scheduler)


if __name__ == "__main__":
    init_stdout_logging()
    init_sentry(disable_sentry=not EnvironmentUtils.is_deployed)
    asyncio.run(main())
