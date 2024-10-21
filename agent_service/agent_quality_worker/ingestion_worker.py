import json
import logging
import random
from datetime import datetime

import boto3
from gbi_common_py_utils.utils.environment import (
    DEV_TAG,
    LOCAL_TAG,
    get_environment_tag,
)
from gbi_common_py_utils.utils.event_logging import json_serial

from agent_service.agent_quality_worker.constants import (
    HORIZON_USERS_DEV,
    HORIZON_USERS_PROD,
)
from agent_service.agent_quality_worker.models import HorizonTabs
from agent_service.endpoints.models import AgentQC, Status
from agent_service.utils.async_db import AsyncDB
from agent_service.utils.constants import AGENT_QUALITY_WORKER_QUEUE
from agent_service.utils.date_utils import get_now_utc
from agent_service.utils.postgres import SyncBoostedPG


async def send_agent_quality_message(
    agent_id: str, plan_id: str, status: Status, db: AsyncDB
) -> None:
    sqs = boto3.resource("sqs", region_name="us-west-2")
    arguments = {
        "agent_id": agent_id,
        "plan_id": plan_id,
        "user_id": await db.get_agent_owner(agent_id),
        "status": "CS",
    }
    message = {
        "method": "agent_quality",
        "arguments": arguments,
        "send_time_utc": get_now_utc().isoformat(),
    }
    queue = sqs.get_queue_by_name(QueueName=AGENT_QUALITY_WORKER_QUEUE)
    queue.send_message(MessageBody=json.dumps(message, default=json_serial))


async def assign_agent_quality_reviewers(
    agent_id: str, plan_id: str, user_id: str, status: str, skip_db_commit: bool = False
) -> None:
    db = AsyncDB(pg=SyncBoostedPG(skip_commit=skip_db_commit))
    # get the correct quality agent
    # we can potentially get multiple for
    # one agent id
    agent_qc_id = await db.get_agent_qc_id_by_agent_id(agent_id=agent_id, plan_id=plan_id)

    if not agent_qc_id:
        logging.info(f"Agent QC not found for {agent_qc_id=}")
        return

    env = get_environment_tag()
    is_dev = env in (DEV_TAG, LOCAL_TAG)
    horizon_users_dict = HORIZON_USERS_PROD

    if is_dev:
        horizon_users_dict = HORIZON_USERS_DEV

    # get the members of the teams to assign cs_reviewer, prod_reviewer, eng_reviewer
    cs_team_members = []
    eng_team_members = []
    reviewer_team_members = []

    for horizon_user in horizon_users_dict.values():
        if horizon_user.userType == HorizonTabs.CS.value:
            cs_team_members.append(horizon_user.userId)
        elif horizon_user.userType == HorizonTabs.ENG.value:
            eng_team_members.append(horizon_user.userId)
        elif horizon_user.userType == HorizonTabs.PROD.value:
            reviewer_team_members.append(horizon_user.userId)

    # randomly choose the reviewer
    cs_reviewer = random.choice(cs_team_members)
    eng_reviewer = random.choice(eng_team_members)
    prod_reviewer = random.choice(reviewer_team_members)

    agent_qc = AgentQC(
        agent_qc_id=agent_qc_id,
        agent_id=agent_id,
        user_id=user_id,
        agent_status=status,
        plan_id=plan_id,
        cs_reviewer=cs_reviewer,
        eng_reviewer=eng_reviewer,
        prod_reviewer=prod_reviewer,
        last_updated=datetime.now(),
    )

    await db.update_agent_qc(agent_qc)
