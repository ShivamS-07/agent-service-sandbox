import datetime
import json
import logging
import random
from typing import List

import boto3
from gbi_common_py_utils.utils.environment import (
    DEV_TAG,
    LOCAL_TAG,
    get_environment_tag,
)
from gbi_common_py_utils.utils.event_logging import json_serial

from agent_service.agent_quality_worker.constants import (
    CS_REVIEWER_COLUMN,
    HORIZON_USERS_DEV,
    HORIZON_USERS_PROD,
    MAX_REVIEWS,
)
from agent_service.agent_quality_worker.models import HorizonTabs
from agent_service.endpoints.models import (
    AgentQC,
    HorizonCriteria,
    HorizonCriteriaOperator,
    Status,
)
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


async def get_number_of_assigned(
    user_id: str, db: AsyncDB, reviewer_column: str, num_days_lookback: int
) -> int:
    # Get the current date and calculate the date 7 days ago
    now = datetime.datetime.now()
    seven_days_ago = now - datetime.timedelta(days=num_days_lookback)

    # Define the criteria for the 7-day lookback period and reviewer
    criteria = [
        HorizonCriteria(
            column=reviewer_column,
            operator=HorizonCriteriaOperator.equal,
            arg1=user_id,
            arg2=None,
        ),
        HorizonCriteria(
            column="aqc.created_at",
            operator=HorizonCriteriaOperator.between,
            arg1=seven_days_ago,
            arg2=now,
        ),
    ]

    # Fetch results based on the criteria
    _, count = await db.search_agent_qc(criteria, [])

    return count


async def get_users_with_assigned_counts_fewer_than(
    user_ids: List[str],
    db: AsyncDB,
    reviewer_column: str,
    num_days_lookback: int = 7,
    max_allowed: int = MAX_REVIEWS,
) -> List[str]:
    # List to hold the count of assigned tickets for each user_id with count under 150
    users = []

    # Loop over each user_id and get the count using get_number_of_assigned
    for user_id in user_ids:
        count = await get_number_of_assigned(user_id, db, reviewer_column, num_days_lookback)

        # Only add to dictionary the count per user is under 150/ by length
        if count / len(user_ids) < max(1, int(max_allowed / len(user_ids))):
            users.append(user_id)

    return users


async def assign_agent_quality_reviewers(
    agent_id: str, plan_id: str, user_id: str, status: str, skip_db_commit: bool = False
) -> None:
    db = AsyncDB(pg=SyncBoostedPG(skip_commit=skip_db_commit))

    logging.info(f"Retrieving Agent QC for agent_id: {agent_id}, plan_id: {plan_id}")
    # get the correct quality agent
    # we can potentially get multiple for
    # one agent id
    agent_qc_id, is_spoofed = await db.get_agent_qc_id_by_agent_id(
        agent_id=agent_id, plan_id=plan_id
    )

    if not agent_qc_id:
        logging.info(f"Agent QC not found for {agent_qc_id=}")
        return

    if is_spoofed:
        logging.info(f"Agent is spoofed, skipping assignment for agent_id: {agent_id}")
        return None

    env = get_environment_tag()
    is_dev = env in (DEV_TAG, LOCAL_TAG)

    # Check if the user is internal and if the env is prod
    # we want to stop internal queries from getting to triage
    if not is_dev and db.is_user_internal(user_id):
        logging.info(
            f"Agent created by internal user, skipping assignment for agent_id: {agent_id}"
        )
        return None

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

    # get the list of reviews that have space to be assigned new tickets
    cs_team_members = await get_users_with_assigned_counts_fewer_than(
        user_ids=cs_team_members, db=db, reviewer_column=CS_REVIEWER_COLUMN
    )

    # if no CS free then return
    if len(cs_team_members) == 0:
        logging.info(f"CS has no allocation, skipping assignment for agent_id: {agent_id}")
        return

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
        last_updated=datetime.datetime.now(),
        # this field can't be updated but should be False here regardless
        is_spoofed=False,
    )

    await db.update_agent_qc(agent_qc)
    logging.info(f"Successfully assigned reviewers for agent_id: {agent_id}")
