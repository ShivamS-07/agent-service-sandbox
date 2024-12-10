import bisect
import datetime
import json
import logging
import random
from collections import defaultdict
from math import ceil
from typing import List, Optional

import boto3
from gbi_common_py_utils.utils.environment import (
    DEV_TAG,
    LOCAL_TAG,
    get_environment_tag,
)
from gbi_common_py_utils.utils.event_logging import json_serial

from agent_service.agent_quality_worker.constants import (
    CS_REVIEWER_COLUMN,
    CS_TIERED_ASSIGNMENT_ALLOCATIONS,
    HORIZON_USERS_DEV,
    HORIZON_USERS_PROD,
    MAX_REVIEWS,
)
from agent_service.agent_quality_worker.models import HorizonTabs, HorizonUser
from agent_service.endpoints.models import (
    AgentQC,
    HorizonCriteria,
    HorizonCriteriaOperator,
    ScoreRating,
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
    assign_args = {
        "agent_id": agent_id,
        "plan_id": plan_id,
        "user_id": await db.get_agent_owner(agent_id),
    }
    assign_message = {
        "method": "assign_reviewers_agent_qc",
        "arguments": assign_args,
        "send_time_utc": get_now_utc().isoformat(),
    }
    status_args = {
        "agent_id": agent_id,
        "plan_id": plan_id,
        "status": status.value,
    }
    status_message = {
        "method": "update_status_agent_qc",
        "arguments": status_args,
        "send_time_utc": get_now_utc().isoformat(),
    }
    queue = sqs.get_queue_by_name(QueueName=AGENT_QUALITY_WORKER_QUEUE)
    queue.send_message(MessageBody=json.dumps(assign_message, default=json_serial))
    queue.send_message(MessageBody=json.dumps(status_message, default=json_serial))


async def get_number_of_assigned(
    user_id: str, db: AsyncDB, reviewer_column: str, num_days_lookback: int
) -> int:
    # Get the current date and calculate the date 7 days ago
    now = get_now_utc()
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


def get_users_with_least_assigned_counts(
    users: List[HorizonUser],
    user_assigned_counts: dict[str, int],
    tier_allocations: Optional[dict[int, float]] = None,
    max_allowed: int = MAX_REVIEWS,
) -> List[HorizonUser]:
    # List to hold the count of assigned tickets for each user_id with count under 150
    eligible_users: List[dict] = []

    # Holds a map from a users tier to the count of all users in that tier group
    tier_groups: dict[int, int] = defaultdict(int)
    for user in users:
        tier_groups[user.tier] += 1

    # If we dont get provided an allocation mapping setting to empty dict will
    # use the default allocations in the loop below
    if tier_allocations is None:
        tier_allocations = {}

    # Loop over each user_id and get the count using get_number_of_assigned
    for user in users:
        # The allocation is defaulted to 1/(# unique tiers)
        user_allocation_ratio = tier_allocations.get(user.tier, 1.0 / len(tier_groups))
        users_in_group = tier_groups[user.tier]
        user_id = user.userId
        # User assignments are defaulted to 0 if not provided
        count = user_assigned_counts.get(user_id, 0)

        # Only add to dictionary the count per user is under 150 / by length * allocation_ratio
        # allocation_ratio is specified per user as a fraction of the total limit of 150 / users
        if count < max(1, int(ceil(max_allowed / users_in_group * user_allocation_ratio))):
            bisect.insort(eligible_users, {"user": user, "count": count}, key=lambda x: x["count"])

    if len(eligible_users) == 0:
        return []

    fewest_assigned = eligible_users[0]["count"]
    return [user["user"] for user in eligible_users if user["count"] == fewest_assigned]


async def assign_agent_quality_reviewers(
    agent_id: str, plan_id: str, user_id: str, skip_db_commit: bool = False
) -> None:
    db = AsyncDB(pg=SyncBoostedPG(skip_commit=skip_db_commit))

    logging.info(f"Retrieving Agent QC for agent_id: {agent_id}, plan_id: {plan_id}")

    agent_qcs = await db.get_agent_qc_by_agent_ids(agent_ids=[agent_id])

    if len(agent_qcs) != 1:
        logging.info(f"Agent QC not found for {agent_id=}")
        return None

    agent_qc = agent_qcs.pop()

    if agent_qc.is_spoofed:
        logging.info(f"Agent is spoofed, skipping assignment for agent_id: {agent_id}")
        return None

    if agent_qc.cs_reviewer or agent_qc.eng_reviewer or agent_qc.prod_reviewer:
        logging.info(
            f"Agent has already been assigned, skipping assignment for agent_id: {agent_id}"
        )
        return None

    env = get_environment_tag()
    is_dev = env in (DEV_TAG, LOCAL_TAG)
    is_user_internal = await db.is_user_internal(user_id)

    # Check if the user is internal and if the env is prod
    # we want to stop internal queries from getting to triage
    if not is_dev and is_user_internal:
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
            cs_team_members.append(horizon_user)
        elif horizon_user.userType == HorizonTabs.ENG.value:
            eng_team_members.append(horizon_user)
        elif horizon_user.userType == HorizonTabs.PROD.value:
            reviewer_team_members.append(horizon_user)

    cs_user_counts = {
        user.userId: await get_number_of_assigned(
            user_id=user.userId, db=db, reviewer_column=CS_REVIEWER_COLUMN, num_days_lookback=7
        )
        for user in cs_team_members
    }

    cs_team_members = get_users_with_least_assigned_counts(
        users=cs_team_members,
        user_assigned_counts=cs_user_counts,
        tier_allocations=CS_TIERED_ASSIGNMENT_ALLOCATIONS,
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
        agent_qc_id=agent_qc.agent_qc_id,
        agent_id=agent_id,
        user_id=user_id,
        agent_status=agent_qc.agent_status,
        plan_id=plan_id,
        cs_reviewer=cs_reviewer.userId,
        eng_reviewer=eng_reviewer.userId,
        prod_reviewer=prod_reviewer.userId,
        last_updated=datetime.datetime.now(),
        # this field can't be updated but should be False here regardless
        is_spoofed=False,
    )

    await db.update_agent_qc(agent_qc)
    logging.info(f"Successfully assigned reviewers for agent_id: {agent_id}")


async def update_agent_qc_status(
    agent_id: str, plan_id: str, status: str, skip_db_commit: bool = False
) -> None:
    db = AsyncDB(pg=SyncBoostedPG(skip_commit=skip_db_commit))

    logging.info(f"Retrieving Agent QC for agent_id: {agent_id}, plan_id: {plan_id}")

    agent_qcs = await db.get_agent_qc_by_agent_ids(agent_ids=[agent_id])

    if len(agent_qcs) != 1:
        logging.info(f"Agent QC not found for {agent_id=}")
        return None

    agent_qc = agent_qcs.pop()

    if agent_qc.score_rating == ScoreRating.BROKEN:
        logging.info("Agent QC already scored as BROKEN, skipping status update")
        return None

    agent_qc = AgentQC(
        agent_qc_id=agent_qc.agent_qc_id,
        agent_id=agent_id,
        user_id=agent_qc.user_id,
        agent_status=status,
        plan_id=plan_id,
        last_updated=datetime.datetime.now(),
        is_spoofed=agent_qc.is_spoofed,
    )

    if status == Status.ERROR:
        logging.info("Agent status is ERROR, setting agent QC to BROKEN, skipping CS review")

        agent_qc = AgentQC(
            agent_qc_id=agent_qc.agent_qc_id,
            agent_id=agent_id,
            user_id=agent_qc.user_id,
            agent_status=status,
            plan_id=plan_id,
            last_updated=datetime.datetime.now(),
            is_spoofed=agent_qc.is_spoofed,
            cs_reviewer="",
            score_rating=ScoreRating.BROKEN,
            cs_reviewed=True,
        )

    await db.update_agent_qc(agent_qc)
    logging.info(f"Successfully updated agent status for agent id: {agent_id}")
