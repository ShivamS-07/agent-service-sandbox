"""
This script was needed because some agents were failing on the recommendation step
due to the client not being subscribed to any strategy. Currently, all clients get subscribed
to at least one strategy when they are onboarded, but this wasn't the case before, so we had
to subscribe the older ones to stop the issue from happening.
Jira ticket:
https://gradientboostedinvestments.atlassian.net/browse/QL24-2967?atlOrigin=eyJpIjoiNjI0NjdmOTViMjU5NDc1OWEzOWMwMmU1NTU4MmE3ZDUiLCJwIjoiaiJ9  # noqa
"""

import asyncio
import logging
from typing import List, Tuple

from gbi_common_py_utils.utils.environment import (
    PROD_TAG,
    get_environment_tag,
)

from agent_service.external.pa_svc_client import (
    get_all_authorized_live_strategy_ids,
    subscribe_to_marketplace_strategy,
)
from agent_service.utils.postgres import get_psql

logger = logging.getLogger(__name__)


async def get_active_users() -> List[str]:
    # TODO: we might have to switch this in the future to hit any users that are old but get
    # TODO: but get upgraded to alfa later (run the SQL without "WHERE has_alfa_access")
    sql = """
    SELECT DISTINCT(id::TEXT) FROM user_service.users WHERE last_accessed IS NOT NULL and has_alfa_access
    """
    res = get_psql().generic_read(sql)
    return [x["id"] for x in res]


async def is_user_subscribed_to_at_least_one_strategy(user_id: str) -> bool:
    return len(await get_all_authorized_live_strategy_ids(user_id=user_id)) > 0


async def get_name_email_team_for_user_id(user_id: str) -> Tuple[str, str, str]:
    sql = "SELECT name, email FROM user_service.users WHERE id = %s"
    res = get_psql().generic_read(sql, [user_id])
    if not res:
        return "Name not found", "Email not found", "Team not found"
    name, email = res[0]["name"], res[0]["email"]

    sql = "SELECT team_id::TEXT FROM team_membership WHERE user_id = %s"
    res = get_psql().generic_read(sql, [user_id])
    if not res:
        return name, email, "Team not found"
    team_id = res[0]["team_id"]

    sql = "SELECT name FROM user_service.teams WHERE id = %s"
    res = get_psql().generic_read(sql, [team_id])
    if not res:
        return name, email, "Team not found"
    team_name = res[0]["name"]

    return name, email, team_name


async def subscribe() -> None:
    users = await get_active_users()
    for user_id in users:
        try:
            if await is_user_subscribed_to_at_least_one_strategy(user_id):
                print(f"User {user_id} is already migrated!")
                continue
        except Exception as e:
            print(repr(e))
            print(f"User {user_id} is likely inactive")
            continue
        try:
            await subscribe_to_marketplace_strategy(
                user_id=user_id, is_prod=get_environment_tag() == PROD_TAG
            )
            username, email, team = await get_name_email_team_for_user_id(user_id)
            print(
                f"Subscribed (User ID: {user_id}, Username: {username}, Email: {email}, Team Name: {team}) to the default strategy!"
            )  # noqa
        except Exception as e:
            print(repr(e))


if __name__ == "__main__":
    asyncio.run(subscribe())
