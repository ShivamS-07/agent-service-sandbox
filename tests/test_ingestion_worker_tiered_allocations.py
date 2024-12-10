from unittest import IsolatedAsyncioTestCase

from agent_service.agent_quality_worker.ingestion_worker import (
    get_users_with_least_assigned_counts,
)
from agent_service.agent_quality_worker.models import HorizonTabs, HorizonUser

CS_TEAM: list[HorizonUser] = [
    HorizonUser(
        userId="a",
        name="Simon",
        userType=HorizonTabs.CS,
        tier=1,
    ),
    HorizonUser(
        userId="b",
        name="simon-test-user",
        userType=HorizonTabs.CS,
        tier=1,
    ),
    HorizonUser(
        userId="c",
        name="Richard",
        userType=HorizonTabs.CS,
        tier=2,
    ),
]

TIER_DICT: dict[int, float] = {1: 0.8, 2: 0.2}


class TestIngestionWorkerTieredAllocations(IsolatedAsyncioTestCase):
    async def test_allusers_maxed(self):
        counts = {"a": 40, "b": 40, "c": 20}
        eligible_users = get_users_with_least_assigned_counts(
            users=CS_TEAM, user_assigned_counts=counts, tier_allocations=TIER_DICT, max_allowed=100
        )
        self.assertEqual(len(eligible_users), 0)

    async def test_take_fewest(self):
        counts = {"a": 39, "b": 39, "c": 19}
        eligible_users = get_users_with_least_assigned_counts(
            users=CS_TEAM, user_assigned_counts=counts, tier_allocations=TIER_DICT, max_allowed=100
        )
        # picks c since it has the fewest total tickets
        self.assertEqual(len(eligible_users), 1)
        self.assertEqual(eligible_users[0].userId, "c")

    async def test_roundup(self):
        counts = {
            "a": 39,  # 39.2 is cutoff
            "b": 39,  # 39.2 is cutoff
            "c": 19,  # 19.6 is cutoff
        }  # all should roundup
        eligible_users = get_users_with_least_assigned_counts(
            users=CS_TEAM, user_assigned_counts=counts, tier_allocations=TIER_DICT, max_allowed=98
        )

        # picks c since it has the fewest total tickets
        self.assertEqual(len(eligible_users), 1)
        self.assertEqual(eligible_users[0].userId, "c")

    async def test_tie(self):
        counts = {"a": 10, "b": 10, "c": 19}
        eligible_users = get_users_with_least_assigned_counts(
            users=CS_TEAM, user_assigned_counts=counts, tier_allocations=TIER_DICT, max_allowed=100
        )

        # a,b tie
        self.assertEqual(len(eligible_users), 2)
        eligible_users_id = [user.userId for user in eligible_users]
        self.assertIn("a", eligible_users_id)
        self.assertIn("b", eligible_users_id)
