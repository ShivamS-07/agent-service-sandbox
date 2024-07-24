# Please keep these tests to read-only until skip-commit behavior is figured out
# this is (currently) just to make sure basic functions work
# and no broken db code or dependencies are committed

import unittest

from agent_service.utils.async_db import AsyncDB
from agent_service.utils.async_postgres_base import AsyncPostgresBase


class AsyncPostgresTest(unittest.IsolatedAsyncioTestCase):
    async def test_generic_read(self):
        pg = self.pg
        failures_sql = """
        select * from boosted_dag.audit_log al
        limit 100
        """

        rows = await pg.generic_read(failures_sql)
        self.assertEqual(100, len(rows))

    async def test_get_agents(self):
        user_id = "f6fe6a54-c15c-4893-9909-90657be7f19f"  # DG
        await self.db.get_existing_agents_names(user_id)

    @classmethod
    def setUpClass(cls) -> None:
        # WARNING: this is capable of writing to db permanently
        # so stick with read-only tests for now
        cls.pg = AsyncPostgresBase()
        cls.db = AsyncDB(cls.pg)

    @classmethod
    async def tearDownClass(cls) -> None:
        await cls.pg.close()
        cls.pg = None
