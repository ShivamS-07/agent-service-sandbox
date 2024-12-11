import unittest

from gbi_common_py_utils.utils.environment import PROD_TAG

from agent_service.planner.planner import ExecutionPlan
from agent_service.tool import default_tool_registry
from agent_service.tools import *  # noqa
from agent_service.utils.postgres import Postgres


class TestLivePlanTools(unittest.TestCase):
    def test_prod_live_plan_tool_resolution(self):
        """
        Make sure that tools are not removed that are used in live plans on prod.
        """
        db = Postgres(skip_commit=True, environment=PROD_TAG)

        sql = """
        SELECT DISTINCT ON (a.agent_id) a.agent_id, plan FROM agent.execution_plans ep
        JOIN agent.agents a ON a.agent_id = ep.agent_id
        WHERE a.automation_enabled
        ORDER BY a.agent_id, ep.last_updated DESC
        """

        rows = db.generic_read(sql)

        plans = [(row["agent_id"], ExecutionPlan.model_validate(row["plan"])) for row in rows]
        tools = default_tool_registry()

        for agent_id, plan in plans:
            with self.subTest(agent_id):
                for step in plan.nodes:
                    tools.get_tool(step.tool_name)
