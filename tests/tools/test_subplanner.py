import os
import unittest
from unittest.mock import AsyncMock, patch

from agent_service.io_types.table import StockTable
from agent_service.planner.planner_types import ExecutionPlan
from agent_service.tools.subplanner import PerRowProcessingArgs, per_row_processing
from agent_service.types import PlanRunContext


class TestSubPlanner(unittest.IsolatedAsyncioTestCase):
    @unittest.skip("Expensive test")
    @patch("agent_service.planner.planner.Planner")
    async def test_per_row_processing(self, mock_planner):
        with open(os.path.dirname(__file__) + "/data/subplanner/plan.json") as f:
            plan = ExecutionPlan.model_validate_json(f.read())

        with open(os.path.dirname(__file__) + "/data/subplanner/args.json") as f:
            args = PerRowProcessingArgs.model_validate_json(f.read())

        mock_planner.return_value.create_subplan = AsyncMock()
        mock_planner.return_value.create_subplan.return_value = plan
        result = await per_row_processing(args=args, context=PlanRunContext.get_dummy())
        self.assertIsInstance(result, StockTable)
