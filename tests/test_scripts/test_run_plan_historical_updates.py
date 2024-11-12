import datetime
import logging
from unittest import IsolatedAsyncioTestCase

import agent_service.utils.date_utils as date_utils
from agent_service.utils.logs import init_stdout_logging
from scripts.run_plan_historical_updates import run_plan_historical_updates

date_utils.disable_mock_time()

logger = logging.getLogger(__name__)


class TestRunPlanHistoricalUpdates(IsolatedAsyncioTestCase):
    async def test_main(self):
        init_stdout_logging()
        # "ibm's sector"
        # https://agent-dev.boosted.ai/chat/b2a3de66-4790-4374-af66-98c0dd08c0f9/worklog
        try:
            await run_plan_historical_updates(
                agent_id="b2a3de66-4790-4374-af66-98c0dd08c0f9",
                skip_commit=False,
                start_date=datetime.date(2024, 11, 9),
                end_date=datetime.date(2024, 11, 10),
            )
        finally:
            date_utils.disable_mock_time()
