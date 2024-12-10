import logging
import unittest

import agent_service.utils.date_utils as date_utils
from agent_service.utils.logs import init_stdout_logging
from scripts.run_plan_run_locally import run_plan_run_id_task_id

date_utils.disable_mock_time()

logger = logging.getLogger(__name__)


class TestRunPlanRunLocally(unittest.IsolatedAsyncioTestCase):
    # @unittest.skip("Flaky on jenkins, works locally")
    async def test_main(self):
        init_stdout_logging()
        # "ibm's sector"
        # https://agent-dev.boosted.ai/chat/b2a3de66-4790-4374-af66-98c0dd08c0f9/worklog
        try:
            res = await run_plan_run_id_task_id(
                plan_run_id="edd214f9-d5b9-46c2-b323-a71351ca6134",
                start_with_task_id="edf910ff-4628-46c0-9930-b589c76f29c5",
                env="DEV",
            )

            logger.info(f"{res=}")
        finally:
            date_utils.disable_mock_time()
