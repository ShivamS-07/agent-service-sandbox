import logging
import os
from typing import Dict

from dateutil.parser import parse

from agent_service.planner.executor import run_execution_plan
from agent_service.planner.plan_creation import create_execution_plan
from agent_service.utils.date_utils import get_now_utc
from agent_service.utils.event_logging import log_event

logger = logging.getLogger(__name__)


class MessageHandler:
    async def handle_message(self, message: Dict) -> Dict:
        method = message.get("method")
        arguments = message["arguments"]
        no_cache = message.get("no_gpt_cache")
        if no_cache:
            os.environ["NO_GPT_CACHE"] = "1"
        pod_start = os.getenv("STARTED_AT")
        pod_start_dt = None
        if pod_start:
            try:
                pod_start_dt = parse(pod_start)
            except Exception:
                logger.exception("Unable to parse STARTED_AT env var")
        now = get_now_utc()
        if method == "run_execution_plan":
            log_event(
                "agent-svc-run-execution-plan-start",
                event_data={
                    "pod_start": pod_start_dt,
                    "execution_start": now,
                    "arguments": arguments,
                },
            )
            await run_execution_plan(**arguments)
            return message
        elif method == "create_execution_plan":
            log_event(
                "agent-svc-create-execution-plan-start",
                event_data={
                    "pod_start": pod_start_dt,
                    "execution_start": now,
                    "arguments": arguments,
                },
            )
            await create_execution_plan(**arguments)
            return message
        else:
            raise NotImplementedError(f"Method {method} is not supported")
