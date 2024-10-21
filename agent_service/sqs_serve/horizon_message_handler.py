import asyncio
import logging
from typing import Dict

from agent_service.agent_quality_worker.ingestion_worker import (
    assign_agent_quality_reviewers,
)
from agent_service.utils.date_utils import get_now_utc


class HorizonMessageHandler:

    async def handle_message(self, message: Dict) -> Dict:
        """
        arguments = {
            "agent_id": agent_id,
            "plan_id": plan_id,
            "user_id": user_id
            "status": what stage of the quality triage we are at,
        }
        message = {
            "method": "agent_quality",
            "arguments": arguments,
            "send_time_utc": get_now_utc().isoformat(),
        }

        Returns:

        """
        method = message.get("method")
        arguments = message["arguments"]
        if method == "agent_quality":
            await assign_agent_quality_reviewers(**arguments)
            return message
        else:
            raise NotImplementedError(f"Method {method} is not supported for Horizon")


if __name__ == "__main__":

    LOGGER = logging.getLogger(__name__)

    arguments = {
        "agent_id": "3423ae35-5270-4688-afbb-5dd151a8f396",
        "plan_id": "ef17469c-d69f-4044-9fc6-3cdeec6787a7",
        "user_id": "a5d534c9-5426-4387-a298-723c5e09ecab",  # william
        "status": "CS",
    }
    agent_qc_message = {
        "method": "agent_quality",
        "arguments": arguments,
        "send_time_utc": get_now_utc().isoformat(),
    }
    message_handler = HorizonMessageHandler()
    asyncio.run(message_handler.handle_message(agent_qc_message))
