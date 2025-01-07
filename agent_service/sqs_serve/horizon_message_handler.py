import asyncio
import logging
from typing import Dict

from agent_service.agent_quality_worker.ingestion_worker import (
    assign_agent_quality_reviewers,
    update_agent_qc_status,
)
from agent_service.utils.date_utils import get_now_utc


class HorizonMessageHandler:
    async def handle_message(self, message: Dict) -> Dict:
        """
        assign_args = {
            "agent_id": agent_id,
            "plan_id": plan_id,
            "user_id": agent_owner
        }
        assign_message = {
            "method": "assign_reviewers_agent_qc",
            "arguments": assign_args,
            "send_time_utc": get_now_utc().isoformat(),
        }
        status_args = {
            "agent_id": agent_id,
            "plan_id": plan_id,
            "user_id": agent_owner
            "status": Status.value (ERROR, CANCELLED, NOT_STARTED, etc.)
        }
        status_message = {
            "method": "update_status_agent_qc",
            "arguments": status_args,
            "send_time_utc": get_now_utc().isoformat(),
        }

        Returns:

        """
        method = message.get("method")
        args = message["arguments"]
        if method == "assign_reviewers_agent_qc":
            await assign_agent_quality_reviewers(**args)
            return message
        elif method == "update_status_agent_qc":
            await update_agent_qc_status(**args)
            return message
        else:
            raise NotImplementedError(f"Method {method} is not supported for Horizon")


if __name__ == "__main__":
    LOGGER = logging.getLogger(__name__)

    arguments = {
        "agent_id": "b90dde4a-92a0-4a94-84bc-8885d3f9054a",
        "plan_id": "3ee2a1b8-3fcf-4f0c-991e-433a41153d6a",
        "user_id": "3a2eaf66-3d4f-4f9f-b9eb-dbe15972c894",
    }
    agent_qc_message = {
        "method": "assign_reviewers_agent_qc",
        "arguments": arguments,
        "send_time_utc": get_now_utc().isoformat(),
    }
    message_handler = HorizonMessageHandler()
    asyncio.run(message_handler.handle_message(agent_qc_message))

    arguments = {
        "agent_id": "b90dde4a-92a0-4a94-84bc-8885d3f9054a",
        "plan_id": "3ee2a1b8-3fcf-4f0c-991e-433a41153d6a",
        "status": "BROKEN",
    }
    agent_qc_message = {
        "method": "update_status_agent_qc",
        "arguments": arguments,
        "send_time_utc": get_now_utc().isoformat(),
    }
    asyncio.run(message_handler.handle_message(agent_qc_message))
