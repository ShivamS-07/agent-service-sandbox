import json
from typing import Dict

from prefect import tags

from agent_service.planner.executor import create_execution_plan, run_execution_plan
from agent_service.utils.s3_upload import download_json_from_s3


class MessageHandler:
    async def handle_message(self, message: Dict) -> None:
        if "s3_path" in message:
            message = json.loads(download_json_from_s3(message["s3_path"]))
        method = message.get("method")
        arguments = message["arguments"]
        if method == "run_execution_plan":
            agent_id = arguments["context"]["agent_id"]
            with tags(agent_id):
                await run_execution_plan(**arguments)
        elif method == "create_execution_plan":
            agent_id = arguments["agent_id"]
            with tags(agent_id):
                await create_execution_plan(**arguments)
        else:
            raise NotImplementedError(f"Method {method} is not supported")
