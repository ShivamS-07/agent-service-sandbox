from typing import Dict

from prefect import tags

from agent_service.planner.executor import create_execution_plan, run_execution_plan


class MessageHandler:

    async def handle_message(self, message: Dict) -> Dict:
        method = message.get("method")
        arguments = message["arguments"]
        if method == "run_execution_plan":
            agent_id = arguments["context"]["agent_id"]
            with tags(agent_id):
                await run_execution_plan(**arguments)
                return message
        elif method == "create_execution_plan":
            agent_id = arguments["agent_id"]
            with tags(agent_id):
                await create_execution_plan(**arguments)
                return message
        else:
            raise NotImplementedError(f"Method {method} is not supported")
