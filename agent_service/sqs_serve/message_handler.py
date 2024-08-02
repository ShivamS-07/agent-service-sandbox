from typing import Dict

from prefect import tags

from agent_service.io_type_utils import load_io_type_dict
from agent_service.planner.executor import create_execution_plan, run_execution_plan


class MessageHandler:

    async def handle_message(self, message: Dict) -> Dict:
        method = message.get("method")
        arguments = message["arguments"]
        if method == "run_execution_plan":
            agent_id = arguments["context"]["agent_id"]
            with tags(agent_id):
                override_task_output_lookup = arguments.get("override_task_output_lookup", None)
                if override_task_output_lookup:
                    for k, v in override_task_output_lookup.items():
                        override_task_output_lookup[k] = load_io_type_dict(v)
                await run_execution_plan(**arguments)
                return message
        elif method == "create_execution_plan":
            agent_id = arguments["agent_id"]
            with tags(agent_id):
                await create_execution_plan(**arguments)
                return message
        else:
            raise NotImplementedError(f"Method {method} is not supported")
