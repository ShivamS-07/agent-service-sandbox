# type: ignore
# flake8: noqa
import json
import os

os.environ["PREFECT_API_URL"] = "http://prefect-dev.boosted.ai:4200/api"
import asyncio
import unittest
import uuid
from typing import Dict

from agent_service.sqs_serve.message_handler import MessageHandler


class TestSQSMessageHandler(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.loop = asyncio.get_event_loop()  # type: ignore[assignment]
        cls.message_handler = MessageHandler()

    def handle_message(self, message: Dict):
        self.loop.run_until_complete(self.message_handler.handle_message(message=message))

    def test_create_execution_plan(self):
        raw_message = """
        {
  "action": "CREATE",
  "do_chat": true,
  "error_info": null,
  "chat_context": null,
  "skip_db_commit": true,
  "skip_task_cache": false,
  "run_tasks_without_prefect": false,
  "run_plan_in_prefect_immediately": true,
  "plan_id": "34a47450-5340-4d88-ad62-06a11c7463a4",
  "user_id": "ac7c96d7-3e57-40e7-a1a5-8e2ce5e23639",
  "agent_id": "870a70cd-2aa4-4e09-af73-026bb86ac8e2"
}
        """
        message = {"method": "create_execution_plan", "arguments": json.loads(raw_message)}
        self.handle_message(message=message)

    def test_create_execution_plan_rewrite(self):
        raw_message = """
           {"method": "create_execution_plan", "arguments": {"agent_id": "e9bd2ab2-16c3-4934-b372-c39dd4dd7f7a", "plan_id": "08b9174b-4321-40cd-98d5-af968e58bacc", "user_id": "ac7c96d7-3e57-40e7-a1a5-8e2ce5e23639", "skip_db_commit": false, "skip_task_cache": false, "run_plan_in_prefect_immediately": true, "action": "REPLAN", "error_info": null}}
            """
        message = json.loads(raw_message)
        self.handle_message(message=message)

    def test_run_execution_plan(self):
        raw_message = """
        {
  "do_chat": true,
  "log_all_outputs": false,
  "replan_execution_error": true,
  "run_plan_in_prefect_immediately": true,
  "plan": {
    "nodes": [
      {
        "args": {
          "universe_name": "S&P 500"
        },
        "tool_name": "get_stock_universe",
        "description": "Get the list of S&P 500 stocks",
        "tool_task_id": "5b7c0f96-6276-4f5a-b354-a49651415900",
        "is_output_node": false,
        "output_variable_name": "sp500_stocks"
      },
      {
        "args": {
          "buy": true,
          "horizon": "1M",
          "stock_ids": {
            "var_name": "sp500_stocks"
          },
          "news_horizon": "1W",
          "delta_horizon": "1M",
          "num_stocks_to_return": 1
        },
        "tool_name": "get_recommended_stocks",
        "description": "Get the top recommended stock to buy from the S&P 500",
        "tool_task_id": "667d1dd9-1a12-471b-85a0-c59a040380c8",
        "is_output_node": true,
        "output_variable_name": "top_stock"
      }
    ]
  },
  "context": {
    "chat": {
      "messages": [
        {
          "message": "What is the best S&P 500 stock?",
          "agent_id": "870a70cd-2aa4-4e09-af73-026bb86ac8e2",
          "message_id": "b9f54c6d-9155-4232-81f2-95aa9e1c2972",
          "message_time": "2024-06-04T20:03:11.005202+00:00",
          "is_user_message": true
        },
        {
          "message": "Understood, you're looking for the top-performing S&P 500 stock. I'm on it, considering the best approach to identify that for you.",
          "agent_id": "870a70cd-2aa4-4e09-af73-026bb86ac8e2",
          "message_id": "9b475298-a765-44c3-8fd3-ba5dca614d33",
          "message_time": "2024-06-04T20:03:12.686646+00:00",
          "is_user_message": false
        }
      ]
    },
    "plan_id": "34a47450-5340-4d88-ad62-06a11c7463a4",
    "task_id": null,
    "user_id": "ac7c96d7-3e57-40e7-a1a5-8e2ce5e23639",
    "agent_id": "870a70cd-2aa4-4e09-af73-026bb86ac8e2",        
    "skip_task_cache": false,
    "run_tasks_without_prefect": false
  }
}
        """
        arguments = json.loads(raw_message)
        arguments["context"]["plan_run_id"] = str(uuid.uuid4())
        arguments["context"]["skip_db_commit"] = True
        message = {"method": "run_execution_plan", "arguments": arguments}
        self.handle_message(message=message)
