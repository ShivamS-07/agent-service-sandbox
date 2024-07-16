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
           {"method": "create_execution_plan", "arguments": {"agent_id": "fb0400ef-c7d5-4d8e-8817-ad489e29a24e", "plan_id": "806cda01-8329-451a-8a58-974953f04cd6", "user_id": "6c14fe54-de50-4d05-9533-57541715064f", "skip_db_commit": false, "skip_task_cache": false, "run_plan_in_prefect_immediately": true, "action": "REPLAN", "error_info": null}}
            """
        message = json.loads(raw_message)
        self.handle_message(message=message)

    def test_run_execution_plan(self):
        raw_message = """
{
  "plan": {
    "nodes": [
      {
        "tool_name": "get_stock_universe",
        "tool_task_id": "e9519b1c-9a40-4b08-a426-862a72cde647",
        "args": {
          "universe_name": "Russell 1000"
        },
        "description": "Get the list of stocks in the Russell 1000",
        "output_variable_name": "stock_ids",
        "is_output_node": false,
        "store_output": true
      },
      {
        "tool_name": "get_statistic_data_for_companies",
        "tool_task_id": "d3c7dba7-62fe-47ba-9f7c-eab69450a657",
        "args": {
          "statistic_reference": "P/E ratio",
          "stock_ids": {
            "var_name": "stock_ids"
          }
        },
        "description": "Get P/E ratios for the Russell 1000 stocks",
        "output_variable_name": "pe_ratios",
        "is_output_node": false,
        "store_output": true
      },
      {
        "tool_name": "transform_table",
        "tool_task_id": "2e318d62-a4e5-4512-b622-8fe714f8be55",
        "args": {
          "input_table": {
            "var_name": "pe_ratios"
          },
          "transformation_description": "Filter to P/E ratios less than 25.5"
        },
        "description": "Filter to stocks with P/E ratios less than 25.5",
        "output_variable_name": "filtered_pe_ratios",
        "is_output_node": false,
        "store_output": true
      },
      {
        "tool_name": "get_stock_identifier_list_from_table",
        "tool_task_id": "bd993483-5f9d-4697-813b-68959385d10b",
        "args": {
          "input_table": {
            "var_name": "filtered_pe_ratios"
          }
        },
        "description": "Extract the list of stock IDs from the filtered table",
        "output_variable_name": "filtered_stock_ids",
        "is_output_node": false,
        "store_output": true
      },
      {
        "tool_name": "prepare_output",
        "tool_task_id": "ab910b2c-355e-4ab2-a690-f6c502d6ea9e",
        "args": {
          "object_to_output": {
            "var_name": "filtered_stock_ids"
          },
          "title": "Companies in the Russell 1000 with P/E ratios less than 25.5"
        },
        "description": "Output the list of companies",
        "output_variable_name": "output",
        "is_output_node": true,
        "store_output": false
      }
    ]
  },
  "context": {
    "agent_id": "1a4e844d-b8d3-4251-a5ab-407880c61008",
    "plan_id": "0872bc45-3b17-4b75-9a8e-b7b2b335bf57",
    "user_id": "ac7c96d7-3e57-40e7-a1a5-8e2ce5e23639",
    "plan_run_id": "78379d44-9871-4729-b90b-e4cf9a789185",
    "chat": {
      "messages": [
        {
          "agent_id": "1a4e844d-b8d3-4251-a5ab-407880c61008",
          "message_id": "3d95e1a4-688d-470b-b8f8-ca8ff222ba01",
          "message": "What companies in the R1K have P/E ratios less than 25.5?",
          "is_user_message": true,
          "message_time": "2024-07-05T14:45:20.967093+00:00",
          "unread": false,
          "visible_to_llm": true
        },
        {
          "agent_id": "1a4e844d-b8d3-4251-a5ab-407880c61008",
          "message_id": "aa1c0339-e688-4174-8b93-97e4b3b5f20a",
          "message": "Understood, you're looking for Russell 1000 companies with P/E ratios under 25.5. I'll start figuring out how to gather this information for you.",
          "is_user_message": false,
          "message_time": "2024-07-05T14:45:22.312060+00:00",
          "unread": true,
          "visible_to_llm": true
        }
      ]
    },
    "task_id": null,
    "skip_db_commit": false,
    "skip_task_cache": false,
    "run_tasks_without_prefect": false
  },
  "do_chat": true
}
        """
        arguments = json.loads(raw_message)
        arguments["context"]["plan_run_id"] = str(uuid.uuid4())
        arguments["context"]["skip_db_commit"] = True
        message = {"method": "run_execution_plan", "arguments": arguments}
        self.handle_message(message=message)
