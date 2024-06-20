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
           "plan":{
              "nodes":[
                 {
                    "tool_name":"get_stock_universe",
                    "tool_task_id":"dc1fd5a5-835c-4353-b94f-ea0a1a74e4be",
                    "args":{
                       "universe_name":"S&P 500"
                    },
                    "description":"Get the list of S&P 500 stocks",
                    "output_variable_name":"stock_ids",
                    "is_output_node":false,
                    "store_output":true
                 },
                 {
                    "tool_name":"get_stock_recommendations",
                    "tool_task_id":"2ecc404f-fb7c-46f5-9ac9-b3078179ccb5",
                    "args":{
                       "stock_ids":{
                          "var_name":"stock_ids"
                       },
                       "filter":true,
                       "buy":true,
                       "num_stocks_to_return":1
                    },
                    "description":"Get the top recommended stock from the S&P 500",
                    "output_variable_name":"recommended_stock",
                    "is_output_node":false,
                    "store_output":true
                 },
                 {
                    "tool_name":"prepare_output",
                    "tool_task_id":"a542593a-062a-4de6-9ef1-09461a30ef8a",
                    "args":{
                       "object_to_output":{
                          "var_name":"recommended_stock"
                       },
                       "title":"Top Recommended S&P 500 Stock"
                    },
                    "description":"Output the top recommended stock",
                    "output_variable_name":"output",
                    "is_output_node":true,
                    "store_output":false
                 }
              ]
           },
           "context":{
              "agent_id":"1298b4ea-ca8b-4d62-af03-2df2b0c13cc5",
              "plan_id":"3c9c0714-0842-4a61-9101-5df5effd794c",
              "user_id":"6c14fe54-de50-4d05-9533-57541715064f",
              "plan_run_id":"919ac09c-7cee-4efd-9764-98dbd4455c1b",
              "chat":{
                 "messages":[
                    {
                       "agent_id":"1298b4ea-ca8b-4d62-af03-2df2b0c13cc5",
                       "message_id":"97b14200-1276-4c2d-a1d8-eac71d716c44",
                       "message":"What is the best S&P 500 stock?",
                       "is_user_message":true,
                       "message_time":"2024-06-19T18:01:03.574113+00:00",
                       "unread":false
                    },
                    {
                       "agent_id":"1298b4ea-ca8b-4d62-af03-2df2b0c13cc5",
                       "message_id":"abaddb83-98b5-4b29-94c8-bd020b7175ff",
                       "message":"Understood, you're looking for insights on the top-performing S&P 500 stock. I'll start considering how to approach this for you.",
                       "is_user_message":false,
                       "message_time":"2024-06-19T18:01:05.959762+00:00",
                       "unread":false
                    }
                 ]
              },
              "task_id":null,
              "skip_db_commit":false,
              "skip_task_cache":false,
              "run_tasks_without_prefect":false
           },
           "do_chat":true
        }
        """
        arguments = json.loads(raw_message)
        arguments["context"]["plan_run_id"] = str(uuid.uuid4())
        arguments["context"]["skip_db_commit"] = True
        message = {"method": "run_execution_plan", "arguments": arguments}
        self.handle_message(message=message)
