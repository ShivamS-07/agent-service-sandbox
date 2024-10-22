import json
import os
from typing import Any, Dict

import boto3

if __name__ == "__main__":
    BOOSTED_DAG_QUEUE = os.getenv("BOOSTED_DAG_QUEUE", "insights-backend-dev-boosted-dag")
    msg: Dict[str, Any] = {"AgentServiceSecFilingsIngestion": {}}
    sqs = boto3.resource("sqs", region_name="us-west-2")
    queue = sqs.get_queue_by_name(QueueName=BOOSTED_DAG_QUEUE)
    response = queue.send_message(MessageBody=json.dumps(msg), DelaySeconds=0)
