import argparse
import asyncio
import datetime
import json
import logging
import signal
import traceback
from datetime import timedelta
from enum import Enum
from typing import Any, Dict

import boto3

from agent_service.planner.errors import AgentExecutionError
from agent_service.sqs_serve.message_handler import MessageHandler
from agent_service.utils.constants import AGENT_RUN_EXECUTION_PLAN_QUEUE, BOOSTED_DAG_QUEUE
from agent_service.utils.date_utils import get_now_utc
from agent_service.utils.event_logging import log_event
from agent_service.utils.postgres import Postgres
from agent_service.utils.s3_upload import download_json_from_s3

LOGGER = logging.getLogger(__name__)
DEFAULT_IDLE_TTL = 10
EVENT_NAME = "agent_launch_sqs_handler_replacement"


class MessageTypes(Enum):
    MESSAGE_SENT = "message_sent"
    MESSAGE_PROCESSED = "message_processed"


class GracefulSigterm:
    def __init__(self) -> None:
        self.kill_now = False
        signal.signal(signal.SIGTERM, self.exit_gracefully)

    def exit_gracefully(self, *args: Any) -> None:
        self.kill_now = True


def replace_and_set_metadata(
    sqs: Any,
    queue_url: str,
    message_id: str,
    parent_message_id: str,
    idle_ttl: int,
    spawn_time: datetime.date,
    event_name: str,
    message: Dict = {},
) -> None:
    """
    Sends a replacement SQS message and logs the event.
    Also writes to PG that this job is no longer idle
    """
    # Send SQS message to self replace
    boosted_dag_message = {
        "method": "AgentServiceKickoffSQSHandler",
        "idle_ttl": idle_ttl,
        "message_id": message_id,
        "parent_message_id": parent_message_id,
    }
    sqs.send_message(
        QueueUrl=queue_url,
        DelaySeconds=1,
        MessageBody=json.dumps(boosted_dag_message),
    )

    # Update that this job is no longer idle, add in the extra info
    # For tracking if we have it.
    upsert = {
        "message_id": message_id,
        "parent_message_id": parent_message_id,
        "listening": False,
        "agent_id": message.get("agent_id"),
        "plan_id": message.get("plan_id"),
        "plan_run_id": message.get("plan_run_id"),
        "user_id": message.get("user_id"),
        "message": json.dumps(message),
    }
    pg = Postgres()
    pg.generic_insert_or_update(
        table_name="agent.sqs_handler_logs",
        values_to_insert=upsert,
        conflict_columns=["message_id"],
        columns_to_update=list(upsert.keys()),
    )

    log_event(
        event_name=event_name,
        event_data={
            "start_time_utc": spawn_time.isoformat(),
            "end_time_utc": get_now_utc().isoformat(),
            "message": boosted_dag_message,
            "type": MessageTypes.MESSAGE_SENT.value,
            "message_id": message_id,
        },
    )


async def poll_sqs_for_time(
    message_id: str, parent_message_id: str, idle_ttl: int, outbound_queue: str
) -> None:
    """
    Listener job for run_execution_plan messages. Spins up a replacement when a message is received.
    Else will exit after the idle time to live period passes, also spinning up a replacement before
    exiting.
    """
    queue_url = AGENT_RUN_EXECUTION_PLAN_QUEUE
    message_handler = MessageHandler()

    if not queue_url:
        raise Exception("Environment variable AGENT_RUN_EXECUTION_PLAN_QUEUE must be set.")
    sqs = boto3.client("sqs", region_name="us-west-2")

    LOGGER.info(f"Now live, SQS listener job with id: {message_id}, parent id: {parent_message_id}")
    LOGGER.info(f"Listening to queue {queue_url}")
    graceful_sigterm = GracefulSigterm()

    spawn_time = get_now_utc()

    boosted_dag_queue = sqs.get_queue_url(QueueName=outbound_queue)
    boosted_dag_queue_url = boosted_dag_queue["QueueUrl"]

    while not graceful_sigterm.kill_now:
        # Check if idle_ttl has been exceeded, if so spawn a replacement and exit
        if (get_now_utc() - spawn_time) > timedelta(minutes=idle_ttl):
            LOGGER.info(f"No messages received for {idle_ttl} minutes. Exiting due to idle TTL.")
            replace_and_set_metadata(
                sqs,
                boosted_dag_queue_url,
                message_id,
                parent_message_id,
                idle_ttl,
                spawn_time,
                EVENT_NAME,
            )
            exit(0)

        messages = sqs.receive_message(
            QueueUrl=queue_url,
            AttributeNames=["All"],
            MessageAttributeNames=["All"],
            VisibilityTimeout=60 * 60 * 4,  # 4 hours
            MaxNumberOfMessages=1,
            WaitTimeSeconds=20,
        )

        if not messages or "Messages" not in messages:
            continue

        for message in messages["Messages"]:
            sqs.delete_message(QueueUrl=queue_url, ReceiptHandle=message["ReceiptHandle"])
            sqs_message = message["Body"]
            LOGGER.info(f"Received Message: {sqs_message}")
            message_dict = json.loads(sqs_message)

            # First, before processing send a message to boosted-dag to create a replacement
            LOGGER.info(f"Creating replacement workflow to queue: {queue_url}")
            replace_and_set_metadata(
                sqs,
                boosted_dag_queue_url,
                message_id,
                parent_message_id,
                idle_ttl,
                spawn_time,
                EVENT_NAME,
                message=message_dict,
            )

            start_time_utc = get_now_utc().isoformat()
            converted_message_str = sqs_message
            if "s3_path" in message_dict:
                converted_message_str = download_json_from_s3(message_dict["s3_path"])
                message_dict = json.loads(converted_message_str)
            try:
                await message_handler.handle_message(message_dict)
                log_event(
                    event_name="agent_worker_message_processed",
                    event_data={
                        "start_time_utc": start_time_utc,
                        "end_time_utc": get_now_utc().isoformat(),
                        "raw_message": sqs_message,
                        "message": converted_message_str,
                        "type": MessageTypes.MESSAGE_PROCESSED.value,
                        "message_id": message_id,
                    },
                )
            except AgentExecutionError as e:
                log_event(
                    event_name="agent_worker_message_processed",
                    event_data={
                        "start_time_utc": start_time_utc,
                        "end_time_utc": get_now_utc().isoformat(),
                        "raw_message": sqs_message,
                        "message": converted_message_str,
                        "error_msg": traceback.format_exc(),
                        "message_id": message_id,
                    },
                )
                if e.alert_on_error:
                    LOGGER.exception("Encountered exception processing message")
            except Exception:
                log_event(
                    event_name="agent_worker_message_processed",
                    event_data={
                        "start_time_utc": start_time_utc,
                        "end_time_utc": get_now_utc().isoformat(),
                        "raw_message": sqs_message,
                        "message": converted_message_str,
                        "error_msg": traceback.format_exc(),
                        "message_id": message_id,
                    },
                )
                LOGGER.exception("Encountered exception processing message")

            LOGGER.info(f"Message Processed: {sqs_message}")
            exit(0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create an SQS Handler for run_execution_plan agent jobs"
    )
    parser.add_argument(
        "--message-id",
        type=str,
        required=True,
        help="Internal ID of the message",
    )
    parser.add_argument(
        "--parent-message-id",
        type=str,
        default="none",
        help="Internal ID of the parent of the current message",
    )
    parser.add_argument(
        "--idle-ttl", type=int, default=10, help="How long to run without receiving a message"
    )
    parser.add_argument(
        "--queue",
        type=str,
        default=BOOSTED_DAG_QUEUE,
        help="Queue to send the self replacement sqs messages",
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s.%(funcName)s:%(lineno)d - %(levelname)s - %(message)s",
        force=True,
    )
    args = parse_args()
    asyncio.run(
        poll_sqs_for_time(args.message_id, args.parent_message_id, args.idle_ttl, args.queue)
    )
