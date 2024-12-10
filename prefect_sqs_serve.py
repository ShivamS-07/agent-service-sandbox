import asyncio
import json
import logging
import signal
import traceback
from typing import Any

import boto3

from agent_service.planner.errors import AgentExecutionError
from agent_service.sqs_serve.message_handler import MessageHandler
from agent_service.utils.constants import AGENT_WORKER_QUEUE
from agent_service.utils.date_utils import get_now_utc
from agent_service.utils.event_logging import log_event
from agent_service.utils.s3_upload import download_json_from_s3

LOGGER = logging.getLogger(__name__)


class GracefulSigterm:
    def __init__(self) -> None:
        self.kill_now = False
        signal.signal(signal.SIGTERM, self.exit_gracefully)

    def exit_gracefully(self, *args: Any) -> None:
        self.kill_now = True


async def poll_sqs_forever() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s.%(funcName)s:%(lineno)d - %(levelname)s - %(message)s",
        force=True,
    )
    queue_url = AGENT_WORKER_QUEUE
    message_handler = MessageHandler()

    if not queue_url:
        raise Exception("Environment variable AGENT_WORKER_QUEUE must be set.")
    sqs = boto3.client("sqs", region_name="us-west-2")

    LOGGER.info(f"Listening to queue {queue_url}")
    graceful_sigterm = GracefulSigterm()
    while not graceful_sigterm.kill_now:
        messages = sqs.receive_message(
            QueueUrl=queue_url,
            AttributeNames=["All"],
            MessageAttributeNames=["All"],
            VisibilityTimeout=60 * 60 * 4,  # 4 hours to be safe
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
                    },
                )
                LOGGER.exception("Encountered exception processing message")

            LOGGER.info(f"Message Processed: {sqs_message}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s.%(funcName)s:%(lineno)d - %(levelname)s - %(message)s",
        force=True,
    )
    asyncio.run(poll_sqs_forever())
