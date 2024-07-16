import asyncio
import datetime
import json
import logging
import signal
import traceback
from typing import Any

import boto3

from agent_service.sqs_serve.message_handler import MessageHandler
from agent_service.utils.constants import AGENT_WORKER_QUEUE
from agent_service.utils.event_logging import log_event
from agent_service.utils.gpt_input_output_logger import json_serial

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

            start_time_utc = datetime.datetime.utcnow().isoformat()
            converted_message = {}
            try:
                converted_message = message_handler.convert_message(message=message_dict)
                await message_handler.handle_message(converted_message)
                log_event(
                    event_name="agent_worker_message_processed",
                    event_data={
                        "start_time_utc": start_time_utc,
                        "end_time_utc": datetime.datetime.utcnow().isoformat(),
                        "raw_message": sqs_message,
                        "message": json.dumps(converted_message, default=json_serial),
                    },
                )
            except Exception:
                log_event(
                    event_name="agent_worker_message_processed",
                    event_data={
                        "start_time_utc": start_time_utc,
                        "end_time_utc": datetime.datetime.utcnow().isoformat(),
                        "raw_message": sqs_message,
                        "message": json.dumps(converted_message, default=json_serial),
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
