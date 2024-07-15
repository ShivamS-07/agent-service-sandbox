import argparse
import asyncio
import datetime
import json
import logging
import resource
import time
import traceback

from agent_service.sqs_serve.message_handler import MessageHandler
from agent_service.utils.event_logging import log_event
from agent_service.utils.s3_upload import download_json_from_s3


async def process_message_string(message_string: str) -> None:
    message_dict = json.loads(message_string)
    message_handler = MessageHandler()
    await message_handler.handle_message(message=message_dict)


# TODO (Tommy): When we run messages as pods on Argo, the pod will be deleted as soon as the task
# completes. This is normally a good thing, but the downside is that the stdout logs get deleted
# immediately as well. Because we are relying on Vector to process stdout logs and importantly
# send events logged with log_event to Clickhouse, we need to make sure that Vector has enough
# time to process all of the logs. This is a hack to accomplish this - in the meantime,
# Tommy will look into how we can make Argo not automatically delete pods, or at least wait a minute
def wait_if_needed(start_time: float) -> None:
    while (time.time() - start_time) < 75:
        time.sleep(5)


async def main() -> None:
    message_string = ""
    start_time_utc = datetime.datetime.utcnow().isoformat()
    start_time = time.time()
    try:
        parser = argparse.ArgumentParser()

        parser.add_argument(
            "-m",
            "--message",
            type=str,
        )
        parser.add_argument(
            "-s",
            "--s3-path",
            type=str,
        )
        args = parser.parse_args()
        if not args.message and not args.s3_path:
            raise Exception("You must set either --message or --s3-path")
        message_string = (
            args.message if args.message else download_json_from_s3(s3_path=args.s3_path)
        )
        await process_message_string(message_string=message_string)
        log_event(
            event_name="agent_worker_message_processed",
            event_data={
                "start_time_utc": start_time_utc,
                "end_time_utc": datetime.datetime.utcnow().isoformat(),
                "message": message_string,
            },
        )
        wait_if_needed(start_time=start_time)
        with open("/exit_script/memory.txt", "w") as f:
            total_mem = (
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                + resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
            )
            f.write(str(total_mem))

    except Exception as e:
        exception_text = traceback.format_exc()
        with open("/exit_script/exception.txt", "w") as f:
            f.write(exception_text)
        with open("/exit_script/memory.txt", "w") as f:
            total_mem = (
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                + resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
            )
            f.write(str(total_mem))
        log_event(
            event_name="agent_worker_message_processed",
            event_data={
                "start_time_utc": start_time_utc,
                "end_time_utc": datetime.datetime.utcnow().isoformat(),
                "message": message_string,
                "error_msg": traceback.format_exc(),
            },
        )
        wait_if_needed(start_time=start_time)
        raise e


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s.%(funcName)s:%(lineno)d - %(levelname)s - %(message)s",
        force=True,
    )
    asyncio.run(main())
