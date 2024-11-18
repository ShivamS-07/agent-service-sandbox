import argparse
import asyncio
import json
import logging
import resource
import time
import traceback

from gbi_common_py_utils.utils.environment import DEV_TAG, PROD_TAG, get_environment_tag

from agent_service.endpoints.models import Status
from agent_service.planner.errors import AgentExecutionError
from agent_service.slack.slack_sender import SlackSender, get_user_info_slack_string
from agent_service.sqs_serve.message_handler import MessageHandler
from agent_service.tools.output import EMPTY_OUTPUT_FLAG
from agent_service.utils.async_db import AsyncDB
from agent_service.utils.async_postgres_base import AsyncPostgresBase
from agent_service.utils.clickhouse import Clickhouse
from agent_service.utils.date_utils import get_now_utc
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


class NoOutputException(AgentExecutionError):
    result_status = Status.NO_RESULTS_FOUND


async def main() -> None:
    message_string = ""
    start_time_utc = get_now_utc().isoformat()
    start_time = time.time()
    converted_message_str = ""
    message_dict = None
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
        message_dict = json.loads(message_string)
        message_handler = MessageHandler()
        converted_message_str = message_string
        if "s3_path" in message_dict:
            converted_message_str = download_json_from_s3(message_dict["s3_path"])
            message_dict = json.loads(converted_message_str)
        await message_handler.handle_message(message=message_dict)
        log_event(
            event_name="agent_worker_message_processed",
            event_data={
                "start_time_utc": start_time_utc,
                "end_time_utc": get_now_utc().isoformat(),
                "raw_message": message_string,
                "message": converted_message_str,
            },
            force=True,
        )
        if EMPTY_OUTPUT_FLAG.get():
            raise NoOutputException()
        wait_if_needed(start_time=start_time)
        with open("/exit_script/memory.txt", "w") as f:
            total_mem = (
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                + resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
            )
            f.write(str(total_mem))

    except Exception as e:
        if message_dict and "arguments" in message_dict:
            if "scheduled_by_automation" in message_dict["arguments"]:
                message_context = message_dict["arguments"]["context"]
                agent_id = message_context["agent_id"]
                user_id = message_context["user_id"]
                env = get_environment_tag()
                channel = f"client-live-agent-failures-{'prod' if env == PROD_TAG else 'dev'}"
                base_url = f"https://{'alfa' if env == PROD_TAG else 'agent-dev'}.boosted.ai"
                pg = AsyncPostgresBase()
                async_db = AsyncDB(pg)
                user_email, user_info_slack_string = await get_user_info_slack_string(
                    async_db, user_id
                )

                top_level_message = "LIVE AGENT FAILURE\n"
                if isinstance(e, NoOutputException):
                    top_level_message = "EMPTY OUTPUT FOR LIVE AGENT:\n"
                agent_name = await async_db.get_agent_name(agent_id=agent_id)
                if env == DEV_TAG or (
                    not user_email.endswith("@boosted.ai")
                    and not user_email.endswith("@gradientboostedinvestments.com")
                ):
                    ch = Clickhouse()
                    last_login = await ch.get_last_login_for_user(user_id=user_id)
                    message_text = (
                        f"{top_level_message}Agent Name: {agent_name}\n"
                        f"link: {base_url}/chat/{agent_id}\n"
                        f"{user_info_slack_string}\n"
                        f"Last Login (NYC Time): {last_login.isoformat() if last_login else 'N/A'}"
                    )
                    slack_sender = SlackSender(channel=channel)
                    slack_sender.send_message(
                        message_text=message_text, send_at=int(time.time()) + 60
                    )
        log_event(
            event_name="agent_worker_message_processed",
            event_data={
                "start_time_utc": start_time_utc,
                "end_time_utc": get_now_utc().isoformat(),
                "raw_message": message_string,
                "message": converted_message_str,
                "error_msg": traceback.format_exc(),
            },
            force=True,
        )
        exception_text = traceback.format_exc()
        with open("/exit_script/exception.txt", "w") as f:
            f.write(exception_text)
        with open("/exit_script/memory.txt", "w") as f:
            total_mem = (
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                + resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
            )
            f.write(str(total_mem))
        wait_if_needed(start_time=start_time)
        if isinstance(e, AgentExecutionError) and not e.alert_on_error:
            return
        raise e


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s.%(funcName)s:%(lineno)d - %(levelname)s - %(message)s",
        force=True,
    )
    asyncio.run(main())
