import json
import re
from typing import List, cast

from agent_service.planner.constants import FOLLOW_UP_QUESTION, REPORT_UPDATED_LINE
from agent_service.types import Message
from agent_service.utils.async_db import AsyncDB
from agent_service.utils.get_agent_outputs import LOGGER


def append_widget_chip_to_report_updated_message(chat_message: str, plan_run_id: str | None) -> str:
    # get the first two words
    report_updated_text = REPORT_UPDATED_LINE
    report_updated_dict = {
        "type": "output_report",
        "text": report_updated_text,
        "plan_run_id": plan_run_id,
    }
    return chat_message.replace(
        report_updated_text,
        "```" + json.dumps(report_updated_dict) + "```",
    )


async def append_widget_chip_to_execution_complete_message(
    message: Message, agent_id: str, db: AsyncDB
) -> str:
    chat_message = cast(str, message.message)

    if not message.message_metadata or not message.message_metadata.output_title_task_id_pairs:
        LOGGER.warning(f"No message metadata or output title task id pairs for {agent_id=}")
        return chat_message

    task_ids = list(message.message_metadata.output_title_task_id_pairs.values())
    task_id_mapping = await db.get_latest_id_metadata_by_task_ids(
        agent_id=agent_id,
        task_ids=task_ids,
    )

    # regex to find titles in quotation marks
    pattern = r'"(.*?)"'
    matches = list(re.finditer(pattern, chat_message))

    # Process matches in reverse order to avoid messing up string indices when substituting
    for match in reversed(matches):
        title = match.group(1)
        task_id = message.message_metadata.output_title_task_id_pairs[title]
        widget_chip_dict = {
            "type": "output_widget",
            "name": title,
            "output_id": task_id_mapping[task_id]["output_id"],
            "plan_run_id": task_id_mapping[task_id]["plan_run_id"],
        }

        # Replace the matched text with the widget chip
        chat_message = (
            chat_message[: match.start()]
            + f"```{json.dumps(widget_chip_dict)}```"
            + chat_message[match.end() :]
        )
    return chat_message


async def append_widget_chip_to_list_of_chat_messages(
    messages: List[Message], agent_id: str, pg: AsyncDB
) -> List[Message]:
    for message in messages:
        chat_message = cast(str, message.message)
        if not message.is_user_message and chat_message.startswith(REPORT_UPDATED_LINE):
            message.message = append_widget_chip_to_report_updated_message(
                chat_message=chat_message, plan_run_id=message.plan_run_id
            )
        if not message.is_user_message and chat_message.endswith(FOLLOW_UP_QUESTION):
            message.message = await append_widget_chip_to_execution_complete_message(
                message=message, agent_id=agent_id, db=pg
            )
    return messages
