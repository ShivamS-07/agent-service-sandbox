import asyncio
import datetime
from typing import List

from agent_service.types import Message, Notification
from agent_service.utils.async_db import AsyncDB
from agent_service.utils.logs import async_perf_logger

UNREAD_UPDATES_MESSAGE_FIRST_LINE = "Hello! Here is a list of previous reports you have missed:"
UNREAD_UPDATES_MESSAGE_TEMPLATE = UNREAD_UPDATES_MESSAGE_FIRST_LINE + "\n{unread_updates}"


# GREETING_MESSAGE_TEMPLATE = (
#     "Hello! Here is a summary of the new updates since your last visit:"
#     "\n{unread_updates}"
#     "\n{latest_report}"
#     "Let me know if you have any questions or need further information."
# )

# GREETING_MESSAGE_NOUPDATE_TEMPLATE = (
#     "Hey! No new updates yet since your last visit. "
#     "Let me know if you have any questions or need further information. "
#     "\n{latest_report}"
# )


# @async_perf_logger
# async def gen_unread_updates_summary(
#     unread_updates: List[Message], last_plan_run_id: Optional[str]
# ) -> str:
#     if last_plan_run_id:
#         latest_report_markdown = (
#             "Here is the last report if you need it: "
#             + "```"
#             + json.dumps(
#                 {
#                     "type": "output_greeting",
#                     "text": "last report",
#                     "plan_run_id": last_plan_run_id,
#                 }
#             )
#             + "```"
#         )
#     else:
#         latest_report_markdown = ""

#     if len(unread_updates) == 0:
#         updates_summary = GREETING_MESSAGE_NOUPDATE_TEMPLATE.format(
#             latest_report=latest_report_markdown
#         )
#     else:
#         unread_updates_formated = "\n\n".join(
#             [
#                 str(update.message).replace(
#                     "with important changes found:", f"on {update.message_time.date()}:"
#                 )
#                 for update in unread_updates
#             ]
#         )
#         updates_summary = GREETING_MESSAGE_TEMPLATE.format(
#             unread_updates=unread_updates_formated,
#             latest_report=latest_report_markdown,
#         )
#     return updates_summary


@async_perf_logger
async def add_unread_updates_message_to_chat_history(
    chat_context_messages: List[Message], unread_updates: List[Message], agent_id: str, pg: AsyncDB
) -> List[Message]:
    """Add unread updates message to a single message at the beginning of the chat history."""
    if len(unread_updates) == 0:
        return chat_context_messages

    unread_updates_formated = "\n\n".join([str(update.message) for update in unread_updates])
    updates_summary = UNREAD_UPDATES_MESSAGE_TEMPLATE.format(unread_updates=unread_updates_formated)
    # update message_time of updates_summary_message to be 3 microseconds before the min message_time in chat_context_messages
    message_time = min(
        [message.message_time for message in chat_context_messages]
    ) - datetime.timedelta(microseconds=3)
    updates_summary_message = Message(
        agent_id=agent_id,
        message=updates_summary,
        is_user_message=False,
        message_time=message_time,
        unread=True,
    )
    notification = Notification(
        agent_id=updates_summary_message.agent_id,
        message_id=updates_summary_message.message_id,
        summary=updates_summary,
        unread=True,
    )
    await asyncio.gather(
        pg.insert_notifications(notifications=[notification]),
        pg.insert_chat_messages(
            messages=[updates_summary_message],
        ),
    )
    # add updates_summary_message to the beginning of the chat history
    chat_context_messages.insert(0, updates_summary_message)
    return chat_context_messages
