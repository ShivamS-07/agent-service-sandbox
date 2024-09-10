import traceback
from typing import Tuple

import requests
from gbi_common_py_utils.utils.ssm import get_param

from agent_service.external.user_svc_client import get_users
from agent_service.utils.async_db import AsyncDB
from agent_service.utils.event_logging import log_event


async def get_user_info_slack_string(pg: AsyncDB, user_id: str) -> Tuple[str, str]:
    user_info = (
        await get_users(  # TODO: maybe we should cache it in memory?
            user_id=user_id, user_ids=[user_id], include_user_enabled=False
        )
    )[0]
    org_id = user_info.organization_membership.organization_id.id
    org_name = await pg.get_org_name(org_id=org_id)

    user_info_slack_string = ""
    user_info_slack_string += f"cognito_username: {user_info.cognito_username}"
    user_info_slack_string += f"\nname: {user_info.name}"
    user_info_slack_string += f"\nuser_id: {user_id}"
    user_info_slack_string += f"\norganization_name: {org_name}"
    user_info_slack_string += f"\norganization_id: {org_id}"
    return user_info.email, user_info_slack_string


class SlackSender:
    def __init__(self, channel: str):
        self.channel = channel
        self.auth_token = get_param("/alpha/slack/api_token")

    def send_message_at(self, message_text: str, send_at: int) -> None:
        try:
            url = "https://slack.com/api/chat.scheduleMessage"

            headers = {
                "Content-type": "application/json",
                "Authorization": f"Bearer {self.auth_token}",
            }

            data = {
                "channel": self.channel,
                "text": message_text,
                "post_at": send_at,
            }

            response = requests.post(url, json=data, headers=headers)
            log_event(
                event_name="agent-service-slack-message-sent",
                event_data={
                    "message_text": message_text,
                    "send_at": send_at,
                    "channel": self.channel,
                    "response_json": response.json(),
                },
            )

        except Exception:
            log_event(
                event_name="agent-service-slack-message-sent",
                event_data={
                    "message_text": message_text,
                    "send_at": send_at,
                    "error_msg": traceback.format_exc(),
                    "channel": self.channel,
                },
            )
