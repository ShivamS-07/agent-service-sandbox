import traceback
from typing import Any, Optional, Tuple

import aiohttp
import requests
from gbi_common_py_utils.utils.ssm import get_param

from agent_service.external.user_svc_client import get_user_cached
from agent_service.utils.async_db import AsyncDB
from agent_service.utils.event_logging import log_event


async def get_user_info_slack_string(pg: AsyncDB, user_id: str) -> Tuple[str, str]:
    user_info = await get_user_cached(user_id=user_id, async_db=pg)
    if not user_info:
        raise ValueError(f"User not found: {user_id}")

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
    def __init__(self, channel: str) -> None:
        self.channel = channel
        self.auth_token = get_param("/alpha/slack/api_token")
        self.session: Optional[aiohttp.ClientSession] = None

    def send_message(
        self,
        message_text: str,
        send_at: Optional[int] = None,
        channel_override: Optional[str] = None,
    ) -> None:
        if channel_override is not None:
            channel = channel_override
        else:
            channel = self.channel

        try:
            url = "https://slack.com/api/chat.scheduleMessage"
            if send_at is None:
                url = "https://slack.com/api/chat.postMessage"

            headers = {
                "Content-type": "application/json",
                "Authorization": f"Bearer {self.auth_token}",
            }

            data: dict[str, Any] = {
                "channel": channel,
                "text": message_text,
            }
            if send_at is not None:
                data["post_at"] = send_at

            response = requests.post(url, json=data, headers=headers)
            log_event(
                event_name="agent-service-slack-message-sent",
                event_data={
                    "message_text": message_text,
                    "send_at": send_at,
                    "channel": channel,
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
                    "channel": channel,
                },
            )

    async def send_message_async(
        self,
        message_text: str,
        send_at: Optional[int] = None,
        channel_override: Optional[str] = None,
    ) -> None:
        if self.session is None:
            self.session = aiohttp.ClientSession()

        if channel_override is not None:
            channel = channel_override
        else:
            channel = self.channel

        try:
            if send_at is None:
                url = "https://slack.com/api/chat.postMessage"
            else:
                url = "https://slack.com/api/chat.scheduleMessage"

            headers = {
                "Content-type": "application/json",
                "Authorization": f"Bearer {self.auth_token}",
            }

            data: dict[str, Any] = {
                "channel": channel,
                "text": message_text,
            }
            if send_at is not None:
                data["post_at"] = send_at

            async with self.session.post(url, json=data, headers=headers) as response:
                response_json = await response.json()
                log_event(
                    event_name="agent-service-slack-message-sent",
                    event_data={
                        "message_text": message_text,
                        "send_at": send_at,
                        "channel": channel,
                        "response_json": response_json,
                    },
                )

        except Exception:
            log_event(
                event_name="agent-service-slack-message-sent",
                event_data={
                    "message_text": message_text,
                    "send_at": send_at,
                    "error_msg": traceback.format_exc(),
                    "channel": channel,
                },
            )
