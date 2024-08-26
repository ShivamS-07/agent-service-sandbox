import traceback

import requests
from gbi_common_py_utils.utils.ssm import get_param

from agent_service.utils.event_logging import log_event


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
