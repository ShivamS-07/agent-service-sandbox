import logging
from typing import List, Optional, cast

import boto3
from gbi_common_py_utils.utils.environment import PROD_TAG, get_environment_tag
from user_service_proto_v1.user_service_pb2 import User

from agent_service.endpoints.models import (
    AgentNotificationBody,
    AgentNotificationData,
    AgentSubscriptionMessage,
    ForwardingEmailMessage,
    HelpRequestResolvedEmailMessage,
    OnboardingEmailMessage,
    PlanRunFinishEmailMessage,
)
from agent_service.external.user_svc_client import get_user_cached
from agent_service.planner.planner_types import ExecutionPlan
from agent_service.utils.async_db import AsyncDB
from agent_service.utils.constants import NOTIFICATION_SERVICE_QUEUE

logger = logging.getLogger(__name__)


class AgentEmail:
    def __init__(self, db: Optional[AsyncDB] = None) -> None:
        self.queue = boto3.resource("sqs", region_name="us-west-2").get_queue_by_name(
            QueueName=NOTIFICATION_SERVICE_QUEUE
        )
        self.db = db

    async def send_agent_emails(
        self,
        agent_id: str,
        email_subject: str,
        plan_run_id: str,
        run_summary_short: str,
        run_summary_long: str,
    ) -> None:
        """
        Args:
            agent_id: The agent id to retrieve the owner for
            plan_run_id: The plan run id
            run_summary_short: Short summary of run
            run_summary_long: Long Summary of run (What's new section on report)

        Returns: None this function checks at the completion of a plan and checks
        if the agent has email subscriptions and sends a message to the notification service
        to send the email
        """
        if not self.db:
            logger.warning("No db provided, skipping email sending")
            return

        # Get agent Information
        agent_name = await self.db.get_agent_name(agent_id=agent_id)
        agent_owner = await self.db.get_agent_owner(agent_id=agent_id)
        agent_subs = await self.db.get_agent_subscriptions(agent_id=agent_id)

        # if we have any agent subscriptions then send an sqs message
        # Do not automatically include the agent owner in the emailing list
        if agent_subs:
            # share the plan
            await self.db.set_plan_run_share_status(plan_run_id=plan_run_id, status=True)

            # create a subscription message
            detailed_email_user_id_pairs = set()
            for agent_sub in agent_subs:
                if agent_sub.user_id:
                    detailed_email_user_id_pairs.add((agent_sub.user_id, agent_sub.email))
                else:
                    detailed_email_user_id_pairs.add(("", agent_sub.email))

            detailed_message = AgentSubscriptionMessage(
                user_id_email_pairs=list(detailed_email_user_id_pairs),
                agent_data=[
                    AgentNotificationData(
                        agent_name=agent_name,
                        agent_id=agent_id,
                        email_subject=email_subject if email_subject else agent_name,
                        plan_run_id=plan_run_id,
                        agent_owner=agent_owner if agent_owner else "",
                        notification_body=AgentNotificationBody(
                            summary_title=run_summary_short if run_summary_short else "",
                            summary_body=run_summary_long if run_summary_long else "",
                        ),
                    )
                ],
            )

            self.queue.send_message(MessageBody=detailed_message.model_dump_json())

    async def send_welcome_email(self, user_id: str) -> None:
        """Sends welcome email to new users"""
        user = cast(User, await get_user_cached(user_id=user_id, async_db=self.db))

        # This is HubSpot ID for the welcome email we're sending
        WELCOME_EMAIL_ID = 181851687081

        message = OnboardingEmailMessage(
            user_name=user.name,
            user_id=user_id,
            email=user.email,
            hubspot_email_id=WELCOME_EMAIL_ID,
        )

        self.queue.send_message(MessageBody=message.model_dump_json())

    def forward_existing_email(self, notification_key: str, recipient_email: str) -> None:
        """Forwards existing emails in production environment"""
        if get_environment_tag() == PROD_TAG:
            message = ForwardingEmailMessage(
                notification_key=notification_key,
                recipient_email=recipient_email,
            )
            self.queue.send_message(MessageBody=message.model_dump_json())

    async def send_plan_run_finish_email(
        self, agent_id: str, short_summary: str, output_titles: List[str]
    ) -> None:
        """Sends a plan run finish email to the agent owner"""
        if not self.db:
            logger.warning("No db provided, skipping email sending")
            return

        agent_name = await self.db.get_agent_name(agent_id=agent_id)
        agent_owner = await self.db.get_agent_owner(agent_id=agent_id)

        # Limit the number of titles to 3 so the email doesn't get too long
        if len(output_titles) > 3:
            output_titles_str = ", ".join(output_titles[:3]) + ", etc"
        else:
            output_titles_str = ", ".join(output_titles)

        message = PlanRunFinishEmailMessage(
            agent_id=agent_id,
            agent_owner=agent_owner if agent_owner else "",
            agent_name=agent_name,
            short_summary=short_summary,
            output_titles=output_titles_str,
        )
        self.queue.send_message(MessageBody=message.model_dump_json())

    async def send_help_request_resolved_email(self, agent_id: str) -> None:
        """Sends a help request resolved email to the agent owner"""
        if not self.db:
            logger.warning("No db provided, skipping email sending")
            return

        agent_name = await self.db.get_agent_name(agent_id=agent_id)
        agent_owner = await self.db.get_agent_owner(agent_id=agent_id)
        outputs = await self.db.get_agent_outputs_data_from_db(
            agent_id=agent_id, include_output=False
        )

        if not outputs:
            logger.warning(f"No outputs found for agent {agent_id}")
            return

        exeuction_plan = ExecutionPlan.model_validate(outputs[0]["plan"])
        output_titles: List[str] = [
            node.args["title"] for node in exeuction_plan.nodes if node.is_output_node  # type: ignore
        ]

        # Limit the number of titles to 3 so the email doesn't get too long
        if len(output_titles) > 3:
            output_titles_str = ", ".join(output_titles[:3]) + ", etc"
        else:
            output_titles_str = ", ".join(output_titles)

        message = HelpRequestResolvedEmailMessage(
            agent_id=agent_id,
            agent_owner=agent_owner if agent_owner else "",
            agent_name=agent_name,
            short_summary=outputs[0]["run_metadata"]["run_summary_short"],
            output_titles=output_titles_str,
        )
        self.queue.send_message(MessageBody=message.model_dump_json())
