import asyncio
import logging
from typing import List, Optional, Union, cast
from uuid import uuid4

import boto3
from gbi_common_py_utils.utils.environment import PROD_TAG, get_environment_tag
from user_service_proto_v1.user_service_pb2 import User

from agent_service.agent_quality_worker.ingestion_worker import (
    send_agent_quality_message,
)
from agent_service.endpoints.models import (
    AgentEvent,
    AgentNameEvent,
    AgentNotificationBody,
    AgentNotificationData,
    AgentOutput,
    AgentQuickThoughtsEvent,
    AgentSubscriptionMessage,
    EventType,
    ExecutionPlanTemplate,
    ExecutionStatusEvent,
    ForwardingEmailMessage,
    MessageEvent,
    NewPlanEvent,
    NotificationEvent,
    NotifyHelpStatusEvent,
    NotifyMessageEvent,
    OnboardingEmailMessage,
    OutputEvent,
    PlanRunTaskLog,
    PlanStatusEvent,
    PlanTemplateTask,
    QuickThoughts,
    Status,
    TaskStatus,
    TaskStatusEvent,
)
from agent_service.endpoints.utils import get_plan_preview
from agent_service.external.user_svc_client import get_user_cached
from agent_service.io_type_utils import load_io_type
from agent_service.io_types.text import Text, TextOutput
from agent_service.planner.planner_types import ExecutionPlan, OutputWithID, PlanStatus
from agent_service.types import Message, Notification, PlanRunContext
from agent_service.utils.async_db import AsyncDB
from agent_service.utils.async_postgres_base import AsyncPostgresBase
from agent_service.utils.async_utils import run_async_background
from agent_service.utils.cache_utils import CacheBackend
from agent_service.utils.constants import NOTIFICATION_SERVICE_QUEUE
from agent_service.utils.date_utils import get_now_utc
from agent_service.utils.feature_flags import get_ld_flag, get_user_context
from agent_service.utils.output_utils.output_construction import get_output_from_io_type
from agent_service.utils.postgres import Postgres, SyncBoostedPG, get_psql
from agent_service.utils.redis_queue import (
    publish_agent_event,
    publish_notification_event,
)

logger = logging.getLogger(__name__)
sqs = boto3.resource("sqs", region_name="us-west-2")
queue = sqs.get_queue_by_name(QueueName=NOTIFICATION_SERVICE_QUEUE)


async def send_chat_message(
    message: Message,
    db: Optional[Union[AsyncDB, Postgres]] = None,
    insert_message_into_db: bool = True,
    skip_commit: bool = False,
    send_notification: bool = True,
) -> None:
    """
    Sends a chat message from GPT to the user. Updates the redis queue as well
    as the DB. If no DB instance is passed in, uses a SYNCHRONOUS DB instead.
    """
    logger.info(f"Sending message: {message.message}")
    if insert_message_into_db:
        if db and isinstance(db, AsyncDB):
            await db.insert_chat_messages(messages=[message])
        elif db and isinstance(db, Postgres):
            db.insert_chat_messages(messages=[message])
        else:
            get_psql(skip_commit=skip_commit).insert_chat_messages(messages=[message])

    if not message.is_user_message:
        if send_notification:
            # insert notification if not user message
            notification = Notification(
                agent_id=message.agent_id,
                message_id=message.message_id,
                summary=str(message.message),
            )
            if db and isinstance(db, AsyncDB):
                await db.insert_notifications(notifications=[notification])
                user_id = await db.get_agent_owner(agent_id=message.agent_id)
            elif db and isinstance(db, Postgres):
                db.insert_notifications(notifications=[notification])
                user_id = db.get_agent_owner(agent_id=message.agent_id)
            else:
                psql = get_psql(skip_commit=skip_commit)
                psql.insert_notifications(notifications=[notification])
                user_id = psql.get_agent_owner(agent_id=message.agent_id)

            # publish notification event if tied to a user ID
            if user_id:
                if db and isinstance(db, AsyncDB):
                    notification_event_info = await db.get_notification_event_info(
                        agent_id=message.agent_id
                    )
                elif db and isinstance(db, Postgres):
                    notification_event_info = db.get_notification_event_info(
                        agent_id=message.agent_id
                    )
                else:
                    notification_event_info = get_psql(
                        skip_commit=skip_commit
                    ).get_notification_event_info(agent_id=message.agent_id)

                if notification_event_info:
                    notification_event = NotificationEvent(
                        user_id=user_id,
                        event=NotifyMessageEvent(
                            agent_id=message.agent_id,
                            unread_count=notification_event_info["unread_count"],
                            latest_notification_string=notification_event_info[
                                "latest_notification_string"
                            ],
                        ),
                    )
                    await publish_notification_event(
                        user_id=user_id, serialized_event=notification_event.model_dump_json()
                    )

        event = AgentEvent(agent_id=message.agent_id, event=MessageEvent(message=message))
        await publish_agent_event(
            agent_id=message.agent_id, serialized_event=event.model_dump_json()
        )


async def update_agent_help_requested(
    agent_id: str, user_id: str, help_requested: bool, db: AsyncDB
) -> None:
    await db.update_agent_help_requested(agent_id=agent_id, help_requested=help_requested)
    event = NotificationEvent(
        user_id=user_id,
        event=NotifyHelpStatusEvent(agent_id=agent_id, is_help_requested=help_requested),
    )
    await publish_notification_event(user_id=user_id, serialized_event=event.model_dump_json())


async def publish_agent_name(agent_id: str, agent_name: str) -> None:
    """
    Sends a chat message from GPT to the user. Updates the redis queue as well
    as the DB. If no DB instance is passed in, uses a SYNCHRONOUS DB instead.
    """
    event = AgentEvent(agent_id=agent_id, event=AgentNameEvent(agent_name=agent_name))
    await publish_agent_event(agent_id=agent_id, serialized_event=event.model_dump_json())


async def publish_agent_quick_thoughts(
    agent_id: str, quick_thoughts: QuickThoughts, db: AsyncDB
) -> None:
    quick_thought_id = str(uuid4())
    await db.insert_quick_thought_for_agent(
        agent_id=agent_id, quick_thought_id=quick_thought_id, quick_thoughts=quick_thoughts
    )

    await publish_agent_event(
        agent_id=agent_id,
        serialized_event=AgentEvent(
            agent_id=agent_id,
            event=AgentQuickThoughtsEvent(quick_thoughts=quick_thoughts),
        ).model_dump_json(),
    )


async def publish_agent_output(
    outputs_with_ids: List[OutputWithID],
    context: PlanRunContext,
    live_plan_output: bool = False,
    is_intermediate: bool = False,
    db: Optional[Postgres] = None,
    is_locked: bool = False,
    cache_backend: Optional[CacheBackend] = None,
) -> None:
    if not db:
        db = get_psql()

    db.write_agent_multi_outputs(
        outputs_with_ids=outputs_with_ids,
        context=context,
        is_intermediate=is_intermediate,
        live_plan_output=live_plan_output,
    )

    async_pg = AsyncPostgresBase()
    coros = [get_output_from_io_type(o.output, pg=async_pg) for o in outputs_with_ids]
    results = await asyncio.gather(*coros)

    now = get_now_utc()
    event = AgentEvent(
        agent_id=context.agent_id,
        event=OutputEvent(
            output=[
                AgentOutput(
                    agent_id=context.agent_id,
                    output_id=o.output_id,
                    plan_id=context.plan_id,
                    plan_run_id=context.plan_run_id,
                    output=rich_output,  # type: ignore
                    is_intermediate=is_intermediate,
                    created_at=now,
                    shared=False,
                    live_plan_output=live_plan_output,
                    task_id=o.task_id,
                    dependent_task_ids=o.dependent_task_ids,
                    is_locked=is_locked,
                )
                for (rich_output, o) in zip(results, outputs_with_ids)
            ]
        ),
    )
    await publish_agent_event(agent_id=context.agent_id, serialized_event=event.model_dump_json())

    if cache_backend and not context.skip_db_commit:
        run_async_background(
            cache_backend.multiset(
                key_val_map={
                    o.output_id: rich_output for o, rich_output in zip(outputs_with_ids, results)
                },
                ttl=3600 * 24,
            )
        )


async def publish_agent_execution_plan(
    plan: ExecutionPlan, context: PlanRunContext, db: Optional[AsyncDB] = None
) -> None:
    if not db:
        db = AsyncDB(SyncBoostedPG(skip_commit=context.skip_db_commit))

    await db.write_execution_plan(
        plan_id=context.plan_id,
        agent_id=context.agent_id,
        plan=plan,
        status=PlanStatus.READY,
    )
    await db.update_plan_run(
        agent_id=context.agent_id, plan_id=context.plan_id, plan_run_id=context.plan_run_id
    )

    event = AgentEvent(
        agent_id=context.agent_id,
        event=NewPlanEvent(
            event_type=EventType.NEW_PLAN,
            plan=ExecutionPlanTemplate(
                plan_id=context.plan_id,
                upcoming_plan_run_id=context.plan_run_id,
                tasks=[
                    PlanTemplateTask(task_id=node.tool_task_id, task_name=node.description)
                    for node in plan.nodes
                ],
                preview=get_plan_preview(plan),
            ),
        ),
    )

    await publish_agent_event(agent_id=context.agent_id, serialized_event=event.model_dump_json())


async def publish_agent_plan_status(
    agent_id: str, plan_id: str, status: PlanStatus, db: AsyncDB
) -> None:
    await db.update_execution_plan_status(
        plan_id=plan_id,
        agent_id=agent_id,
        status=status,
    )
    await publish_agent_event(
        agent_id=agent_id,
        serialized_event=AgentEvent(
            agent_id=agent_id, event=PlanStatusEvent(status=status)
        ).model_dump_json(),
    )


async def publish_agent_execution_status(
    agent_id: str,
    plan_run_id: str,
    plan_id: str,
    status: Status,
    logger: Optional[Union[logging.Logger, logging.LoggerAdapter]] = None,
    # Only populated if the plan is finished
    updated_output_ids: Optional[List[str]] = None,
    run_summary_long: Optional[str | TextOutput] = None,
    run_summary_short: Optional[str] = None,
    db: Optional[AsyncDB] = None,
) -> None:
    db = db or AsyncDB(pg=SyncBoostedPG())
    try:
        await publish_agent_event(
            agent_id=agent_id,
            serialized_event=AgentEvent(
                agent_id=agent_id,
                event=ExecutionStatusEvent(
                    status=status,
                    plan_run_id=plan_run_id,
                    plan_id=plan_id,
                    newly_updated_outputs=updated_output_ids or [],
                    run_summary_long=run_summary_long,
                    run_summary_short=run_summary_short,
                ),
            ).model_dump_json(),
        )
        await db.update_plan_run(
            agent_id=agent_id, plan_id=plan_id, plan_run_id=plan_run_id, status=status
        )
        user_id = await db.get_agent_owner(agent_id=agent_id)
        if get_ld_flag(
            flag_name="qc_tool_flag",
            default=False,
            user_context=get_user_context(user_id=user_id),
        ):
            # Send a message to update the agent quality
            await send_agent_quality_message(
                agent_id=agent_id, plan_id=plan_id, status=status, db=db
            )
    except Exception as e:
        if logger:
            logger.exception(
                f"Failed to publish execution status for {agent_id=} {plan_run_id=}: {e}"
            )


async def publish_agent_task_status(
    agent_id: str,
    tasks: List[TaskStatus],
    plan_run_id: str,
    logger: Optional[Union[logging.Logger, logging.LoggerAdapter]] = None,
    db: Optional[AsyncDB] = None,
) -> None:
    db = db or AsyncDB(SyncBoostedPG())
    try:
        await publish_agent_event(
            agent_id=agent_id,
            serialized_event=AgentEvent(
                agent_id=agent_id,
                event=TaskStatusEvent(tasks=tasks, plan_run_id=plan_run_id),
            ).model_dump_json(),
        )
        await db.update_task_statuses(agent_id=agent_id, tasks=tasks, plan_run_id=plan_run_id)
    except Exception as e:
        if logger:
            logger.exception(f"Failed to publish task statuses for {agent_id=}: {e}")


async def get_agent_task_logs(
    agent_id: str,
    plan_run_id: str,
    task_id: str,
    db: Optional[Postgres] = None,
) -> List[PlanRunTaskLog]:
    """
    Get a task's logs to let the frontend know about any sub tasks / outputs.
    Retrieves worklogs from db, since the worklogs are populated by individual tools.
    """

    db = db or get_psql()
    rows = db.get_agent_worklogs(agent_id, task_ids=[task_id], plan_run_ids=[plan_run_id])

    if rows:
        log_rows = []
        for row in rows:
            if row["is_task_output"]:
                continue
            message = load_io_type(row["log_message"])
            message_str = (await message.get()).val if isinstance(message, Text) else str(message)
            log_rows.append(
                PlanRunTaskLog(
                    log_id=row["log_id"],
                    log_message=message_str,
                    created_at=row["created_at"],
                    has_output=row["has_output"],
                )
            )
        return sorted(
            log_rows,
            key=lambda tl: tl.created_at,
        )

    return []


async def send_agent_emails(
    pg: AsyncDB,
    agent_id: str,
    email_subject: str,
    plan_run_id: str,
    run_summary_short: str,
    run_summary_long: str,
) -> None:
    """
    Args:
        pg: Async database
        agent_id: The agent id to retrieve the owner for
        plan_run_id: The plan run id
        run_summary_short: Short summary of run
        run_summary_long: Long Summary of run (What's new section on report)

    Returns: None this function checks at the completion of a plan and checks
    if the agent has email subscriptions and sends a message to the notification service
    to send the email
    """
    # Get agent Information
    agent_name = await pg.get_agent_name(agent_id=agent_id)
    agent_owner = await pg.get_agent_owner(agent_id=agent_id)
    agent_subs = await pg.get_agent_subscriptions(agent_id=agent_id)
    # if we have any agent subscriptions then send an sqs message
    # Do not automatically include the agent owner in the emailing list
    if agent_subs:
        # share the plan
        await pg.set_plan_run_share_status(plan_run_id=plan_run_id, status=True)

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

        queue.send_message(MessageBody=detailed_message.model_dump_json())


async def send_welcome_email(user_id: str, async_db: AsyncDB | None = None) -> None:
    user = cast(User, await get_user_cached(user_id=user_id, async_db=async_db))

    # This is HubSpot ID for the welcome email we're sending
    WELCOME_EMAIL_ID = 181851687081

    message = OnboardingEmailMessage(
        user_name=user.name,
        user_id=user_id,
        email=user.email,
        hubspot_email_id=WELCOME_EMAIL_ID,
    )

    queue.send_message(MessageBody=message.model_dump_json())


def forward_existing_email(notification_key: str, recipient_email: str) -> None:
    # only forward emails in prod environment
    if get_environment_tag() == PROD_TAG:
        message = ForwardingEmailMessage(
            notification_key=notification_key,
            recipient_email=recipient_email,
        )
        queue.send_message(MessageBody=message.model_dump_json())
