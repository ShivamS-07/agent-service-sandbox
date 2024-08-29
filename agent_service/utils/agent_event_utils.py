import logging
from typing import List, Optional, Union, cast

import boto3

from agent_service.endpoints.models import (
    AgentEvent,
    AgentNotificationBody,
    AgentNotificationData,
    AgentOutput,
    AgentSubscriptionMessage,
    EventType,
    ExecutionPlanTemplate,
    ExecutionStatusEvent,
    MessageEvent,
    NewPlanEvent,
    NotificationEvent,
    NotifyMessageEvent,
    OutputEvent,
    PlanRunTaskLog,
    PlanStatusEvent,
    PlanTemplateTask,
    Status,
    TaskStatus,
    TaskStatusEvent,
)
from agent_service.io_type_utils import load_io_type
from agent_service.io_types.text import TextOutput
from agent_service.planner.planner_types import ExecutionPlan, OutputWithID, PlanStatus
from agent_service.types import Message, Notification, PlanRunContext
from agent_service.utils.async_db import AsyncDB
from agent_service.utils.async_postgres_base import AsyncPostgresBase
from agent_service.utils.async_utils import gather_with_concurrency
from agent_service.utils.constants import NOTIFICATION_SERVICE_QUEUE
from agent_service.utils.date_utils import get_now_utc
from agent_service.utils.output_utils.output_construction import get_output_from_io_type
from agent_service.utils.postgres import Postgres, SyncBoostedPG, get_psql
from agent_service.utils.redis_queue import (
    publish_agent_event,
    publish_notification_event,
)

logger = logging.getLogger(__name__)


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


async def publish_agent_output(
    outputs_with_ids: List[OutputWithID],
    context: PlanRunContext,
    live_plan_output: bool = False,
    is_intermediate: bool = False,
    db: Optional[Postgres] = None,
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
    results = await gather_with_concurrency(coros, n=len(coros))

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
                )
                for (rich_output, o) in zip(results, outputs_with_ids)
            ]
        ),
    )
    await publish_agent_event(agent_id=context.agent_id, serialized_event=event.model_dump_json())


async def publish_agent_execution_plan(
    plan: ExecutionPlan, context: PlanRunContext, db: Optional[Union[AsyncDB, Postgres]] = None
) -> None:
    if not db:
        db = AsyncDB(SyncBoostedPG(skip_commit=context.skip_db_commit))

    if isinstance(db, AsyncDB):
        await db.write_execution_plan(
            plan_id=context.plan_id,
            agent_id=context.agent_id,
            plan=plan,
            status=PlanStatus.READY,
        )
        await db.insert_plan_run(
            agent_id=context.agent_id, plan_id=context.plan_id, plan_run_id=context.plan_run_id
        )
    elif isinstance(db, Postgres):
        db.write_execution_plan(
            plan_id=context.plan_id,
            agent_id=context.agent_id,
            plan=plan,
            status=PlanStatus.READY,
        )
        # also insert the plan run to avoid weirdness with the frontend
        db.insert_plan_run(
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
) -> None:
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
) -> None:
    try:
        await publish_agent_event(
            agent_id=agent_id,
            serialized_event=AgentEvent(
                agent_id=agent_id,
                event=TaskStatusEvent(tasks=tasks, plan_run_id=plan_run_id),
            ).model_dump_json(),
        )
    except Exception as e:
        if logger:
            logger.exception(f"Failed to publish task statuses for {agent_id=}: {e}")


async def get_agent_task_logs(
    agent_id: str,
    task_id: str,
    db: Optional[Postgres] = None,
) -> List[PlanRunTaskLog]:
    """
    Get a task's logs to let the frontend know about any sub tasks / outputs.
    Retrieves worklogs from db, since the worklogs are populated by individual tools.
    """

    db = db or get_psql()
    rows = db.get_agent_worklogs(agent_id, task_ids=[task_id])

    if rows:
        return sorted(
            [
                PlanRunTaskLog(
                    log_id=row["log_id"],
                    log_message=cast(str, load_io_type(row["log_message"])),
                    created_at=row["created_at"],
                    has_output=row["has_output"],
                )
                for row in rows
                if not row["is_task_output"]
            ],
            key=lambda tl: tl.created_at,
        )

    return []


async def send_agent_emails(
    pg: AsyncDB, agent_id: str, plan_run_id: str, run_summary_short: str, run_summary_long: str
) -> None:
    """

    Args:
        pg: Async database
        agent_id: The agent id to retrieve the owner for
        plan_run_id: The plan run id
        run_summary_short: Summary of run
        run_summary_long: Long Summary of run

    Returns: None this function checks at the completion of a plan and checks
    if the agent has email subscriptions and sends a message to the notification service
    to send the email
    """
    sqs = boto3.resource("sqs", region_name="us-west-2")
    # Get agent Information
    agent_name = await pg.get_agent_name(agent_id=agent_id)
    agent_owner = await pg.get_agent_owner(agent_id=agent_id)
    agent_subs = await pg.get_agent_subscriptions(agent_id=agent_id)
    # if we have any agent subscriptions then send an sqs message
    if agent_subs:
        # share the plan
        await pg.set_plan_run_share_status(plan_run_id=plan_run_id, status=True)
        # Always include the agent owner in emails
        user_ids = set([agent_sub.user_id for agent_sub in agent_subs])
        if agent_owner:
            user_ids.add(agent_owner)
        # create a subscription message
        message = AgentSubscriptionMessage(
            user_ids=list(user_ids),
            agent_data=[
                AgentNotificationData(
                    agent_name=agent_name,
                    agent_id=agent_id,
                    plan_run_id=plan_run_id,
                    agent_owner=agent_owner if agent_owner else "",
                    notification_body=AgentNotificationBody(
                        summary_title=run_summary_short if run_summary_short else "",
                        summary_body=run_summary_long if run_summary_long else "",
                    ),
                )
            ],
        )
        queue = sqs.get_queue_by_name(QueueName=NOTIFICATION_SERVICE_QUEUE)
        queue.send_message(MessageBody=message.model_dump_json())
