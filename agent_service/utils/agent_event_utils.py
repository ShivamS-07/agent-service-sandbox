import asyncio
from collections import defaultdict
from typing import Any, Dict, List, Optional, Union, cast

from agent_service.endpoints.models import (
    AgentEvent,
    AgentOutput,
    EventType,
    ExecutionPlanTemplate,
    MessageEvent,
    NewPlanEvent,
    OutputEvent,
    PlanRun,
    PlanRunTaskLog,
    Status,
    WorklogEvent,
)
from agent_service.endpoints.utils import (
    get_plan_run_task_list,
    reset_plan_run_status_if_needed,
)
from agent_service.io_type_utils import IOType, load_io_type
from agent_service.planner.planner_types import ExecutionPlan
from agent_service.types import Message, Notification, PlanRunContext
from agent_service.utils.async_db import AsyncDB
from agent_service.utils.date_utils import get_now_utc
from agent_service.utils.output_construction import get_output_from_io_type
from agent_service.utils.postgres import Postgres, SyncBoostedPG, get_psql
from agent_service.utils.prefect import (
    get_prefect_plan_run_statuses,
    get_prefect_task_statuses,
)
from agent_service.utils.redis_queue import publish_agent_event


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
            elif db and isinstance(db, Postgres):
                db.insert_notifications(notifications=[notification])
            else:
                get_psql(skip_commit=skip_commit).insert_notifications(notifications=[notification])

        event = AgentEvent(agent_id=message.agent_id, event=MessageEvent(message=message))
        await publish_agent_event(
            agent_id=message.agent_id, serialized_event=event.model_dump_json()
        )


async def publish_agent_output(
    outputs: List[IOType],
    context: PlanRunContext,
    is_intermediate: bool = False,
    db: Optional[Postgres] = None,
) -> None:
    if not db:
        db = get_psql()
    rich_outputs = []
    for output in outputs:
        db.write_agent_output(output=output, context=context)
        rich_output = await get_output_from_io_type(output, pg=SyncBoostedPG())
        rich_outputs.append(rich_output)
    now = get_now_utc()
    event = AgentEvent(
        agent_id=context.agent_id,
        event=OutputEvent(
            output=[
                AgentOutput(
                    agent_id=context.agent_id,
                    plan_id=context.plan_id,
                    plan_run_id=context.plan_run_id,
                    output=rich_output,  # type: ignore
                    is_intermediate=is_intermediate,
                    created_at=now,
                    shared=False,
                )
                for rich_output in rich_outputs
            ]
        ),
    )
    await publish_agent_event(agent_id=context.agent_id, serialized_event=event.model_dump_json())


async def publish_agent_execution_plan(
    plan: ExecutionPlan, context: PlanRunContext, db: Optional[Union[AsyncDB, Postgres]] = None
) -> None:
    if db:
        if isinstance(db, AsyncDB):
            await db.write_execution_plan(
                plan_id=context.plan_id, agent_id=context.agent_id, plan=plan
            )
        elif isinstance(db, Postgres):
            db.write_execution_plan(plan_id=context.plan_id, agent_id=context.agent_id, plan=plan)
    else:  # probably never hits this
        await AsyncDB(SyncBoostedPG()).write_execution_plan(
            plan_id=context.plan_id, agent_id=context.agent_id, plan=plan
        )

    event = AgentEvent(
        agent_id=context.agent_id,
        event=NewPlanEvent(
            event_type=EventType.NEW_PLAN,
            plan=ExecutionPlanTemplate(
                plan_id=context.plan_id, task_names=[node.description for node in plan.nodes]
            ),
        ),
    )

    await publish_agent_event(agent_id=context.agent_id, serialized_event=event.model_dump_json())


async def publish_agent_updated_worklogs(
    context: PlanRunContext,
    plan: ExecutionPlan,
    db: Optional[Postgres] = None,
) -> None:
    db = db or get_psql()
    rows = db.get_agent_worklogs(context.agent_id, plan_run_ids=[context.plan_run_id])
    if not rows:
        return

    plan_run_id_to_status, run_task_pair_to_status = await asyncio.gather(
        get_prefect_plan_run_statuses([context.plan_run_id]),
        get_prefect_task_statuses([context.plan_run_id]),
    )
    if not plan_run_id_to_status or not run_task_pair_to_status:
        return

    task_id_to_logs: Dict[str, List[PlanRunTaskLog]] = defaultdict(list)
    task_id_to_task_output: Dict[str, Dict[str, Any]] = defaultdict(dict)
    for row in rows:
        if row["is_task_output"]:  # there should only be 1 task output per task
            task_id_to_task_output[row["task_id"]] = row
        else:
            task_id_to_logs[row["task_id"]].append(
                PlanRunTaskLog(
                    log_id=row["log_id"],
                    log_message=cast(str, load_io_type(row["log_message"])),
                    created_at=row["created_at"],
                    has_output=row["has_output"],
                )
            )

    prefect_flow_run = plan_run_id_to_status[context.plan_run_id]
    plan_run_status = Status.from_prefect_state(prefect_flow_run.state_type)
    plan_run_start = prefect_flow_run.start_time
    plan_run_end = prefect_flow_run.end_time

    full_tasks = get_plan_run_task_list(
        context.plan_run_id,
        plan.nodes,
        task_id_to_logs,
        task_id_to_task_output,
        run_task_pair_to_status,
    )

    plan_run_status = reset_plan_run_status_if_needed(plan_run_status, full_tasks)

    plan_run = PlanRun(
        plan_run_id=context.plan_run_id,
        status=plan_run_status,
        plan_id=context.plan_id,
        start_time=plan_run_start or full_tasks[0].start_time,  # type: ignore # noqa
        end_time=plan_run_end,
        tasks=full_tasks,
    )
    event = AgentEvent(
        agent_id=context.agent_id,
        event=WorklogEvent(event_type=EventType.WORKLOG, worklog=plan_run),
    )
    await publish_agent_event(agent_id=context.agent_id, serialized_event=event.model_dump_json())
