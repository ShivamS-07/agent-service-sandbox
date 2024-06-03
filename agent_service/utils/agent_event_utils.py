from typing import Optional, Union

from agent_service.endpoints.models import (
    AgentEvent,
    AgentOutput,
    EventType,
    ExecutionPlanTemplate,
    MessageEvent,
    NewPlanEvent,
    OutputEvent,
)
from agent_service.io_type_utils import IOType
from agent_service.planner.planner_types import ExecutionPlan
from agent_service.types import Message, PlanRunContext
from agent_service.utils.async_db import AsyncDB, get_output_from_io_type
from agent_service.utils.date_utils import get_now_utc
from agent_service.utils.postgres import Postgres, SyncBoostedPG, get_psql
from agent_service.utils.redis_queue import publish_agent_event


async def send_chat_message(
    message: Message,
    db: Optional[Union[AsyncDB, Postgres]] = None,
    insert_message_into_db: bool = True,
    skip_commit: bool = False,
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
        event = AgentEvent(agent_id=message.agent_id, event=MessageEvent(message=message))
        await publish_agent_event(
            agent_id=message.agent_id, serialized_event=event.model_dump_json()
        )


async def publish_agent_output(
    output: IOType,
    context: PlanRunContext,
    is_intermediate: bool = False,
    db: Optional[Postgres] = None,
) -> None:
    if not db:
        db = get_psql()
    db.write_agent_output(output=output, context=context)
    rich_output = await get_output_from_io_type(output, pg=SyncBoostedPG())
    event = AgentEvent(
        agent_id=context.agent_id,
        event=OutputEvent(
            output=AgentOutput(
                agent_id=context.agent_id,
                plan_id=context.plan_id,
                plan_run_id=context.plan_run_id,
                output=rich_output,  # type: ignore
                is_intermediate=is_intermediate,
                created_at=get_now_utc(),
            )
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
