from typing import Optional

from agent_service.endpoints.models import (
    AgentEvent,
    AgentOutput,
    MessageEvent,
    OutputEvent,
)
from agent_service.io_type_utils import IOType
from agent_service.types import Message, PlanRunContext
from agent_service.utils.async_db import AsyncDB, get_output_from_io_type
from agent_service.utils.date_utils import get_now_utc
from agent_service.utils.postgres import SyncBoostedPG, get_psql
from agent_service.utils.redis_queue import publish_agent_event


async def send_chat_message(
    message: Message, db: Optional[AsyncDB] = None, insert_message_into_db: bool = True
) -> None:
    """
    Sends a chat message from GPT to the user. Updates the redis queue as well
    as the DB. If no DB instance is passed in, uses a SYNCHRONOUS DB instead.
    """
    if insert_message_into_db:
        if db:
            await db.insert_chat_messages(messages=[message])
        else:
            get_psql().insert_chat_messages(messages=[message])

    if not message.is_user_message:
        event = AgentEvent(agent_id=message.agent_id, event=MessageEvent(message=message))
        await publish_agent_event(
            agent_id=message.agent_id, serialized_event=event.model_dump_json()
        )


async def publish_agent_output(
    output: IOType, context: PlanRunContext, is_intermediate: bool = False
) -> None:
    get_psql().write_agent_output(output=output, context=context)
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
