from typing import Optional, Tuple

from agent_service.io_type_utils import IOType, load_io_type
from agent_service.tool import ToolArgs
from agent_service.types import PlanRunContext
from agent_service.utils.async_db import AsyncDB
from agent_service.utils.clickhouse import Clickhouse
from agent_service.utils.postgres import SyncBoostedPG


async def get_prev_run_info(context: PlanRunContext) -> Optional[Tuple[ToolArgs, IOType]]:
    pg_db = AsyncDB(pg=SyncBoostedPG(skip_commit=context.skip_db_commit))
    previous_run = await pg_db.get_previous_plan_run(
        agent_id=context.agent_id, plan_id=context.plan_id, latest_plan_run_id=context.plan_run_id
    )
    if previous_run is None:
        return None

    ch_db = Clickhouse()
    if context.task_id is None:  # shouldn't happen
        return None
    io = ch_db.get_io_for_tool_run(previous_run, context.task_id)
    if io is None:
        return None
    inputs_str, output_str = io
    inputs = ToolArgs.model_validate_json(inputs_str)
    output = load_io_type(output_str)

    return inputs, output
