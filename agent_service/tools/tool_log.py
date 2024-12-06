import inspect
from typing import Optional

from agent_service.io_type_utils import IOType, dump_io_type
from agent_service.types import PlanRunContext
from agent_service.utils.postgres import get_psql
from agent_service.utils.prefect import get_prefect_logger


def caller_module_name(depth: int = 1) -> str | None:
    default = __name__
    frame = inspect.currentframe()
    if not frame:
        return default

    # Go back up the stack to the caller (or caller's caller, etc.)
    # (add 1 to depth to account for this function itself)
    for _ in range(depth + 1):
        if not (frame := frame.f_back):
            return default

    if module := inspect.getmodule(frame):
        return module.__name__

    return default


async def tool_log(
    log: IOType,
    context: PlanRunContext,
    associated_data: Optional[IOType] = None,
) -> None:
    if not context.skip_db_commit and not context.skip_task_logging:
        db = get_psql(skip_commit=context.skip_db_commit)
        db.write_tool_log(log=log, context=context, associated_data=associated_data)
    io_str = dump_io_type(log)

    # log as if from the caller of this function

    # get logger name same as caller's
    try:
        prefect_logger = get_prefect_logger(caller_module_name() or __name__)
    except Exception:
        prefect_logger = get_prefect_logger(__name__)

    # get the file/lineno from the caller
    prefect_logger.warning(io_str, stacklevel=2)
