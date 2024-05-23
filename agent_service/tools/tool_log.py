from typing import Optional

from agent_service.io_type_utils import IOType, dump_io_type
from agent_service.types import PlanRunContext
from agent_service.utils.async_utils import async_wrap
from agent_service.utils.postgres import get_psql
from agent_service.utils.prefect import get_prefect_logger


@async_wrap
def tool_log(
    log: IOType, context: PlanRunContext, associated_data: Optional[IOType] = None
) -> None:
    if not context.skip_db_commit:
        db = get_psql(skip_commit=context.skip_db_commit)
        db.write_tool_log(log=log, context=context, associated_data=associated_data)
    logger = get_prefect_logger(__name__)
    logger.info(dump_io_type(log))
