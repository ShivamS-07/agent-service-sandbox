import datetime
import json
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional

from agent_service.io_type_utils import HistoryEntry, IOType, load_io_type
from agent_service.io_types.stock import StockID
from agent_service.io_types.text import StockText
from agent_service.types import PlanRunContext
from agent_service.utils.async_db import get_async_db
from agent_service.utils.clickhouse import Clickhouse
from agent_service.utils.feature_flags import get_ld_flag
from agent_service.utils.prefect import get_prefect_logger

MAX_TRIES = 3
ONE_MINUTE = datetime.timedelta(minutes=1)


@dataclass
class PrevRunInfo:
    inputs_str: str
    output: Optional[IOType]
    debug: Dict[str, str]
    timestamp: datetime.datetime


async def get_prev_run_info(
    context: PlanRunContext, tool_name: str, plan_run_filter: bool = True
) -> Optional[PrevRunInfo]:
    logger = get_prefect_logger(__name__)
    if context.skip_db_commit:
        pg_db = get_async_db(sync_db=True, skip_commit=True)
    else:
        pg_db = get_async_db()

    if context.task_id is None:  # shouldn't happen
        return None

    null_overrides = get_ld_flag("empty-prev-run-info-task-ids", default={}, user_context=None)

    if context.task_id in null_overrides.get("task_ids", []):
        logger.warning(
            "prev_run_info() returning None due to ld_flag: 'empty-prev-run-info-task-ids'"
        )
        return None

    io = None
    tries = 0

    search_dt = context.as_of_date

    read_from_postgres = False
    while io is None and tries < MAX_TRIES:
        previous_run_id, previous_run_time = await pg_db.get_previous_plan_run(
            agent_id=context.agent_id,
            plan_id=context.plan_id,
            latest_plan_run_id=context.plan_run_id,
            cutoff_dt=search_dt,
            filter_current_plan_run=plan_run_filter,
        )

        read_from_postgres = True
        if previous_run_id is None or previous_run_time is None:
            return None

        ch_db = Clickhouse()

        logger.info("Checking postgres for prior run info")
        io = await pg_db.get_task_run_info(
            plan_run_id=previous_run_id, task_id=context.task_id, tool_name=tool_name
        )
        if io is None:
            logger.info("Checking clickhouse for prior run info")
            io = await ch_db.get_io_for_tool_run(previous_run_id, context.task_id, tool_name)
            read_from_postgres = False

        search_dt = previous_run_time - ONE_MINUTE  # look back before the last run time
        tries += 1

    if io is None:
        logger.info("No prior run info found")
        return None
    inputs_str, output_str, debug_str, timestamp = io
    if read_from_postgres and not output_str:
        # Handle the case where the output might not be populated
        return None
    output = load_io_type(output_str)
    debug = json.loads(debug_str) if debug_str else {}
    return PrevRunInfo(inputs_str=inputs_str, output=output, debug=debug, timestamp=timestamp)


def get_stock_text_lookup(texts: List[StockText]) -> Dict[StockID, List[StockText]]:
    output_dict = defaultdict(list)
    for text in texts:
        if text.stock_id:
            output_dict[text.stock_id].append(text)
    return output_dict


def get_text_diff(texts1: List[StockText], texts2: List[StockText]) -> List[StockText]:
    return list(set(texts1) - set(texts2))


def add_task_id_to_stocks_history(stocks: List[StockID], task_id: str) -> List[StockID]:
    return [stock.inject_history_entry(HistoryEntry(task_id=task_id)) for stock in stocks]
