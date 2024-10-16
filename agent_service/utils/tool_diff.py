import datetime
import json
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional

from agent_service.io_type_utils import HistoryEntry, IOType, load_io_type
from agent_service.io_types.stock import StockID
from agent_service.io_types.text import StockText, TextCitation
from agent_service.types import PlanRunContext
from agent_service.utils.async_db import AsyncDB
from agent_service.utils.clickhouse import Clickhouse
from agent_service.utils.postgres import SyncBoostedPG
from agent_service.utils.prefect import get_prefect_logger


@dataclass
class PrevRunInfo:
    inputs_str: str
    output: Optional[IOType]
    debug: Dict[str, str]
    timestamp: datetime.datetime


async def get_prev_run_info(context: PlanRunContext, tool_name: str) -> Optional[PrevRunInfo]:
    logger = get_prefect_logger(__name__)
    pg_db = AsyncDB(pg=SyncBoostedPG(skip_commit=context.skip_db_commit))
    previous_run_id = await pg_db.get_previous_plan_run(
        agent_id=context.agent_id,
        plan_id=context.plan_id,
        latest_plan_run_id=context.plan_run_id,
        cutoff_dt=context.as_of_date,
    )
    if previous_run_id is None:
        return None

    ch_db = Clickhouse()
    if context.task_id is None:  # shouldn't happen
        return None
    logger.info("Checking clickhouse for prior run info")
    io = await ch_db.get_io_for_tool_run(previous_run_id, context.task_id, tool_name)
    if io is None:
        logger.info("Checking postgres for prior run info")
        io = await pg_db.get_task_run_info(
            plan_run_id=previous_run_id, task_id=context.task_id, tool_name=tool_name
        )
    if io is None:
        logger.info("No prior run info found")
        return None
    inputs_str, output_str, debug_str, timestamp = io
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


def add_old_history(
    new_stock: StockID,
    old_stock: StockID,
    task_id: str,
    new_texts: Optional[List[StockText]] = None,
) -> Optional[StockID]:

    for history_entry in old_stock.history:
        if history_entry.task_id == task_id:
            break

    if new_texts is not None:
        new_text_set = set(new_texts)
        for citation in history_entry.citations:
            if isinstance(citation, TextCitation):
                if citation.source_text not in new_text_set:
                    return None
    return new_stock.inject_history_entry(history_entry)


def add_task_id_to_stocks_history(stocks: List[StockID], task_id: str) -> List[StockID]:
    return [stock.inject_history_entry(HistoryEntry(task_id=task_id)) for stock in stocks]
