from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from agent_service.io_type_utils import IOType, load_io_type
from agent_service.io_types.stock import StockID
from agent_service.io_types.text import StockText, TextCitation
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
