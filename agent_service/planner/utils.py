import json
import logging
import random
from typing import Dict, List, Optional, Set, Union

from agent_service.GPT.constants import GPT4_O
from agent_service.GPT.requests import GPT
from agent_service.planner.constants import MAX_SAMPLE_INPUT_MULTIPLER
from agent_service.planner.planner_types import SamplePlan
from agent_service.planner.prompts import (
    SAMPLE_PLANS_MAIN_PROMPT,
    SAMPLE_PLANS_SYS_PROMPT,
)
from agent_service.utils.async_db import AsyncDB
from agent_service.utils.async_utils import gather_with_concurrency
from agent_service.utils.enablement_function_registry import is_plan_enabled
from agent_service.utils.logs import async_perf_logger
from agent_service.utils.postgres import Postgres, get_psql
from agent_service.utils.prefect import get_prefect_logger
from agent_service.utils.string_utils import clean_to_json_if_needed

logger = logging.getLogger(__name__)


async def check_cancelled(
    db: Union[Postgres, AsyncDB],
    agent_id: Optional[str] = None,
    plan_id: Optional[str] = None,
    plan_run_id: Optional[str] = None,
) -> bool:
    if not agent_id and not plan_id and not plan_run_id:
        return False

    ids = [val for val in (plan_id, plan_run_id) if val is not None]
    if isinstance(db, Postgres):
        return db.is_cancelled(ids_to_check=ids) or db.is_agent_deleted(agent_id=agent_id)
    else:
        res = await gather_with_concurrency(
            [db.is_cancelled(ids_to_check=ids), db.is_agent_deleted(agent_id=agent_id)]
        )
        return res[0] or res[1]


async def _get_similar_plans(gpt: GPT, sample_plans: List[SamplePlan], input: str) -> Set[str]:
    try:
        sample_plans_str = "\n".join(
            [f"{i}. {sample_plan.input}" for i, sample_plan in enumerate(sample_plans)]
        )
        relevant_sample_plans = json.loads(
            clean_to_json_if_needed(
                await gpt.do_chat_w_sys_prompt(
                    main_prompt=SAMPLE_PLANS_MAIN_PROMPT.format(
                        old_requests=sample_plans_str, new_request=input
                    ),
                    sys_prompt=SAMPLE_PLANS_SYS_PROMPT.format(),
                    max_tokens=30,
                )
            )
        )
        return set(
            sample_plans[i].id
            for i in relevant_sample_plans
            if isinstance(i, int) and i < len(sample_plans)
        )
    except json.JSONDecodeError as e:
        logger = get_prefect_logger(__name__)
        logger.warning(f"Failed to select simple plans with error: {e}")
        return set()


@async_perf_logger
async def get_similar_sample_plans(
    input: str, enabled_only: bool = True, context: Optional[Dict[str, str]] = None
) -> List[SamplePlan]:
    db = get_psql()
    gpt = GPT(model=GPT4_O, context=context)
    sample_plans = db.get_all_sample_plans()
    len_filtered_sample_plans = []
    for sample_plan in sample_plans:
        if not (len(sample_plan.input) > len(input) * MAX_SAMPLE_INPUT_MULTIPLER):
            len_filtered_sample_plans.append(sample_plan)

    if enabled_only:
        enablement_filtered_sample_plans = [
            plan for plan in len_filtered_sample_plans if is_plan_enabled(plan, context)
        ]

        sample_plans = enablement_filtered_sample_plans

        count = len(len_filtered_sample_plans) - len(enablement_filtered_sample_plans)

        if count > 0:
            logger.info(f"Skipped {count} disabled sample plans")
    else:
        sample_plans = len_filtered_sample_plans
    tasks = []
    tasks.append(_get_similar_plans(gpt=gpt, sample_plans=sample_plans[:], input=input))
    reversed_list = sample_plans[::-1]
    tasks.append(_get_similar_plans(gpt=gpt, sample_plans=reversed_list, input=input))
    shuffled_list = sample_plans[:]
    random.shuffle(shuffled_list)
    tasks.append(_get_similar_plans(gpt=gpt, sample_plans=shuffled_list, input=input))
    results = await gather_with_concurrency(tasks)
    results_union = set.union(*results)
    return [sample_plan for sample_plan in sample_plans if sample_plan.id in results_union]
