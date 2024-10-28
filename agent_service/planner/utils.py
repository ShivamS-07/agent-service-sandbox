import json
import random
from typing import Dict, List, Optional, Set

from agent_service.GPT.constants import GPT4_O
from agent_service.GPT.requests import GPT
from agent_service.planner.constants import MAX_SAMPLE_INPUT_MULTIPLER
from agent_service.planner.planner_types import SamplePlan
from agent_service.planner.prompts import (
    SAMPLE_PLANS_MAIN_PROMPT,
    SAMPLE_PLANS_SYS_PROMPT,
)
from agent_service.utils.async_utils import gather_with_concurrency
from agent_service.utils.logs import async_perf_logger
from agent_service.utils.postgres import get_psql
from agent_service.utils.prefect import get_prefect_logger
from agent_service.utils.string_utils import clean_to_json_if_needed


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
        return set(sample_plans[i].id for i in relevant_sample_plans)
    except json.JSONDecodeError as e:
        logger = get_prefect_logger(__name__)
        logger.warning(f"Failed to select simple plans with error: {e}")
        return set()


@async_perf_logger
async def get_similar_sample_plans(
    input: str, context: Optional[Dict[str, str]] = None
) -> List[SamplePlan]:
    db = get_psql()
    gpt = GPT(model=GPT4_O, context=context)
    sample_plans = db.get_all_sample_plans()
    len_filtered_sample_plans = []
    for sample_plan in sample_plans:
        if not (len(sample_plan.input) > len(input) * MAX_SAMPLE_INPUT_MULTIPLER):
            len_filtered_sample_plans.append(sample_plan)
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
