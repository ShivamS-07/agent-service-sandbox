import json
from typing import Dict, List, Optional

from agent_service.GPT.constants import GPT4_O
from agent_service.GPT.requests import GPT
from agent_service.planner.planner_types import SamplePlan
from agent_service.planner.prompts import (
    SAMPLE_PLANS_MAIN_PROMPT,
    SAMPLE_PLANS_SYS_PROMPT,
)
from agent_service.utils.logs import async_perf_logger
from agent_service.utils.postgres import get_psql
from agent_service.utils.prefect import get_prefect_logger
from agent_service.utils.string_utils import clean_to_json_if_needed


@async_perf_logger
async def get_similar_sample_plans(
    input: str, context: Optional[Dict[str, str]] = None
) -> List[SamplePlan]:
    db = get_psql()
    gpt = GPT(model=GPT4_O, context=context)
    sample_plans = db.get_all_sample_plans()
    sample_plans_str = "\n".join(
        [f"{i}. {sample_plan.input}" for i, sample_plan in enumerate(sample_plans)]
    )
    try:
        result = set(
            json.loads(
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
        )
    except json.JSONDecodeError as e:
        logger = get_prefect_logger(__name__)
        logger.warning(f"Failed to select simple plans with error: {e}")
        return []

    return [sample_plan for i, sample_plan in enumerate(sample_plans) if i in result]
