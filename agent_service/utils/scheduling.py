import logging
from typing import Tuple

from cron_descriptor import get_description

from agent_service.endpoints.models import AgentSchedule
from agent_service.GPT.constants import GPT4_O
from agent_service.GPT.requests import GPT
from agent_service.utils.gpt_logging import GptJobIdType, GptJobType, create_gpt_context
from agent_service.utils.prompt_utils import FilledPrompt, Prompt
from agent_service.utils.string_utils import strip_code_backticks

logger = logging.getLogger(__name__)

DEFAULT_CRON_SCHEDULE = "0 8 * * *"  # Daily at 8am
DEFAULT_CRON_DESCRIPTION = get_description(DEFAULT_CRON_SCHEDULE)


CRON_GEN_PROMPT = Prompt(
    template="""Given this description for a schedule in natural language,
return ONLY a cron string that matches the input description as closely as
possible. The job that will be run is a financial analysis job. If the user asks
for running something 'weekly' please default to Monday at 8am. If the user asks
for running something 'daily' please default to 8am.

Input schedule description:
{schedule_desc}

Your cron schedule output:
""",
    name="AGENT_SCHEDULER_CRON_GENERATION",
)


async def _get_cron_from_gpt(agent_id: str, user_desc: str) -> str:
    gpt_context = create_gpt_context(GptJobType.AGENT_CHATBOT, agent_id, GptJobIdType.AGENT_ID)
    llm = GPT(context=gpt_context, model=GPT4_O)
    prompt = CRON_GEN_PROMPT.format(schedule_desc=user_desc)
    output = await llm.do_chat_w_sys_prompt(
        main_prompt=prompt, sys_prompt=FilledPrompt(filled_prompt="")
    )
    return strip_code_backticks(output.strip()).strip()


async def get_schedule_from_user_description(
    agent_id: str, user_desc: str
) -> Tuple[AgentSchedule, bool]:
    """
    Given a description of a schedule, attempt to create an AgentSchedule object
    based on the description. If not possible, return the default
    schedule. Second return value is a boolean indicating success.
    """
    try:
        cron_schedule = await _get_cron_from_gpt(agent_id=agent_id, user_desc=user_desc)
        cron_description = get_description(cron_schedule)
        success = True
    except Exception:
        logger.exception(f"Failed to handle user schedule request: {user_desc=}")
        cron_schedule = DEFAULT_CRON_SCHEDULE
        cron_description = DEFAULT_CRON_DESCRIPTION
        success = False

    return (
        AgentSchedule(
            cron_schedule=cron_schedule,
            user_schedule_description=user_desc,
            generated_schedule_description=cron_description,
        ),
        success,
    )
