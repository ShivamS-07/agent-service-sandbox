import datetime
import logging
from typing import Optional, Tuple

import pytz
from apscheduler.triggers.cron import CronTrigger
from cron_descriptor import get_description
from pydantic import BaseModel

from agent_service.GPT.constants import GPT4_O
from agent_service.GPT.requests import GPT
from agent_service.utils.constants import DEFAULT_CRON_SCHEDULE
from agent_service.utils.date_utils import get_now_utc
from agent_service.utils.gpt_logging import GptJobIdType, GptJobType, create_gpt_context
from agent_service.utils.prompt_utils import FilledPrompt, Prompt
from agent_service.utils.string_utils import strip_code_backticks

logger = logging.getLogger(__name__)

CRON_GEN_PROMPT = Prompt(
    template="""Given this description for a schedule in natural language,
return ONLY a cron string that matches the input description as closely as
possible. The job that will be run is a financial analysis job. If the user asks
for running something 'weekly' please default to Monday at 8am. If the user asks
for running something 'daily' please default to 8am.

Please return a cron string with exactly 5 components:
    minute
    hour
    day of month
    month
    day of week

Commas, hyphens, and slashes are also supported.

However, your maximum frequency allowed is once per hour. So the minute field
CANNOT be a star. It must be a number.

Input schedule description:
{schedule_desc}

Your cron schedule output:
""",
    name="AGENT_SCHEDULER_CRON_GENERATION",
)

_DAY_OF_WEEK_MAP = {
    "0": "sun",
    "1": "mon",
    "2": "tue",
    "3": "wed",
    "4": "thu",
    "5": "fri",
    "6": "sat",
}


class AgentSchedule(BaseModel):
    cron_schedule: str = DEFAULT_CRON_SCHEDULE
    user_schedule_description: Optional[str] = None
    # A human-readable description of the above cron schedule
    generated_schedule_description: str = "Daily at 8am"
    timezone: str = "US/Eastern"

    @staticmethod
    def default() -> "AgentSchedule":
        return AgentSchedule()

    def get_next_run(self) -> Optional[datetime.datetime]:
        trigger = self.to_cron_trigger()
        return trigger.get_next_fire_time(previous_fire_time=get_now_utc(), now=get_now_utc())

    def to_cron_trigger(self) -> CronTrigger:
        values = self.cron_schedule.split()
        day_of_week = values[4]
        # Hacky because we need to handle cases with commas, dashes, etc.
        if day_of_week != "*":
            for num, weekday in _DAY_OF_WEEK_MAP.items():
                day_of_week = day_of_week.replace(num, weekday)
        trigger = CronTrigger(
            minute=values[0],
            hour=values[1],
            day=values[2],
            month=values[3],
            day_of_week=day_of_week,
            timezone=pytz.timezone(self.timezone),
        )
        return trigger


async def _get_cron_from_gpt(agent_id: str, user_desc: str) -> str:
    gpt_context = create_gpt_context(GptJobType.AGENT_CHATBOT, agent_id, GptJobIdType.AGENT_ID)
    llm = GPT(context=gpt_context, model=GPT4_O)
    prompt = CRON_GEN_PROMPT.format(schedule_desc=user_desc)
    output = await llm.do_chat_w_sys_prompt(
        main_prompt=prompt, sys_prompt=FilledPrompt(filled_prompt="")
    )
    output = output.replace("`", "")
    return strip_code_backticks(output.strip()).strip()


async def get_schedule_from_user_description(
    agent_id: str, user_desc: str
) -> Tuple[AgentSchedule, bool, Optional[str]]:
    """
    Given a description of a schedule, attempt to create an AgentSchedule object
    based on the description. If not possible, return the default
    schedule. Second return value is a boolean indicating success and an optional error message.
    """
    try:
        cron_schedule = await _get_cron_from_gpt(agent_id=agent_id, user_desc=user_desc)
        logger.info(f"Got cron schedule from GPT: {cron_schedule}")
        cron_pieces = cron_schedule.split()
        assert len(cron_pieces) == 5, "Unable to generate a schedule from the input."
        assert cron_pieces[0] != "*", "Schedules may run at most hourly."
        cron_description = get_description(cron_schedule)
        return (
            AgentSchedule(
                cron_schedule=cron_schedule,
                user_schedule_description=user_desc,
                generated_schedule_description=cron_description,
            ),
            True,
            None,
        )
    except AssertionError as ae:
        err = str(ae)
        logger.exception(f"Failed to handle user schedule request: '{user_desc=}', {err=}")
        return (AgentSchedule.default(), False, err)
    except Exception:
        err = f"Failed to handle user schedule request: '{user_desc=}'"
        logger.exception(err)
        return (AgentSchedule.default(), False, err)
