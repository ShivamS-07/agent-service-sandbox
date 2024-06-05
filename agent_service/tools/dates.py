import datetime
import math
from typing import Optional, Union

import dateparser

from agent_service.GPT.constants import DEFAULT_CHEAP_MODEL
from agent_service.GPT.requests import GPT
from agent_service.io_types.dates import DateRange
from agent_service.tool import ToolArgs, ToolCategory, ToolRegistry, tool
from agent_service.tools.tool_log import tool_log
from agent_service.types import PlanRunContext
from agent_service.utils.prefect import get_prefect_logger
from agent_service.utils.prompt_utils import Prompt

MAX_RETRIES = 3

# PROMPTS

DATE_SYS_PROMPT = Prompt(
    name="LLM_DATE_SYS_PROMPT",
    template=(
        "You are an assistant designed to process date-like strings "
        "and convert them into a proper date format that can be used "
        "with Python's datetime package. Given a date-like string, "
        "you should return the date in the format of YYYY-MM-DD. "
        "In your response you must only return a SINGLE date in the format of YYYY-MM-DD."
        "For example, if the input is 'last two months,' you should return "
        "the date of 60 days ago based on today's date."
        "If the input is 'last quarter,' you should return the date of the first day of "
        "the last quarter based on today's date. "
        "Definitions:"
        "- The year is divided into four quarters:"
        "  - Q1: January 1 to March 31"
        "  - Q2: April 1 to June 30"
        "  - Q3: July 1 to September 30"
        "  - Q4: October 1 to December 31"
    ),
)

DATE_MAIN_PROMPT = Prompt(
    name="LLM_DATE_MAIN_PROMPT",
    template=(
        "Convert the following string to a proper date in the format of YYYY-MM-DD. "
        "The string is as follows:\n"
        "{string_date}\n"
        "Today's date is {today_date}."
        "Now convert this string to a date in the format of YYYY-MM-DD and only return the date."
    ),
)


class DateFromDateStrInput(ToolArgs):
    date_str: str


@tool(
    description=(
        "This function takes a string which refers to a time, either absolute or "
        "relative to the current time, and converts it to a Python date. "
        "You should always input some string, never an empty string. "
    ),
    category=ToolCategory.DATES,
    tool_registry=ToolRegistry,
    is_visible=False,
)
async def get_date_from_date_str(
    args: DateFromDateStrInput, context: PlanRunContext
) -> datetime.date:
    val = dateparser.parse(args.date_str)
    if not val:
        # use gpt to parse the date string
        llm = GPT(model=DEFAULT_CHEAP_MODEL)
        for _ in range(MAX_RETRIES):
            result = await llm.do_chat_w_sys_prompt(
                main_prompt=DATE_MAIN_PROMPT.format(
                    string_date=args.date_str, today_date=datetime.date.today().isoformat()
                ),
                sys_prompt=DATE_SYS_PROMPT.format(),
            )
            # check if result has the format of a date
            if (len(result.split("-")) == 3) and (len(result) == 10):
                break
        val = datetime.datetime.strptime(result, "%Y-%m-%d")
        await tool_log(
            log=f"The computed date for {args.date_str} is {val.date()}",
            context=context,
        )
        return val.date()

    return val.date()


class DateRangeInput(ToolArgs):
    near_date: Optional[Union[datetime.date, datetime.datetime]] = None
    end_date: Optional[Union[datetime.date, datetime.datetime]] = None
    start_date: Optional[Union[datetime.date, datetime.datetime]] = None
    width_years: int = 0
    width_quarters: int = 0
    width_months: int = 0
    width_weeks: int = 0
    width_days: int = 0


@tool(
    description=(
        "This function returns a date range 'DateRange' with dates "
        "that are chosen to be relative to and near to the one of the passed in dates "
        "or near today's date if no date was passed in. "
        "it is extremely useful for when a you need to a date range for a graph but were provided no "
        "references to a date or only a reference to a single date. "
        "Also useful when you need a date range near to a date or close to a date. "
        "Can also be used to create a custom width date range relative to one of the dates "
        "such as '2 years before march 2nd', or for '3 weeks  starting on july 3rd' "
        "'after' should fill in the start_date, 'before' should fill in the end_date "
        "The width can be stated in a combination of years, months, weeks and/or days "
        "the width paramaters should not be negative numbers and always greater than or equal to zero "
        "if no width is provided, the width will default to 1 year "
    ),
    category=ToolCategory.DATES,
    tool_registry=ToolRegistry,
    is_visible=False,
)
async def get_n_width_date_range_near_date(
    args: DateRangeInput, context: PlanRunContext
) -> DateRange:
    logger = get_prefect_logger(__name__)
    logger.info(f"constructing a date range from: {args=}")
    width_days = (
        abs(args.width_years) * 365.25
        + abs(args.width_quarters) * 365.25 / 4
        + abs(args.width_months) * 365.25 / 12
        + abs(args.width_weeks) * 7
        + abs(args.width_days)
    )

    width_days = math.ceil(width_days)

    if width_days <= 0:
        width_days = 365

    if isinstance(args.near_date, datetime.datetime):
        args.near_date = args.near_date.date()

    if isinstance(args.start_date, datetime.datetime):
        args.start_date = args.start_date.date()

    if isinstance(args.end_date, datetime.datetime):
        args.end_date = args.end_date.date()

    # this can happen when only a relative period is given like last 30 days
    if not args.near_date and not args.start_date and not args.end_date:
        args.near_date = datetime.datetime.today().date()

    if isinstance(args.start_date, datetime.datetime):
        args.start_date = args.start_date.date()

    if isinstance(args.end_date, datetime.datetime):
        args.end_date = args.end_date.date()

    if args.start_date and args.end_date:
        start_date = args.start_date
        end_date = args.end_date
    elif args.near_date:
        forward_days = width_days // 2
        end_date = args.near_date + datetime.timedelta(days=forward_days)
        if end_date > datetime.datetime.today().date():
            end_date = datetime.datetime.today().date()

        start_date = end_date - datetime.timedelta(days=width_days)
    else:
        # 'if not near_date:'
        if args.end_date:
            end_date = args.end_date
            start_date = end_date - datetime.timedelta(days=width_days)

        if args.start_date:
            start_date = args.start_date
            end_date = args.start_date + datetime.timedelta(days=width_days)

    # ideally this shouldnt happen, and I havent seen it happen yet
    # but trying to come up with some rules for how to handle inconsistent or
    # extra inputs
    if args.start_date:
        start_date = min(start_date, args.start_date)

    if args.end_date:
        end_date = max(end_date, args.end_date)

    res = DateRange(start_date=start_date, end_date=end_date)
    await tool_log(
        log=(
            "Constructed a date range from the given inputs - "
            f"start date: {res.start_date.isoformat()}, end date: {res.end_date.isoformat()}"
        ),
        context=context,
    )
    logger.info(f"returning {res=}")
    return res
