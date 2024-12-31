import asyncio
import datetime
import json
import math
from typing import Optional, Union

from agent_service.GPT.constants import DEFAULT_CHEAP_MODEL, GPT4_O
from agent_service.GPT.requests import GPT
from agent_service.io_types.dates import DateRange
from agent_service.tool import ToolArgs, ToolCategory, default_tool_registry, tool
from agent_service.tools.tool_log import tool_log
from agent_service.types import ChatContext, Message, PlanRunContext
from agent_service.utils.date_utils import get_now_utc
from agent_service.utils.gpt_logging import GptJobIdType, GptJobType, create_gpt_context
from agent_service.utils.prefect import get_prefect_logger
from agent_service.utils.prompt_utils import Prompt
from agent_service.utils.string_utils import clean_to_json_if_needed

MAX_RETRIES = 3

START_DATE = "start_date"
END_DATE = "end_date"

# PROMPTS

DATE_SYS_PROMPT = Prompt(
    name="LLM_DATE_SYS_PROMPT",
    template=(
        "You are an assistant designed to process date-like strings "
        "and convert them into a proper date format that can be used "
        "with Python's datetime package. Given a date-like string, "
        "you MUST return the date in the format of YYYY-MM-DD. "
        "In your response you must only return a SINGLE date in the format of YYYY-MM-DD."
        "For example, if the input is 'last two months,' and you should compute "
        "the date of 60 days ago based on today's date, and MUST only return the date in the format of YYYY-MM-DD "
        "with no additional text. "
        "If the input is 'last quarter,' you should compute the date of the first day of "
        "the last quarter based on today's date and MUST only return the date in the format of YYYY-MM-DD "
        "with no additional text. "
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
        "This function should be used when a start_date is needed given a string."
        " As a reminder, keywords in the date_str such as: last, previous, before, ago, etc"
        " will tend to create date ranges in the past."
        " Keywords like next, future, upcoming, forthcoming, expected, projected, etc"
        " will tend to create date ranges in the future"
    ),
    category=ToolCategory.DATES,
    tool_registry=default_tool_registry(),
    is_visible=False,
    enabled=False,
)
async def get_date_from_date_str(
    args: DateFromDateStrInput, context: PlanRunContext
) -> datetime.date:
    import dateparser

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


DATE_RANGE_SYS_PROMPT = Prompt(
    name="LLM_DATE_SYS_PROMPT",
    template=(
        "You are an assistant designed to process a string that refers to some kind of date range"
        "and convert it into a proper date range, with each of the start_date and end_date in "
        "YYYY-MM-DD format. You will be provided with the string reference to the date, as well as"
        "the larger chat context in which it occurs, and today's date. Do your best to select "
        "a range that makes sense in the context. If one of the start or end dates is clearly "
        "underspecified, you should make a sensible decision based on the larger context, with "
        "the idea that people are most interested in recent data "
        "(today should be the default end of your range!) "
        "and probably want at least of a year of data for when doing line graphs of statistics"
        "and at least a quarter of data for most text data. Try to satisfy any stated client "
        "needs as much as possible, and for both the start and end date you must output a specific date "
        "Please distinguish carefully between single-day ranges in the past (Apple's stock price from June 25th) "
        "and a date range from that date to the present (e.g. Apple's stock price since June 25th). "
        "In the first case, the end_date should be the same as the start date (June 25th), in the "
        "second case the end date is usually today unless otherwise specified."
        "Your output will be a json with keys `start_date` and `end_date`. "
        "For example, if the input is 'last two months,' and today's date is 2024-07-15, you should "
        'return:\n {{"start_date":"2024-05-15", "end_date":"2024-07-15"}}'
        "Note that today is ALWAYS included in the range in these cases, this is very important!!!"
        "Generally you should assume that weeks/months/quarters/years refer to 7/30/90/365 days "
        "respectively, and so both `last week/month/quarter/year` or `this week/month/quarter/year' refers "
        "to a date range beginning 7/30/90/365 days ago and ending today (inclusive) and NOT the "
        "specific week/month/quarter/year we are in, nor the one before "
        "the one we are in, though if a user asks for a range 'since the beginning of last quarter' for "
        "the same today as above, your output would be: "
        '{{"start_date":"2024-04-01", "end_date":"2024-07-15"}} '
        "since July 15 in in Q3, and April 1st is the beginning of Q2. "
        "Be very careful if the context suggests that multiple non-overlapping ranges are required, "
        "such as in a comparison. If you are interpreting 'previous month' in the context of "
        "a request to compare last month to the previous month, since last month is the last 30 days "
        "the previous month must be the 30 days before that."
        "Usually date ranges are in the past, but please check the context carefully to see if "
        "something ambiguious like '1 month' might be asking for dates in the future, for example "
        "upcoming earnings. "
        "Do not include any wrapper around the JSON. (no ```)"
    ),
)

DATE_RANGE_MAIN_PROMPT = Prompt(
    name="LLM_DATE_MAIN_PROMPT",
    template=(
        "Convert the following string to a date range."
        "The string is as follows:\n"
        "{string_date}\n"
        "Here is the larger context in which it appears:\n{chat_context}"
        "Today's date is {today_date}."
        "Now output a json with start_date and end_date keys, with the dates in YYYY-MM-DD format"
    ),
)


class GetDateRangeInput(ToolArgs):
    date_range_str: str


@tool(
    description="""This function returns a date range object which includes
    start and end dates based on a sensible interpretation of the provided date_range_str.
    Note the function has access to client's chat context to help interpret the date, however
    it is essential that the specific date range can be unambiguously
    identified relevant to any other date ranges possibly mentioned in the context.
    chat context, do not make up a date range, just use the default for the function
    (i.e. do not pass just "one year" to this function if "one year" appears twice in the client input)
    A single date is also valid date range.
    If only the start or end of the range is clearly
    specified, just include that information, and the tool will make a sensible decision for
    underspecified part of the range. Never pass in an empty string.
    Note that not every date range the user mentions should be converted into a date range
    using this function, statistics that are defined using some kind of date range,
    e.g. "percentage gain over the last quarter", will be handled inside the get statistic tool.
    Also, in many cases the default date range for a particular function/tool is exactly what is
    needed and there is no need to build a date range, please check carefully for each tool before
    calling this function. You must never, ever run this function with a date_range_str that does
    not appear explicitly in the client's request (the chat context)!!!!!
    If the user asks for quarterly data and there is no date range provided, then it is of
    utmost importance that you provide a default date range going 2 years into the past.
    If the user asks for yearly data and there is no date range provided, then it is of
    utmost importance that you provide a default date range going 5 years into the past.
    This tool has no access to upcoming earnings information, and so you must absolutely never
    call it with a date_range_str such as "upcoming earnings call".
    If the user talks about an "upcoming earnings call" you must ignore that mention, do
    not call this tool!!! In that case, you should simply use default date range for the relevant tools.
    You must never pass 'upcoming earnings call' or something similar to this tool as the date_range_str,
    especially if you are using the date range to retrieve text (there are no upcoming texts!!!!!)
    Seriously, listen to me about this earnings call thing or you will be fired!
    """,
    category=ToolCategory.DATES,
    tool_registry=default_tool_registry(),
    is_visible=False,
)
async def get_date_range(args: GetDateRangeInput, context: PlanRunContext) -> DateRange:
    # use gpt to parse the date string
    gpt_context = create_gpt_context(
        GptJobType.AGENT_TOOLS, context.agent_id, GptJobIdType.AGENT_ID
    )
    llm = GPT(context=gpt_context, model=GPT4_O)
    result = await llm.do_chat_w_sys_prompt(
        main_prompt=DATE_RANGE_MAIN_PROMPT.format(
            string_date=args.date_range_str,
            today_date=(
                context.as_of_date.date().isoformat()
                if context.as_of_date
                else datetime.date.today().isoformat()
            ),
            chat_context=context.chat.get_gpt_input(client_only=True) if context.chat else "",
        ),
        sys_prompt=DATE_RANGE_SYS_PROMPT.format(),
    )
    date_range_json = json.loads(clean_to_json_if_needed(result))
    await tool_log(
        log=f"Interpreting '{args.date_range_str}' as a start date of {date_range_json[START_DATE]}"
        f" and an end date of {date_range_json[END_DATE]}",
        context=context,
    )
    return DateRange(
        start_date=DateRange.clean_and_convert_str_to_date(date_range_json[START_DATE]),
        end_date=DateRange.clean_and_convert_str_to_date(date_range_json[END_DATE]),
    )


class DateRangeInput(ToolArgs):
    near_date: Optional[Union[datetime.date, datetime.datetime]] = None
    range_ending_on: Optional[Union[datetime.date, datetime.datetime]] = None
    range_starting_from: Optional[Union[datetime.date, datetime.datetime]] = None
    width_years: int = 0
    width_quarters: int = 0
    width_months: int = 0
    width_weeks: int = 0
    width_days: int = 0


@tool(
    description="""This function returns a date range 'DateRange' with dates
    that are chosen to be relative to a passed in dates or
    near today's date if no date was passed in. It is extremely useful for when a
    you need to a date range for a graph but were provided no references to a date
    or only a reference to a single date. Also useful when you need a date range
    near to a date or close to a date. Can also be used to create a custom width
    date range relative to one of the dates such as '2 years before march 2nd', or
    for '3 weeks starting on july 3rd' 'after' should fill in the
    range_starting_from, 'before' should fill in the range_ending_on. The width can
    be stated in a combination of years, months, weeks and/or days the width
    paramaters should not be negative numbers and always greater than or equal to
    zero if no width is provided, the width will default to 1 year.
    """,
    category=ToolCategory.DATES,
    tool_registry=default_tool_registry(),
    is_visible=False,
    enabled=False,
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

    if isinstance(args.range_starting_from, datetime.datetime):
        args.range_starting_from = args.range_starting_from.date()

    if isinstance(args.range_ending_on, datetime.datetime):
        args.range_ending_on = args.range_ending_on.date()

    # this can happen when only a relative period is given like last 30 days
    if not args.near_date and not args.range_starting_from and not args.range_ending_on:
        args.near_date = datetime.datetime.today().date()

    if isinstance(args.range_starting_from, datetime.datetime):
        args.range_starting_from = args.range_starting_from.date()

    if isinstance(args.range_ending_on, datetime.datetime):
        args.range_ending_on = args.range_ending_on.date()

    start_date = get_now_utc().date()
    end_date = start_date
    if args.range_starting_from and args.range_ending_on:
        start_date = args.range_starting_from
        end_date = args.range_ending_on
    elif args.near_date:
        forward_days = width_days // 2
        end_date = args.near_date + datetime.timedelta(days=forward_days)
        if end_date > datetime.datetime.today().date():
            end_date = datetime.datetime.today().date()

        start_date = end_date - datetime.timedelta(days=width_days)
    else:
        # 'if not near_date:'
        if args.range_ending_on:
            end_date = args.range_ending_on
            start_date = end_date - datetime.timedelta(days=width_days)

        if args.range_starting_from:
            start_date = args.range_starting_from
            end_date = args.range_starting_from + datetime.timedelta(days=width_days)

    # ideally this shouldnt happen, and I havent seen it happen yet
    # but trying to come up with some rules for how to handle inconsistent or
    # extra inputs
    if args.range_starting_from:
        start_date = min(start_date, args.range_starting_from)

    if args.range_ending_on:
        end_date = max(end_date, args.range_ending_on)

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


class GetStartOfDateRangeInput(ToolArgs):
    date_range: DateRange


@tool(
    description="""
    This function returns a date range object representing the start or beginning of
    the  input date_range with start_date and end_date both equal to the start_date of the
    input date_range.
    You might need this when client refers to one larger date range needed for the overall request
    but another part of the plan needs access to the start of that date_range
    """,
    category=ToolCategory.DATES,
    tool_registry=default_tool_registry(),
    is_visible=False,
)
async def get_start_of_date_range(
    args: GetStartOfDateRangeInput, context: PlanRunContext
) -> DateRange:
    await tool_log(
        log=(f"{args.date_range.start_date.isoformat()}"),
        context=context,
    )

    return DateRange(start_date=args.date_range.start_date, end_date=args.date_range.start_date)


class GetEndOfDateRangeInput(ToolArgs):
    date_range: DateRange


@tool(
    description="""
    This function returns a date range object representing the end of the input date_range
    with start_date and end_date both equal to the end_date of the input date_range.
    You might need this when client refers to one larger date range needed for the overall request
    but another part of the plan needs access to the end of that date_range
    """,
    category=ToolCategory.DATES,
    tool_registry=default_tool_registry(),
    is_visible=False,
    enabled=False,
)
async def get_end_of_date_range(args: GetEndOfDateRangeInput, context: PlanRunContext) -> DateRange:
    await tool_log(
        log=(f"{args.date_range.end_date.isoformat()}"),
        context=context,
    )
    return DateRange(start_date=args.date_range.end_date, end_date=args.date_range.end_date)


async def main() -> None:
    # input_text = "Need a summary of all the earnings calls since Dec 3rd that might impact stocks in the TSX composite"  # noqa: E501
    input_text = (
        "Graph Apple's stock price starting from the beginning of last year to 3 months ago"  # noqa: E501
    )
    user_message = Message(message=input_text, is_user_message=True, message_time=get_now_utc())
    chat_context = ChatContext(messages=[user_message])
    plan_context = PlanRunContext(
        agent_id="123",
        plan_id="123",
        user_id="123",
        plan_run_id="123",
        chat=chat_context,
        run_tasks_without_prefect=True,
        skip_db_commit=True,
    )

    date_range = await get_date_range(
        GetDateRangeInput(date_range_str="from the beginning of last year to 3 months ago"),
        plan_context,
    )  # Get the date for one month ago

    print(date_range)


if __name__ == "__main__":
    asyncio.run(main())
