import datetime
import math
from typing import Optional, Union

import dateparser

from agent_service.io_types.dates import DateRange
from agent_service.tool import ToolArgs, ToolCategory, ToolRegistry, tool
from agent_service.types import PlanRunContext
from agent_service.utils.prefect import get_prefect_logger

logger = get_prefect_logger(__name__)


class DateFromDateStrInput(ToolArgs):
    date_str: str


@tool(
    description=(
        "This function takes a string which refers to a time, either absolute or"
        " relative to the current time, and converts it to a Python date. "
        "This uses the python dateparser package, so any input should be compatible with that."
        " You should always input some string, never an empty string."
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
        raise ValueError(f"Unable to parse date string '{args.date_str}'")
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
        "This function returns a date range 'DateRange' with a start_date and end_date "
        "that are chosen to be relative to and near to the one of the passed in dates "
        "or near today's date if no date was passed in. "
        "it is useful for when a you need to a date range for a graph but were provided no "
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
    logger.info(f"returning {res=}")
    return res
