import datetime

import dateparser

from agent_service.tool import ToolArgs, ToolCategory, ToolRegistry, tool
from agent_service.types import PlanRunContext


class DateFromDateStrInput(ToolArgs):
    date_str: str


@tool(
    description=(
        "This function takes a string which refers to a time, either absolute or"
        " relative to the current time, and converts it to a Python date. "
        "This uses the python dateparser package, so any input should be compatible with that."
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
