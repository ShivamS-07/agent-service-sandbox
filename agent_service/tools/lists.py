from typing import List

from agent_service.io_type_utils import IOType
from agent_service.tool import ToolArgs, ToolCategory, ToolRegistry, tool
from agent_service.types import PlanRunContext


class CollapseListsInput(ToolArgs):
    lists_of_lists: List[List[IOType]]


@tool(
    description="This function flattens a list of lists into a list",
    category=ToolCategory.LIST,
    tool_registry=ToolRegistry,
    is_visible=False,
)
async def collapse_lists(args: CollapseListsInput, context: PlanRunContext) -> List[IOType]:
    return [item for inner_list in args.lists_of_lists for item in inner_list]


class GetIndexInput(ToolArgs):
    list: List[IOType]
    index: int


@tool(
    description="Get the nth element of a list. You must use this instead of the Python indexing ([])",
    category=ToolCategory.LIST,
    tool_registry=ToolRegistry,
    is_visible=False,
)
async def get_element_from_list(args: GetIndexInput, context: PlanRunContext) -> IOType:
    return args.list[args.index]
