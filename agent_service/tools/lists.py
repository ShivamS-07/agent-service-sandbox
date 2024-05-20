from typing import List

from agent_service.io_type_utils import IOType
from agent_service.tool import ToolArgs, ToolCategory, ToolRegistry, tool
from agent_service.types import PlanRunContext


class AddListsInput(ToolArgs):
    list1: List[IOType]
    list2: List[IOType]


@tool(
    description=(
        "This function forms a single list from the elements of two lists. "
        "For example, [1, 2, 3] and [4, 5, 6] would add to [1, 2, 3, 4, 5, 6]."
    ),
    category=ToolCategory.LIST,
    tool_registry=ToolRegistry,
    is_visible=False,
    enabled=False,
)
async def add_lists(args: AddListsInput, context: PlanRunContext) -> List[IOType]:
    return args.list1 + args.list2


class GetIndexInput(ToolArgs):
    list: List[IOType]
    index: int


@tool(
    description="Get the nth element of a list. You must use this instead of the Python indexing ([])",
    category=ToolCategory.LIST,
    tool_registry=ToolRegistry,
    is_visible=False,
    enabled=False,
)
async def get_element_from_list(args: GetIndexInput, context: PlanRunContext) -> IOType:
    return args.list[args.index]


class GetListNInput(ToolArgs):
    list: List[IOType]
    n: int


@tool(
    description="Get the first N elements of a list. You must use this instead of the Python indexing ([])",
    category=ToolCategory.LIST,
    tool_registry=ToolRegistry,
    is_visible=False,
    enabled=False,
)
async def get_first_n_from_list(args: GetListNInput, context: PlanRunContext) -> IOType:
    return args.list[: args.n]


@tool(
    description="Get the last N elements of a list. You must use this instead of the Python indexing ([])",
    category=ToolCategory.LIST,
    tool_registry=ToolRegistry,
    is_visible=False,
    enabled=False,
)
async def get_last_n_from_list(args: GetListNInput, context: PlanRunContext) -> IOType:
    return args.list[-args.n :]
