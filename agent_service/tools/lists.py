from typing import List

from agent_service.io_type_utils import IOType
from agent_service.tool import ToolArgs, ToolCategory, ToolRegistry, tool
from agent_service.types import PlanRunContext


class FlattenListsInput(ToolArgs):
    lists_of_lists: List[List[IOType]]


@tool(
    description="This function flattens a list of lists into a list",
    category=ToolCategory.LIST,
    tool_registry=ToolRegistry,
)
async def flatten_lists_of_lists(args: FlattenListsInput, context: PlanRunContext) -> List[IOType]:
    return [item for inner_list in FlattenListsInput.lists_of_lists for item in inner_list]
