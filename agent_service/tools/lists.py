from typing import List

from agent_service.io_type_utils import ComplexIOBase, IOType
from agent_service.tool import ToolArgs, ToolCategory, ToolRegistry, tool
from agent_service.tools.tool_log import tool_log
from agent_service.types import PlanRunContext


class CombineListsInput(ToolArgs):
    list1: List[IOType]
    list2: List[IOType]


@tool(
    description=(
        "This function forms a single deduplicated list from the elements of two lists. "
        " For example, [1, 2, 3] and [3, 4, 5] would add to [1, 2, 3, 4, 5]."
        " This is particularly useful if you created two lists of stocks or texts and want to"
        " put them together into a single list"
        " This is equivalent to `boolean OR` or `Union` logic, if you only want elements in "
        " both lists, use intersect_lists"
    ),
    category=ToolCategory.LIST,
    tool_registry=ToolRegistry,
    is_visible=False,
)
async def add_lists(args: CombineListsInput, context: PlanRunContext) -> List[IOType]:
    result: List[IOType] = []
    try:
        # Do this if the lists have complex io types in them
        result = list(ComplexIOBase.union_sets(set(args.list1), set(args.list2)))  # type: ignore
    except Exception:
        # otherwise just do a normal union
        result = list(set(args.list1 + args.list2))

    await tool_log(
        log=f"Merged list has {len(result)} items",
        context=context,
    )

    return result


@tool(
    description=(
        "This function forms a list of the elements included in both of two lists. "
        " For example, [1, 2, 3, 4] and [3, 4, 5] would intersect to to [3, 4]."
        " You will want to use this function if, for example, you have two lists of stocks that each have a certain"
        " property and you want a list of stocks with both properties, though note it usually better to apply filters "
        " iteratively on a single list rather than intersecting the output of two filters. "
        " This is equivalent to boolean AND logic, use add_lists if you want OR logic."
    ),
    category=ToolCategory.LIST,
    tool_registry=ToolRegistry,
    is_visible=False,
)
async def intersect_lists(args: CombineListsInput, context: PlanRunContext) -> List[IOType]:
    try:
        # Do this if the lists have complex io types in them
        result = list(ComplexIOBase.intersect_sets(set(args.list1), set(args.list2)))  # type: ignore
    except Exception:
        # otherwise just do a normal intersection
        result = list(set(args.list1) & set(args.list2))  # type: ignore

    await tool_log(
        log=f"Intersection has {len(result)} items",
        context=context,
    )

    return result  # type: ignore


@tool(
    description=(
        "This function outputs a list of things in the first list that are not in the second (set difference). "
        " For example, if list1 is [1, 2, 3, 4] and list2 is [3, 4, 5], the output diff is [1, 2]."
        " You will want to use this function if, for example, you want all the stocks in a particular universe"
        " except for a subset of them (e.g. R1k stocks excluding S&P 500 stocks)"
    ),
    category=ToolCategory.LIST,
    tool_registry=ToolRegistry,
    is_visible=False,
)
async def diff_lists(args: CombineListsInput, context: PlanRunContext) -> List[IOType]:
    result = list(set(args.list1) - set(args.list2))
    await tool_log(
        log=f"Difference has {len(result)} items",
        context=context,
    )
    return result


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
