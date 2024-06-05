from typing import List

from agent_service.io_type_utils import IOType
from agent_service.tool import ToolArgs, ToolCategory, tool
from agent_service.types import PlanRunContext


class OutputArgs(ToolArgs):
    objects_to_output: List[IOType]


@tool(
    description="""This is a very important function that displays any object
to the user.  EVERY plan you come up with should have at least one call to this
function. Objects will be displayed in the same order the function is called
in. For example, assuming there are three objects stored in variables called
"text", "graph", and "table", the call:

   result = output([text, graph, table])

Will show the objects top to bottom in that order. Please display only what the
user asks for, and no other extraneous information.
""",
    category=ToolCategory.OUTPUT,
    is_visible=False,
    is_output_tool=True,
)
async def output(args: OutputArgs, context: PlanRunContext) -> List[IOType]:
    # For now does nothing
    return args.objects_to_output
