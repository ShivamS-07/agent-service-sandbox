from agent_service.io_type_utils import IOType
from agent_service.tool import ToolArgs, ToolCategory, ToolRegistry, tool
from agent_service.types import PlanRunContext
from agent_service.utils.output_utils.output_construction import PreparedOutput


class OutputArgs(ToolArgs):
    object_to_output: IOType
    title: str


@tool(
    description="""This is a very important function that displays any object
to the user, along with a short title for the object. EVERY plan you come up with
should have at least one call to this function. Objects will be displayed in the
same order the function is called in. For example, assuming there are three
objects stored in variables called "text", "graph", and "table", the calls:

   result1 = output(object_to_output=text, title="Description of Stocks")
   result2 = output(object_to_output=graph, title="Price Graph")
   result3 = output(object_to_output=table, title="Table of Weights")

Will show the objects top to bottom in that order. Please display only what the
user asks for, and no other extraneous information. Please only display every
variable once, otherwise the user will see duplicates. Remember you need to
assign this function call to a variable!!
""",
    category=ToolCategory.OUTPUT,
    is_visible=False,
    is_output_tool=True,
    store_output=False,
)
async def prepare_output(args: OutputArgs, context: PlanRunContext) -> PreparedOutput:
    return PreparedOutput(title=args.title, val=args.object_to_output)


# Hack for backwards compatibility
ToolRegistry._REGISTRY_CATEGORY_MAP[ToolCategory.OUTPUT]["output"] = (
    ToolRegistry._REGISTRY_CATEGORY_MAP[ToolCategory.OUTPUT]["prepare_output"]
)
ToolRegistry._REGISTRY_ALL_TOOLS_MAP["output"] = ToolRegistry._REGISTRY_CATEGORY_MAP[
    ToolCategory.OUTPUT
]["prepare_output"]
