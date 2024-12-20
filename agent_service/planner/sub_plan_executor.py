from typing import Dict, List, Optional

from agent_service.io_type_utils import IOType, split_io_type_into_components
from agent_service.planner.planner_types import ExecutionPlan
from agent_service.tool import default_tool_registry
from agent_service.types import PlanRunContext
from agent_service.utils.prefect import get_prefect_logger


async def run_plan_simple(
    plan: ExecutionPlan,
    context: PlanRunContext,
    variable_lookup: Optional[Dict[str, IOType]] = None,
    supplemental_args: Optional[Dict[str, Dict[str, IOType]]] = None,
) -> List[IOType]:
    """
    Runs a plan very simply as a function that produces a list of outputs. No DB
    inserts outside of the usual tool call stuff.
    Brings me back to the days when the executor was simple :')

    supplemental_args maps task ID to a (arg name, value) mapping. These will be
    passed into the args for that task ID.
    """
    logger = get_prefect_logger(__name__)
    final_outputs: List[IOType] = []
    variable_lookup = variable_lookup or {}

    for step in plan.nodes:
        try:
            tool = default_tool_registry().get_tool(step.tool_name)
            resolved_args = step.resolve_arguments(variable_lookup=variable_lookup)
            if supplemental_args and step.tool_task_id in supplemental_args:
                resolved_args.update(supplemental_args[step.tool_task_id])
            step_args = tool.input_type(**resolved_args)
            context.task_id = step.tool_task_id
            context.tool_name = step.tool_name

            tool_output = await tool.func(args=step_args, context=context)

            if step.output_variable_name:
                variable_lookup[step.output_variable_name] = tool_output
            if step.is_output_node:
                split_outputs = await split_io_type_into_components(tool_output)
                final_outputs.extend(split_outputs)
        except Exception:
            logger.exception(f"Failed in {step=}, {context=}")
            return []

    return final_outputs
