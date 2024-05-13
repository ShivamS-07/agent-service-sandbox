from typing import Dict, Optional
from uuid import uuid4

from prefect import flow
from prefect.testing.utilities import prefect_test_harness

from agent_service.io_type_utils import IOType
from agent_service.planner.constants import (
    CREATE_EXECUTION_PLAN_FLOW_NAME,
    RUN_EXECUTION_PLAN_FLOW_NAME,
)
from agent_service.planner.planner import Planner
from agent_service.planner.planner_types import ExecutionPlan, Variable
from agent_service.tool import ToolRegistry
from agent_service.types import PlanRunContext
from agent_service.utils.postgres import get_psql


@flow(name=RUN_EXECUTION_PLAN_FLOW_NAME, flow_run_name="{context.plan_run_id}")
async def run_execution_plan(plan: ExecutionPlan, context: PlanRunContext) -> Optional[IOType]:
    # Maps variables to their resolved values
    variable_lookup: Dict[str, IOType] = {}
    db = get_psql(skip_commit=context.skip_db_commit)

    tool_output = None
    for step in plan.nodes:
        tool = ToolRegistry.get_tool(step.tool_name)
        # First, resolve the variables
        resolved_args = {}
        for arg, val in step.args.items():
            if isinstance(val, Variable):
                variable_value = variable_lookup[val.var_name]
                resolved_args[arg] = variable_value
            elif isinstance(val, list):
                actual_list = []
                for item in val:
                    if isinstance(item, Variable):
                        variable_value = variable_lookup[item.var_name]
                        actual_list.append(variable_value)
                    else:
                        actual_list.append(item)
                resolved_args[arg] = actual_list
            else:
                resolved_args[arg] = val

        # Now, we can create the input argument type
        step_args = tool.input_type(**resolved_args)

        # Create the context
        context.task_id = step.tool_task_id

        # Run the tool, store its output
        tool_output = await tool.func(args=step_args, context=context)
        if not context.skip_db_commit:
            db.write_tool_output(output=tool_output, context=context)

        # Store the output in the associated variable
        if step.output_variable_name:
            variable_lookup[step.output_variable_name] = tool_output

        # Update the chat context in case of new messages
        chat_context = db.get_chat(agent_id=context.agent_id)
        context.chat = chat_context

    return tool_output


@flow(name=CREATE_EXECUTION_PLAN_FLOW_NAME, flow_run_name="{plan_id}")
async def create_execution_plan(
    agent_id: str,
    plan_id: str,
    user_id: str,
    skip_db_commit: bool = False,
    skip_task_cache: bool = False,
    run_plan_immediately: bool = True,
) -> ExecutionPlan:
    planner = Planner(agent_id=agent_id)
    db = get_psql(skip_commit=skip_db_commit)
    plan = await planner.create_initial_plan(chat_context=db.get_chat(agent_id=agent_id))
    db.write_execution_plan(plan_id=plan_id, agent_id=agent_id, plan=plan)

    if not run_plan_immediately:
        return plan

    _ = PlanRunContext(  # TODO
        agent_id=agent_id,
        plan_id=plan_id,
        user_id=user_id,
        plan_run_id=str(uuid4()),
        skip_db_commit=skip_db_commit,
        skip_task_cache=skip_task_cache,
    )
    # TODO once prefect is setup
    # kickoff_execution_plan(plan, context)

    return plan


# Run these in tests or if you don't want to connect to the prefect server.
async def run_execution_plan_local(
    plan: ExecutionPlan, context: PlanRunContext
) -> Optional[IOType]:
    with prefect_test_harness():
        return await run_execution_plan(plan, context)


async def create_execution_plan_local(
    agent_id: str,
    plan_id: str,
    user_id: str,
    skip_db_commit: bool = False,
    skip_task_cache: bool = False,
    run_plan_immediately: bool = True,
) -> ExecutionPlan:
    with prefect_test_harness():
        return await create_execution_plan(
            agent_id=agent_id,
            plan_id=plan_id,
            user_id=user_id,
            skip_db_commit=skip_db_commit,
            skip_task_cache=skip_task_cache,
            run_plan_immediately=run_plan_immediately,
        )
