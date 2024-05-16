from typing import Dict, Optional
from uuid import uuid4

from prefect import flow

from agent_service.chatbot.chatbot import Chatbot
from agent_service.io_type_utils import IOType
from agent_service.planner.constants import (
    CREATE_EXECUTION_PLAN_FLOW_NAME,
    RUN_EXECUTION_PLAN_FLOW_NAME,
)
from agent_service.planner.planner import Planner
from agent_service.planner.planner_types import ExecutionPlan, Variable
from agent_service.tool import ToolRegistry
from agent_service.types import ChatContext, Message, PlanRunContext
from agent_service.utils.postgres import get_psql
from agent_service.utils.prefect import get_prefect_logger, prefect_run_execution_plan


@flow(name=RUN_EXECUTION_PLAN_FLOW_NAME, flow_run_name="{context.plan_run_id}")
async def run_execution_plan(
    plan: ExecutionPlan,
    context: PlanRunContext,
    send_chat_when_finished: bool = True,
    log_all_outputs: bool = False,
) -> Optional[IOType]:
    logger = get_prefect_logger(__name__)
    # Maps variables to their resolved values
    variable_lookup: Dict[str, IOType] = {}
    db = get_psql(skip_commit=context.skip_db_commit)

    if not context.skip_db_commit:
        db.insert_plan_run(
            agent_id=context.agent_id, plan_id=context.plan_id, plan_run_id=context.plan_run_id
        )

    tool_output = None
    for step in plan.nodes:
        logger.info(f"Running step '{step.tool_name}'")
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
        if log_all_outputs:
            logger.info(f"Output of step '{step.tool_name}': {tool_output}")

        # Store the output in the associated variable
        if step.output_variable_name:
            variable_lookup[step.output_variable_name] = tool_output

        # Update the chat context in case of new messages
        if not context.skip_db_commit:
            context.chat = db.get_chats_history_for_agent(agent_id=context.agent_id)
        logger.info(f"Finished step '{step.tool_name}'")

    # TODO right now we don't handle output tools, and we just output the last
    # thing. Should fix that.

    logger.info(f"Finished running {context.agent_id=}, {context.plan_id=}, {context.plan_run_id=}")
    logger.info("Generating chat message...")
    if send_chat_when_finished and not context.skip_db_commit:
        chatbot = Chatbot(agent_id=context.agent_id)
        message = await chatbot.generate_execution_complete_response(
            chat_context=db.get_chats_history_for_agent(agent_id=context.agent_id),
            execution_plan=plan,
            output=tool_output,
        )
        db.insert_chat_messages(
            messages=[Message(agent_id=context.agent_id, message=message, is_user_message=False)]
        )

    logger.info("Finished generating chat message, storing output in DB...")
    if not context.skip_db_commit:
        db.write_agent_output(output=tool_output, context=context)

    logger.info("Finished run!")
    return tool_output


@flow(name=CREATE_EXECUTION_PLAN_FLOW_NAME, flow_run_name="{plan_id}")
async def create_execution_plan(
    agent_id: str,
    plan_id: str,
    user_id: str,
    skip_db_commit: bool = False,
    skip_task_cache: bool = False,
    run_plan_in_prefect_immediately: bool = True,
    run_tasks_without_prefect: bool = False,
    send_chat_when_finished: bool = True,
    chat_context: Optional[ChatContext] = None,
) -> ExecutionPlan:
    logger = get_prefect_logger(__name__)
    planner = Planner(agent_id=agent_id)
    db = get_psql(skip_commit=skip_db_commit)

    logger.info(f"Starting creation of execution plan for {agent_id=}...")
    chat_context = chat_context or db.get_chats_history_for_agent(agent_id=agent_id)
    plan = await planner.create_initial_plan(chat_context=chat_context)
    if not skip_db_commit:
        db.write_execution_plan(plan_id=plan_id, agent_id=agent_id, plan=plan)
    logger.info(f"Finished creating execution plan for {agent_id=}")
    logger.info(f"Execution Plan\n:{plan.get_formatted_plan()}")
    if not run_plan_in_prefect_immediately:
        return plan

    plan_run_id = str(uuid4())
    ctx = PlanRunContext(
        agent_id=agent_id,
        plan_id=plan_id,
        user_id=user_id,
        plan_run_id=plan_run_id,
        skip_db_commit=skip_db_commit,
        skip_task_cache=skip_task_cache,
        run_tasks_without_prefect=run_tasks_without_prefect,
        chat=chat_context,
    )

    logger.info(f"Running execution plan for {agent_id=}, {plan_id=}, {plan_run_id=}")
    await prefect_run_execution_plan(
        plan=plan, context=ctx, send_chat_when_finished=send_chat_when_finished
    )

    if send_chat_when_finished and not skip_db_commit:
        chatbot = Chatbot(agent_id=agent_id)
        message = await chatbot.generate_initial_postplan_response(
            chat_context=db.get_chats_history_for_agent(agent_id=agent_id), execution_plan=plan
        )
        db.insert_chat_messages(
            messages=[Message(agent_id=agent_id, message=message, is_user_message=False)]
        )

    return plan


# Run these in tests or if you don't want to connect to the prefect server.
async def run_execution_plan_local(
    plan: ExecutionPlan,
    context: PlanRunContext,
    send_chat_when_finished: bool = False,
    log_all_outputs: bool = False,
) -> Optional[IOType]:
    context.run_tasks_without_prefect = True
    return await run_execution_plan.fn(
        plan, context, send_chat_when_finished, log_all_outputs=log_all_outputs
    )


async def create_execution_plan_local(
    agent_id: str,
    plan_id: str,
    user_id: str,
    skip_db_commit: bool = False,
    skip_task_cache: bool = False,
    run_plan_in_prefect_immediately: bool = True,
    run_tasks_without_prefect: bool = True,
    chat_context: Optional[ChatContext] = None,
) -> ExecutionPlan:
    return await create_execution_plan.fn(
        agent_id=agent_id,
        plan_id=plan_id,
        user_id=user_id,
        skip_db_commit=skip_db_commit,
        skip_task_cache=skip_task_cache,
        run_plan_in_prefect_immediately=run_plan_in_prefect_immediately,
        run_tasks_without_prefect=run_tasks_without_prefect,
        chat_context=chat_context,
    )
