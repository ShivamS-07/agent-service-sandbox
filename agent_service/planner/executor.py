from typing import Dict, Optional
from uuid import uuid4

from prefect import flow

from agent_service.chatbot.chatbot import Chatbot
from agent_service.io_type_utils import IOType
from agent_service.planner.action_decide import Action, ActionDecider
from agent_service.planner.constants import (
    CREATE_EXECUTION_PLAN_FLOW_NAME,
    RUN_EXECUTION_PLAN_FLOW_NAME,
)
from agent_service.planner.planner import Planner
from agent_service.planner.planner_types import ExecutionPlan, Variable
from agent_service.tool import ToolRegistry
from agent_service.types import ChatContext, Message, PlanRunContext
from agent_service.utils.postgres import get_psql
from agent_service.utils.prefect import (
    FlowRunType,
    get_prefect_logger,
    prefect_cancel_agent_flow,
    prefect_create_execution_plan,
    prefect_get_current_plan_run_task_id,
    prefect_pause_current_agent_flow,
    prefect_resume_agent_flow,
    prefect_run_execution_plan,
)


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
    action: Action = Action.CREATE,
    skip_db_commit: bool = False,
    skip_task_cache: bool = False,
    run_plan_in_prefect_immediately: bool = True,
    run_tasks_without_prefect: bool = False,
    send_chat_when_finished: bool = True,
    chat_context: Optional[ChatContext] = None,
) -> ExecutionPlan:
    if Action != Action.CREATE:
        await rewrite_execution_plan_after_input(
            agent_id=agent_id,
            user_id=user_id,
            plan_id=plan_id,
            action=action,
            skip_db_commit=skip_db_commit,
            skip_task_cache=skip_task_cache,
            run_plan_in_prefect_immediately=run_plan_in_prefect_immediately,
            run_tasks_without_prefect=run_tasks_without_prefect,
            send_chat_when_finished=send_chat_when_finished,
            chat_context=chat_context,
        )

    logger = get_prefect_logger(__name__)
    planner = Planner(agent_id=agent_id)
    db = get_psql(skip_commit=skip_db_commit)

    logger.info(f"Starting creation of execution plan for {agent_id=}...")
    chat_context = chat_context or db.get_chats_history_for_agent(agent_id=agent_id)
    plan = await planner.create_initial_plan(chat_context=chat_context)
    if not skip_db_commit:
        db.write_execution_plan(plan_id=plan_id, agent_id=agent_id, plan=plan)
    logger.info(f"Finished creating execution plan for {agent_id=}")
    logger.info(f"Execution Plan:\n{plan.get_formatted_plan()}")
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


# Update logic
# First, pause current plan
# Then check to see what action we should take
# If no action, then just simple chatbot prompt (TODO) and resume plan
# If rerun, then check to see if current step is before step that requires
# rerun, if so, can resume, otherwise cancel current run and restart run
# if append or replan, update the plan and rerun


async def update_execution_after_input(
    agent_id: str,
    user_id: str,
    skip_db_commit: bool = False,
    skip_task_cache: bool = False,
    run_plan_in_prefect_immediately: bool = True,
    run_tasks_without_prefect: bool = False,
    send_chat_when_finished: bool = True,
    chat_context: Optional[ChatContext] = None,
) -> None:
    logger = get_prefect_logger(__name__)
    decider = ActionDecider(agent_id=agent_id)
    db = get_psql(skip_commit=skip_db_commit)
    plan_run_ids = db.get_agent_plan_runs(agent_id=agent_id, limit_num=1)
    if not plan_run_ids:
        # TODO No plan is running, what to do here?
        return
    current_plan_run = plan_run_ids[0]

    flow_run = await prefect_pause_current_agent_flow(current_plan_run)

    chat_context = chat_context or db.get_chats_history_for_agent(agent_id=agent_id)

    latest_plan_id, latest_plan = db.get_latest_execution_plan(agent_id)

    if latest_plan is None or latest_plan_id is None:
        # this could happen if original plan isn't done?
        # we should probably either cancel original plan OR wait
        return

    # for now we are assuming that the last message in the chat context is the relevant message

    action = await decider.decide_action(chat_context, latest_plan)

    if action == Action.NONE or (
        action == Action.RERUN
        and (flow_run and flow_run.flow_run_type == FlowRunType.PLAN_CREATION)
    ):
        # TODO add some chatbot response
        if flow_run:
            await prefect_resume_agent_flow(flow_run)
    elif action == Action.RERUN:
        current_task_id = prefect_get_current_plan_run_task_id(current_plan_run)
        for node in latest_plan.nodes:
            if ToolRegistry.does_tool_read_chat(node.tool_name):
                # we've already run into a chat reading node, which means we need to rerun
                if flow_run:
                    await prefect_cancel_agent_flow(flow_run)
                plan_run_id = str(uuid4())
                ctx = PlanRunContext(
                    agent_id=agent_id,
                    plan_id=latest_plan_id,
                    user_id=user_id,
                    plan_run_id=plan_run_id,
                    skip_db_commit=skip_db_commit,
                    skip_task_cache=skip_task_cache,
                    run_tasks_without_prefect=run_tasks_without_prefect,
                    chat=chat_context,
                )

                # TODO: Add some Chatbot response

                logger.info(
                    f"Rerunning execution plan for {agent_id=}, {latest_plan_id=}, {plan_run_id=}"
                )
                await prefect_run_execution_plan(
                    plan=latest_plan, context=ctx, send_chat_when_finished=send_chat_when_finished
                )
                break

            if node.tool_task_id == current_task_id:
                # if we got here without breaking, means no chat reading node
                # has been run, we can just resume
                if flow_run:
                    await prefect_resume_agent_flow(flow_run)

    else:
        if flow_run:
            await prefect_cancel_agent_flow(flow_run)
        new_plan_id = uuid4()
        await prefect_create_execution_plan(
            agent_id,
            user_id,
            new_plan_id,
            action=action,
            skip_db_commit=skip_db_commit,
            skip_task_cache=skip_task_cache,
            run_plan_in_prefect_immediately=run_plan_in_prefect_immediately,
        )


async def rewrite_execution_plan_after_input(
    agent_id: str,
    user_id: str,
    plan_id: str,
    action: Action,
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

    logger.info(f"Starting rewrite of execution plan for {agent_id=}...")
    chat_context = chat_context or db.get_chats_history_for_agent(agent_id=agent_id)
    if action == Action.REPLAN or action == Action.APPEND:
        # treat these the same for now
        # TODO: do append more efficiently, with a different prompt
        # TODO: get all exectution plans, not just the most recent one
        # TODO: Add some chatbot response before we redo the plan
        # To stop potential circular behavior
        old_plan = db.get_latest_execution_plan(agent_id)
        new_plan = await planner.rewrite_plan_after_input(chat_context, old_plan)

    if not skip_db_commit:
        db.write_execution_plan(plan_id=plan_id, agent_id=agent_id, plan=new_plan)
    logger.info(f"Finished rewriting execution plan for {agent_id=}")
    logger.info(f"New Execution Plan:\n{new_plan.get_formatted_plan()}")
    if not run_plan_in_prefect_immediately:
        return new_plan

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

    logger.info(f"Running new execution plan for {agent_id=}, {plan_id=}, {plan_run_id=}")
    await prefect_run_execution_plan(
        plan=new_plan, context=ctx, send_chat_when_finished=send_chat_when_finished
    )

    if send_chat_when_finished and not skip_db_commit:
        chatbot = Chatbot(agent_id=agent_id)
        # TODO: Send a different chatbot response discussing the plan update
        message = await chatbot.generate_initial_postplan_response(
            chat_context=db.get_chats_history_for_agent(agent_id=agent_id), execution_plan=new_plan
        )
        db.insert_chat_messages(
            messages=[Message(agent_id=agent_id, message=message, is_user_message=False)]
        )

    return new_plan


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
