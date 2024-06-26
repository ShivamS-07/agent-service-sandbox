import pprint
from typing import Dict, List, Optional, Tuple, Union
from uuid import uuid4

from prefect import flow

from agent_service.chatbot.chatbot import Chatbot
from agent_service.io_type_utils import IOType
from agent_service.planner.action_decide import (
    Action,
    ErrorActionDecider,
    InputActionDecider,
)
from agent_service.planner.constants import (
    CREATE_EXECUTION_PLAN_FLOW_NAME,
    EXECUTION_TRIES,
    RUN_EXECUTION_PLAN_FLOW_NAME,
    WORKLOG_INTERVAL,
)
from agent_service.planner.planner import Planner
from agent_service.planner.planner_types import (
    ErrorInfo,
    ExecutionPlan,
    PlanStatus,
    ToolExecutionNode,
    Variable,
)
from agent_service.tool import ToolRegistry
from agent_service.types import ChatContext, Message, PlanRunContext
from agent_service.utils.agent_event_utils import (
    publish_agent_execution_plan,
    publish_agent_output,
    publish_agent_plan_status,
    publish_agent_updated_worklogs,
    send_chat_message,
)
from agent_service.utils.async_db import (
    AsyncDB,
    get_chat_history_from_db,
    get_latest_execution_plan_from_db,
)
from agent_service.utils.date_utils import get_now_utc
from agent_service.utils.output_utils.output_diffs import OutputDiffer
from agent_service.utils.postgres import Postgres, SyncBoostedPG, get_psql
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


async def check_cancelled(
    db: Union[Postgres, AsyncDB], plan_id: Optional[str] = None, plan_run_id: Optional[str] = None
) -> bool:
    if not plan_id and not plan_run_id:
        return False
    ids = [val for val in (plan_id, plan_run_id) if val is not None]
    if isinstance(db, Postgres):
        return db.is_cancelled(ids_to_check=ids)
    else:
        return await db.is_cancelled(ids_to_check=ids)


@flow(name=RUN_EXECUTION_PLAN_FLOW_NAME, flow_run_name="{context.plan_run_id}")
async def run_execution_plan(
    plan: ExecutionPlan,
    context: PlanRunContext,
    do_chat: bool = True,
    log_all_outputs: bool = False,
    replan_execution_error: bool = True,
    run_plan_in_prefect_immediately: bool = True,
    # This is meant for testing, basically we can fill in the lookup table to
    # make sure we only run the plan starting from a certain point while passing
    # in precomputed outputs for prior tasks.
    override_task_output_lookup: Optional[Dict[str, IOType]] = None,
    scheduled_by_automation: bool = False,
) -> List[IOType]:
    logger = get_prefect_logger(__name__)
    # Maps variables to their resolved values
    variable_lookup: Dict[str, IOType] = {}
    db = get_psql(skip_commit=context.skip_db_commit)

    db.insert_plan_run(
        agent_id=context.agent_id, plan_id=context.plan_id, plan_run_id=context.plan_run_id
    )

    final_outputs = []
    tool_output = None
    prev_worklog_publish_time = get_now_utc()
    for idx, step in enumerate(plan.nodes):
        # Check both the plan_id and plan_run_id to prevent race conditions
        if await check_cancelled(db=db, plan_id=context.plan_id, plan_run_id=context.plan_run_id):
            raise Exception("Execution plan has been cancelled")
        now = get_now_utc()
        logger.info(f"Running step '{step.tool_name}' (Task ID: {step.tool_task_id})")
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

        # if the tool output already exists in the map, just use that
        if override_task_output_lookup and step.tool_task_id in override_task_output_lookup:
            logger.info(f"Step '{step.tool_name}' already in task lookup, using existing value...")
            tool_output = override_task_output_lookup[step.tool_task_id]
        else:
            # Run the tool, store its output, errors and replan
            try:
                tool_output = await tool.func(args=step_args, context=context)
            except Exception as e:
                logger.exception(f"Step '{step.tool_name}' failed due to {e}")
                if await check_cancelled(
                    db=db, plan_id=context.plan_id, plan_run_id=context.plan_run_id
                ):
                    # NEVER replan if the plan is already cancelled.
                    raise Exception("Execution plan has been cancelled")
                retrying = False
                if replan_execution_error:
                    retrying = await handle_error_in_execution(context, e, step, do_chat)

                if retrying:
                    raise RuntimeError("Plan run attempt failed, retrying")
                raise RuntimeError("All retry attempts failed")

        if not step.is_output_node:
            if step.store_output:
                db.write_tool_output(output=tool_output, context=context)
        else:
            # We have an output node
            await publish_agent_output(outputs=[tool_output], context=context, db=db)
        if log_all_outputs:
            logger.info(f"Output of step '{step.tool_name}': {tool_output}")

        if (
            idx == 0
            or (idx == len(plan.nodes) - 1)
            or (now - prev_worklog_publish_time).seconds >= WORKLOG_INTERVAL
        ):
            try:
                # always publish the first/last output, and also publish every second
                logger.info(f"Publishing updated worklogs at step '{step.tool_name}'...")
                prev_worklog_publish_time = now
                await publish_agent_updated_worklogs(context, plan, db)
            except Exception as e:
                # event publishing shouldn't affect the main flow
                logger.exception(f"Failed to publish worklogs at step '{step.tool_name}': {e}")

        # Store the output in the associated variable
        if step.output_variable_name:
            variable_lookup[step.output_variable_name] = tool_output

        # Update the chat context in case of new messages
        if not context.skip_db_commit:
            context.chat = db.get_chats_history_for_agent(agent_id=context.agent_id)
        if step.is_output_node:
            final_outputs.append(tool_output)
        logger.info(f"Finished step '{step.tool_name}'")

    if await check_cancelled(db=db, plan_id=context.plan_id, plan_run_id=context.plan_run_id):
        raise Exception("Execution plan has been cancelled")

    logger.info(f"Finished running {context.agent_id=}, {context.plan_id=}, {context.plan_run_id=}")
    if not scheduled_by_automation and do_chat:
        logger.info("Generating chat message...")
        chatbot = Chatbot(agent_id=context.agent_id)
        message = await chatbot.generate_execution_complete_response(
            chat_context=db.get_chats_history_for_agent(agent_id=context.agent_id),
            execution_plan=plan,
            outputs=final_outputs,
        )
        await send_chat_message(
            message=Message(agent_id=context.agent_id, message=message, is_user_message=False),
            db=db,
        )
    elif scheduled_by_automation:
        logger.info("Generating diff message...")
        output_differ = OutputDiffer(plan=plan, context=context)
        output_diffs = await output_differ.diff_outputs(
            latest_outputs=final_outputs, db=SyncBoostedPG(skip_commit=context.skip_db_commit)
        )
        logger.info(f"Got output diffs: {output_diffs}")
        important_diffs = [diff for diff in output_diffs if diff.should_notify]
        if not important_diffs:
            await send_chat_message(
                message=Message(
                    agent_id=context.agent_id,
                    message="Agent updated, no important differences found.",
                    is_user_message=False,
                    visible_to_llm=False,
                ),
                db=db,
                send_notification=False,
            )
        else:
            for diff in important_diffs:
                await send_chat_message(
                    message=Message(
                        agent_id=context.agent_id,
                        message=diff.diff_summary_message,
                        is_user_message=False,
                        visible_to_llm=False,
                    ),
                    db=db,
                )

    logger.info("Finished run!")
    return final_outputs


@flow(name=CREATE_EXECUTION_PLAN_FLOW_NAME, flow_run_name="{plan_id}")
async def create_execution_plan(
    agent_id: str,
    plan_id: str,
    user_id: str,
    action: Action = Action.CREATE,
    error_info: Optional[ErrorInfo] = None,
    skip_db_commit: bool = False,
    skip_task_cache: bool = False,
    run_plan_in_prefect_immediately: bool = True,
    run_tasks_without_prefect: bool = False,
    do_chat: bool = True,
    chat_context: Optional[ChatContext] = None,
    use_sample_plans: bool = True,
) -> Optional[ExecutionPlan]:
    db = AsyncDB(pg=SyncBoostedPG(skip_commit=skip_db_commit))
    if action != Action.CREATE:
        return await rewrite_execution_plan(
            agent_id=agent_id,
            user_id=user_id,
            plan_id=plan_id,
            action=action,
            error_info=error_info,
            skip_db_commit=skip_db_commit,
            skip_task_cache=skip_task_cache,
            run_plan_in_prefect_immediately=run_plan_in_prefect_immediately,
            run_tasks_without_prefect=run_tasks_without_prefect,
            do_chat=do_chat,
            chat_context=chat_context,
            use_sample_plans=use_sample_plans,
            db=db,
        )

    logger = get_prefect_logger(__name__)
    planner = Planner(agent_id=agent_id, skip_db_commit=skip_db_commit, send_chat=do_chat)

    logger.info(f"Starting creation of execution plan for {agent_id=}...")
    await publish_agent_plan_status(
        agent_id=agent_id, plan_id=plan_id, status=PlanStatus.CREATING, db=db
    )

    chat_context = chat_context or await get_chat_history_from_db(agent_id, db)
    if await check_cancelled(db=db, plan_id=plan_id):
        await publish_agent_plan_status(
            agent_id=agent_id, plan_id=plan_id, status=PlanStatus.CANCELLED, db=db
        )
        raise RuntimeError("Plan creation has been cancelled.")
    plan = await planner.create_initial_plan(
        chat_context=chat_context, use_sample_plans=use_sample_plans
    )
    if await check_cancelled(db=db, plan_id=plan_id):
        await publish_agent_plan_status(
            agent_id=agent_id, plan_id=plan_id, status=PlanStatus.CANCELLED, db=db
        )
        raise RuntimeError("Plan creation has been cancelled.")
    if plan is None:
        if do_chat:
            chatbot = Chatbot(agent_id=agent_id)
            message = await chatbot.generate_initial_plan_failed_response_suggestions(
                chat_context=chat_context
            )
            await send_chat_message(
                message=Message(agent_id=agent_id, message=message, is_user_message=False),
                db=db,
            )
        await publish_agent_plan_status(
            agent_id=agent_id, plan_id=plan_id, status=PlanStatus.FAILED, db=db
        )
        raise RuntimeError("Failed to create execution plan!")

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

    await publish_agent_execution_plan(plan, ctx, db)

    logger.info(f"Finished creating execution plan for {agent_id=}")
    logger.info(f"Execution Plan:\n{plan.get_formatted_plan()}")

    if do_chat:
        logger.info("Generating initial postplan response...")
        chatbot = Chatbot(agent_id=agent_id)
        message = await chatbot.generate_initial_postplan_response(
            chat_context=await get_chat_history_from_db(agent_id, db), execution_plan=plan
        )
        await send_chat_message(
            message=Message(agent_id=agent_id, message=message, is_user_message=False), db=db
        )

    if not run_plan_in_prefect_immediately:
        return plan

    logger.info(f"Submitting execution plan to Prefect for {agent_id=}, {plan_id=}, {plan_run_id=}")
    # Check one more time for cancellation
    if await check_cancelled(db=db, plan_id=plan_id):
        await publish_agent_plan_status(
            agent_id=agent_id, plan_id=plan_id, status=PlanStatus.CANCELLED, db=db
        )
        raise RuntimeError("Plan creation has been cancelled.")
    await prefect_run_execution_plan(plan=plan, context=ctx, do_chat=do_chat)
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
    do_chat: bool = True,
    chat_context: Optional[ChatContext] = None,
    use_sample_plans: bool = True,
) -> Optional[Tuple[str, ExecutionPlan, Action]]:
    logger = get_prefect_logger(__name__)
    decider = InputActionDecider(agent_id=agent_id)
    db = get_psql(skip_commit=skip_db_commit)

    flow_run = await prefect_pause_current_agent_flow(agent_id=agent_id)

    chat_context = chat_context or db.get_chats_history_for_agent(agent_id=agent_id)

    chatbot = Chatbot(agent_id=agent_id)

    latest_plan_id, latest_plan, _, _ = db.get_latest_execution_plan(agent_id)

    if latest_plan is None or latest_plan_id is None:
        # we must still be creating the first plan, let's just redo it
        # with this latest info
        action = Action.CREATE
        # for mypy
        latest_plan = ExecutionPlan(nodes=[])
        latest_plan_id = ""
    else:
        # for now we are assuming that the last message in the chat context is the relevant message
        action = await decider.decide_action(chat_context, latest_plan)

    logger.info(f"Decided on action: {action} for {agent_id=}, {latest_plan_id=}")
    # For now, no layout metatool
    if action == Action.LAYOUT:
        action = Action.NONE

    if action == Action.NONE or (
        action == Action.RERUN
        and (flow_run and flow_run.flow_run_type == FlowRunType.PLAN_CREATION)
    ):
        if do_chat:
            message = await chatbot.generate_input_update_no_action_response(chat_context)
            await send_chat_message(
                message=Message(agent_id=agent_id, message=message, is_user_message=False), db=db
            )
        if flow_run:
            await prefect_resume_agent_flow(flow_run)
        return None
    elif action == Action.RERUN:
        # In this case, we know that the flow_run_type is PLAN_EXECUTION (or there no flow_run),
        # otherwise we'd have run the above block instead.
        current_task_id = None
        seen_chat_reading_tool = False
        if flow_run:
            current_task_id = await prefect_get_current_plan_run_task_id(flow_run)
        for node in latest_plan.nodes:
            if ToolRegistry.does_tool_read_chat(node.tool_name):
                seen_chat_reading_tool = True
                # we've already run into a chat reading node, which means we need to rerun
                if do_chat:
                    message = await chatbot.generate_input_update_rerun_response(
                        chat_context, latest_plan, str(node.tool_name)
                    )
                    await send_chat_message(
                        message=Message(agent_id=agent_id, message=message, is_user_message=False),
                        db=db,
                    )

                if run_tasks_without_prefect:
                    return latest_plan_id, latest_plan, action

                if flow_run:
                    await prefect_cancel_agent_flow(flow_run, db=db)
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
                await prefect_run_execution_plan(plan=latest_plan, context=ctx, do_chat=do_chat)
                break

            if node.tool_task_id == current_task_id:
                # if we got here without breaking, means no chat reading node
                # has been run, we can just resume
                if do_chat:
                    message = await chatbot.generate_input_update_no_action_response(chat_context)
                    await send_chat_message(
                        message=Message(agent_id=agent_id, message=message, is_user_message=False),
                        db=db,
                    )

                if flow_run:
                    await prefect_resume_agent_flow(flow_run)

            if do_chat and not seen_chat_reading_tool and not current_task_id:
                # At this point, we've had no tasks that are in progress and also no
                # tasks that read the chat. In this case, we don't actually need to
                # do anything, and really GPT should have picked the NONE option.
                message = await chatbot.generate_input_update_no_action_response(chat_context)
                await send_chat_message(
                    message=Message(agent_id=agent_id, message=message, is_user_message=False),
                    db=db,
                )

    else:
        # This handles the cases for REPLAN, APPEND, and CREATE
        if do_chat:
            message = await chatbot.generate_input_update_replan_preplan_response(chat_context)
            await send_chat_message(
                message=Message(agent_id=agent_id, message=message, is_user_message=False), db=db
            )
        if flow_run:
            await prefect_cancel_agent_flow(flow_run, db=db)
        new_plan_id = str(uuid4())
        if run_tasks_without_prefect:
            plan = await create_execution_plan_local(
                agent_id=agent_id,
                user_id=user_id,
                plan_id=new_plan_id,
                action=action,
                skip_db_commit=skip_db_commit,
                skip_task_cache=skip_task_cache,
                run_plan_in_prefect_immediately=run_plan_in_prefect_immediately,
                run_tasks_without_prefect=run_tasks_without_prefect,
                use_sample_plans=use_sample_plans,
            )
            if plan:
                return new_plan_id, plan, action
        await prefect_create_execution_plan(
            agent_id=agent_id,
            user_id=user_id,
            plan_id=new_plan_id,
            action=action,
            skip_db_commit=skip_db_commit,
            skip_task_cache=skip_task_cache,
            run_plan_in_prefect_immediately=run_plan_in_prefect_immediately,
        )

    return None


async def rewrite_execution_plan(
    agent_id: str,
    user_id: str,
    plan_id: str,
    action: Action,
    error_info: Optional[ErrorInfo] = None,
    skip_db_commit: bool = False,
    skip_task_cache: bool = False,
    run_plan_in_prefect_immediately: bool = True,
    run_tasks_without_prefect: bool = False,
    do_chat: bool = True,
    chat_context: Optional[ChatContext] = None,
    use_sample_plans: bool = True,
    db: Optional[AsyncDB] = None,
) -> Optional[ExecutionPlan]:
    logger = get_prefect_logger(__name__)
    planner = Planner(agent_id=agent_id)
    db = db or AsyncDB(pg=SyncBoostedPG(skip_commit=skip_db_commit))
    chatbot = Chatbot(agent_id=agent_id)

    logger.info(f"Starting rewrite of execution plan for {agent_id=}...")
    chat_context = chat_context or await get_chat_history_from_db(agent_id, db)

    _, old_plan, plan_timestamp, _ = await get_latest_execution_plan_from_db(agent_id, db)
    if not old_plan:  # shouldn't happen, just for mypy
        raise RuntimeError("Cannot rewrite a plan that does not exist!")

    await publish_agent_plan_status(
        agent_id=agent_id, plan_id=plan_id, status=PlanStatus.CREATING, db=db
    )

    if await check_cancelled(db=db, plan_id=plan_id):
        await publish_agent_plan_status(
            agent_id=agent_id, plan_id=plan_id, status=PlanStatus.CANCELLED, db=db
        )
        raise RuntimeError("Plan rewrite has been cancelled.")
    if error_info:
        new_plan = await planner.rewrite_plan_after_error(
            error_info, chat_context, old_plan, action=action, use_sample_plans=use_sample_plans
        )
    else:
        new_plan = await planner.rewrite_plan_after_input(
            chat_context, old_plan, plan_timestamp, action=action, use_sample_plans=use_sample_plans
        )

    if await check_cancelled(db=db, plan_id=plan_id):
        await publish_agent_plan_status(
            agent_id=agent_id, plan_id=plan_id, status=PlanStatus.CANCELLED, db=db
        )
        raise RuntimeError("Plan rewrite has been cancelled.")
    if not new_plan:
        if do_chat:
            message = await chatbot.generate_initial_plan_failed_response_suggestions(
                chat_context=await get_chat_history_from_db(agent_id, db),
            )
            await send_chat_message(
                message=Message(agent_id=agent_id, message=message, is_user_message=False), db=db
            )
        await publish_agent_plan_status(
            agent_id=agent_id, plan_id=plan_id, status=PlanStatus.FAILED, db=db
        )
        raise RuntimeError("Failed to replan!")

    logger.info(f"Finished rewriting execution plan for {agent_id=}")
    logger.info(f"New Execution Plan:\n{new_plan.get_formatted_plan()}")

    if do_chat:
        if error_info:
            message = await chatbot.generate_error_replan_postplan_response(
                chat_context=await get_chat_history_from_db(agent_id, db),
                new_plan=new_plan,
                old_plan=old_plan,
            )

        else:
            message = await chatbot.generate_input_update_replan_postplan_response(
                chat_context=await get_chat_history_from_db(agent_id, db),
                new_plan=new_plan,
                old_plan=old_plan,
            )
        await send_chat_message(
            message=Message(agent_id=agent_id, message=message, is_user_message=False), db=db
        )

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

    await publish_agent_execution_plan(new_plan, ctx, db)

    logger.info(
        f"Submitting new execution plan to Prefect for {agent_id=}, {plan_id=}, {plan_run_id=}"
    )
    if await check_cancelled(db=db, plan_id=plan_id):
        await publish_agent_plan_status(
            agent_id=agent_id, plan_id=plan_id, status=PlanStatus.CANCELLED, db=db
        )
        raise RuntimeError("Plan rewrite has been cancelled.")
    await prefect_run_execution_plan(plan=new_plan, context=ctx, do_chat=do_chat)

    return new_plan


async def handle_error_in_execution(
    context: PlanRunContext, error: Exception, step: ToolExecutionNode, do_chat: bool = True
) -> bool:
    """
    Handles an error, and returns a boolean. Returns True if the plan is being
    retried, and false if not.
    """
    db = get_psql(skip_commit=context.skip_db_commit)
    plans, plan_times, _ = db.get_all_execution_plans(context.agent_id)
    chat_context = db.get_chats_history_for_agent(context.agent_id)
    # check to see if EXECUTION_TRIES (3) plans have been written since last user message
    # if so, we should give up, otherwise could retry forever...
    last_user_message_time = max(
        message.message_time for message in chat_context.messages if message.is_user_message
    )

    if sum([last_user_message_time < plan_time for plan_time in plan_times]) >= EXECUTION_TRIES:
        action = Action.NONE
    else:
        decider = ErrorActionDecider(context.agent_id)
        action, change = await decider.decide_action(error, step, plans, chat_context)
    if action == Action.NONE:
        if do_chat:
            chatbot = Chatbot(agent_id=context.agent_id)
            message = await chatbot.generate_initial_plan_failed_response_suggestions(
                chat_context=chat_context
            )
            await send_chat_message(
                message=Message(agent_id=context.agent_id, message=message, is_user_message=False),
                db=db,
            )

        return False

    error_info = ErrorInfo(error=str(error), step=step, change=change)

    if do_chat:
        chatbot = Chatbot(agent_id=context.agent_id)
        message = await chatbot.generate_error_replan_preplan_response(
            chat_context=chat_context, last_plan=plans[-1], error_info=error_info
        )
        await send_chat_message(
            message=Message(agent_id=context.agent_id, message=message, is_user_message=False),
            db=db,
        )
    new_plan_id = str(uuid4())

    if context.run_tasks_without_prefect:
        # For testing
        # This block is only for offline tool
        plan = await create_execution_plan_local(
            agent_id=context.agent_id,
            plan_id=new_plan_id,
            user_id=context.user_id,
            action=action,
            error_info=error_info,
            skip_db_commit=True,
            skip_task_cache=True,
            run_plan_in_prefect_immediately=False,
            run_tasks_without_prefect=True,
            do_chat=do_chat,
        )
        if plan:
            new_plan_run_id = str(uuid4())
            chat_context = db.get_chats_history_for_agent(context.agent_id)
            context = PlanRunContext(
                agent_id=context.agent_id,
                plan_id=new_plan_id,
                user_id=context.user_id,
                plan_run_id=new_plan_run_id,
                skip_db_commit=context.skip_db_commit,
                skip_task_cache=context.skip_task_cache,
                run_tasks_without_prefect=context.run_tasks_without_prefect,
                chat=chat_context,
            )

            output = await run_execution_plan_local(
                plan=plan,
                context=context,
                do_chat=do_chat,
                log_all_outputs=False,
                replan_execution_error=True,
            )
            print("Got replan after error output:")
            pprint.pprint(output)
            return True
        else:
            print("replan failed")
            return False

    else:
        await prefect_create_execution_plan(
            agent_id=context.agent_id,
            user_id=context.user_id,
            plan_id=new_plan_id,
            action=action,
            error_info=error_info,
            skip_db_commit=context.skip_db_commit,
            skip_task_cache=context.skip_task_cache,
            run_plan_in_prefect_immediately=True,
        )
        return True


# Run these in tests or if you don't want to connect to the prefect server.
async def run_execution_plan_local(
    plan: ExecutionPlan,
    context: PlanRunContext,
    do_chat: bool = False,
    log_all_outputs: bool = False,
    replan_execution_error: bool = False,
    override_task_output_lookup: Optional[Dict[str, IOType]] = None,
    scheduled_by_automation: bool = False,
) -> List[IOType]:
    context.run_tasks_without_prefect = True
    return await run_execution_plan.fn(
        plan=plan,
        context=context,
        do_chat=do_chat,
        run_plan_in_prefect_immediately=False,
        log_all_outputs=log_all_outputs,
        replan_execution_error=replan_execution_error,
        override_task_output_lookup=override_task_output_lookup,
        scheduled_by_automation=scheduled_by_automation,
    )


async def create_execution_plan_local(
    agent_id: str,
    plan_id: str,
    user_id: str,
    action: Action = Action.CREATE,
    error_info: Optional[ErrorInfo] = None,
    skip_db_commit: bool = False,
    skip_task_cache: bool = False,
    run_plan_in_prefect_immediately: bool = True,
    run_tasks_without_prefect: bool = True,
    chat_context: Optional[ChatContext] = None,
    do_chat: bool = False,
    use_sample_plans: bool = True,
) -> Optional[ExecutionPlan]:
    return await create_execution_plan.fn(
        agent_id=agent_id,
        plan_id=plan_id,
        user_id=user_id,
        action=action,
        error_info=error_info,
        skip_db_commit=skip_db_commit,
        skip_task_cache=skip_task_cache,
        run_plan_in_prefect_immediately=run_plan_in_prefect_immediately,
        run_tasks_without_prefect=run_tasks_without_prefect,
        chat_context=chat_context,
        do_chat=do_chat,
        use_sample_plans=use_sample_plans,
    )
