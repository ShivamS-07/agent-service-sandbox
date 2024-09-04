import pprint
from typing import DefaultDict, Dict, List, Optional, Tuple, Union
from uuid import uuid4

from prefect import flow

from agent_service.chatbot.chatbot import Chatbot
from agent_service.endpoints.models import Status, TaskStatus
from agent_service.GPT.constants import GPT4_O, NO_PROMPT
from agent_service.GPT.requests import GPT, set_plan_run_context
from agent_service.io_type_utils import IOType, split_io_type_into_components
from agent_service.io_types.text import Text
from agent_service.planner.action_decide import (
    ErrorActionDecider,
    FirstAction,
    FirstActionDecider,
    FollowupAction,
    FollowupActionDecider,
)
from agent_service.planner.constants import (
    CHAT_DIFF_TEMPLATE,
    CREATE_EXECUTION_PLAN_FLOW_NAME,
    EXECUTION_TRIES,
    NO_CHANGE_MESSAGE,
    RUN_EXECUTION_PLAN_FLOW_NAME,
)
from agent_service.planner.errors import NonRetriableError
from agent_service.planner.planner import Planner
from agent_service.planner.planner_types import (
    ErrorInfo,
    ExecutionPlan,
    OutputWithID,
    PlanStatus,
    RunMetadata,
    ToolExecutionNode,
)
from agent_service.tool import ToolRegistry
from agent_service.types import ChatContext, Message, PlanRunContext
from agent_service.utils.agent_event_utils import (
    get_agent_task_logs,
    publish_agent_execution_plan,
    publish_agent_execution_status,
    publish_agent_output,
    publish_agent_plan_status,
    publish_agent_task_status,
    send_agent_emails,
    send_chat_message,
)
from agent_service.utils.async_db import AsyncDB, get_chat_history_from_db
from agent_service.utils.async_utils import gather_with_concurrency
from agent_service.utils.check_cancelled import AgentCancelledError
from agent_service.utils.clickhouse import Clickhouse
from agent_service.utils.do_not_error_exception import DoNotErrorException
from agent_service.utils.feature_flags import get_ld_flag, get_user_context
from agent_service.utils.gpt_logging import (
    GptJobIdType,
    GptJobType,
    create_gpt_context,
    plan_create_context,
)
from agent_service.utils.logs import async_perf_logger
from agent_service.utils.output_utils.output_diffs import (
    OutputDiffer,
    generate_full_diff_summary,
)
from agent_service.utils.output_utils.prompts import SHORT_SUMMARY_WORKLOG_MAIN_PROMPT
from agent_service.utils.output_utils.utils import io_type_to_gpt_input, output_for_log
from agent_service.utils.postgres import Postgres, SyncBoostedPG, get_psql
from agent_service.utils.prefect import (
    FlowRunType,
    get_prefect_logger,
    prefect_cancel_agent_flow,
    prefect_create_execution_plan,
    prefect_pause_current_agent_flow,
    prefect_resume_agent_flow,
    prefect_run_execution_plan,
)


async def check_cancelled(
    db: Union[Postgres, AsyncDB],
    agent_id: Optional[str] = None,
    plan_id: Optional[str] = None,
    plan_run_id: Optional[str] = None,
) -> bool:
    if not agent_id and not plan_id and not plan_run_id:
        return False

    ids = [val for val in (plan_id, plan_run_id) if val is not None]
    if isinstance(db, Postgres):
        return db.is_cancelled(ids_to_check=ids) or db.is_agent_deleted(agent_id=agent_id)
    else:
        res = await gather_with_concurrency(
            [db.is_cancelled(ids_to_check=ids), db.is_agent_deleted(agent_id=agent_id)]
        )
        return res[0] or res[1]


async def check_draft(db: Union[Postgres, AsyncDB], agent_id: Optional[str] = None) -> bool:
    if not agent_id:
        return False

    if isinstance(db, Postgres):
        return db.is_agent_draft(agent_id=agent_id)
    else:
        return await db.is_agent_draft(agent_id=agent_id)


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
    execution_log: Optional[DefaultDict[str, List[dict]]] = None,
    # This is the production ready alternative to
    # `override_task_output_lookup`.  Instead of including the entire
    # output in the dict (which could be very large), simply map task ID's to
    # "replay ID's", which uniquely identify rows in clickhouse's tool_calls
    # table. By doing this, we can avoid having to send massive objects over the
    # network (sqs, to prefect, etc.).
    override_task_output_id_lookup: Optional[Dict[str, str]] = None,
) -> List[IOType]:
    ###########################################
    # PLAN RUN SETUP
    ###########################################

    logger = get_prefect_logger(__name__)
    # Maps variables to their resolved values
    variable_lookup: Dict[str, IOType] = {}
    db = get_psql(skip_commit=context.skip_db_commit)
    async_db = AsyncDB(pg=SyncBoostedPG(skip_commit=context.skip_db_commit))
    db.insert_plan_run(
        agent_id=context.agent_id, plan_id=context.plan_id, plan_run_id=context.plan_run_id
    )
    # publish start plan run execution to FE
    await publish_agent_execution_status(
        agent_id=context.agent_id,
        plan_run_id=context.plan_run_id,
        plan_id=context.plan_id,
        status=Status.RUNNING,
        logger=logger,
        db=async_db,
    )

    if scheduled_by_automation:
        context.diff_info = {}  # this will be populated during the run

    if override_task_output_id_lookup:
        try:
            task_id_output_map = await Clickhouse().get_task_outputs_from_replay_ids(
                replay_ids=list(override_task_output_id_lookup.values())
            )
            override_task_output_lookup = task_id_output_map
        except Exception:
            logger.exception("Unable to fetch task outputs using replay ID's from clickhouse")

    final_outputs = []
    final_outputs_with_ids: List[OutputWithID] = []
    tool_output = None

    # initialize our task list for SSE purposes
    tasks: List[TaskStatus] = [
        TaskStatus(
            status=Status.NOT_STARTED,
            task_id=step.tool_task_id,
            task_name=step.description,
            has_output=False,
            logs=[],
        )
        for step in plan.nodes
    ]
    chatbot = Chatbot(agent_id=context.agent_id)

    ###########################################
    # PLAN RUN BEGINS
    ###########################################
    for i, step in enumerate(plan.nodes):
        # Check both the plan_id and plan_run_id to prevent race conditions
        if await check_cancelled(
            db=db,
            agent_id=context.agent_id,
            plan_id=context.plan_id,
            plan_run_id=context.plan_run_id,
        ):
            await publish_agent_execution_status(
                agent_id=context.agent_id,
                plan_run_id=context.plan_run_id,
                plan_id=context.plan_id,
                status=Status.CANCELLED,
                logger=logger,
                db=async_db,
            )
            raise DoNotErrorException("Execution plan has been cancelled")

        logger.info(f"Running step '{step.tool_name}' (Task ID: {step.tool_task_id})")

        tool = ToolRegistry.get_tool(step.tool_name)
        # First, resolve the variables
        resolved_args = step.resolve_arguments(variable_lookup=variable_lookup)

        # Create the context
        context.task_id = step.tool_task_id
        context.tool_name = step.tool_name

        set_plan_run_context(context, scheduled_by_automation)

        # update current task to running
        tasks[i].status = Status.RUNNING

        # publish start task execution
        await publish_agent_task_status(
            agent_id=context.agent_id,
            plan_run_id=context.plan_run_id,
            tasks=tasks,
            logger=logger,
            db=async_db,
        )

        # if the tool output already exists in the map, just use that
        if override_task_output_lookup and step.tool_task_id in override_task_output_lookup:
            logger.info(f"Step '{step.tool_name}' already in task lookup, using existing value...")
            tool_output = override_task_output_lookup[step.tool_task_id]
        else:
            # Run the tool, store its output, errors and replan
            try:
                step_args = tool.input_type(**resolved_args)
                if execution_log is not None:
                    execution_log[step.tool_name].append(resolved_args)
                tool_output = await tool.func(args=step_args, context=context)
            except NonRetriableError as nre:
                logger.exception(f"Step '{step.tool_name}' failed due to {nre}")

                # Publish task error status to FE
                tasks[i].status = Status.ERROR
                await publish_agent_task_status(
                    agent_id=context.agent_id,
                    plan_run_id=context.plan_run_id,
                    tasks=tasks,
                    logger=logger,
                    db=async_db,
                )

                response = await chatbot.generate_non_retriable_error_response(
                    chat_context=db.get_chats_history_for_agent(agent_id=context.agent_id),
                    plan=plan,
                    step=step,
                    error=nre.get_message_for_llm(),
                )
                msg = Message(agent_id=context.agent_id, message=response, is_user_message=False)
                await send_chat_message(message=msg, db=db)
                raise
            except AgentCancelledError as ace:
                await publish_agent_execution_status(
                    agent_id=context.agent_id,
                    plan_run_id=context.plan_run_id,
                    plan_id=context.plan_id,
                    status=Status.CANCELLED,
                    logger=logger,
                    db=async_db,
                )
                # NEVER replan if the plan is already cancelled.
                raise ace
            except Exception as e:
                logger.exception(f"Step '{step.tool_name}' failed due to {e}")

                # Publish task error status to FE
                tasks[i].status = Status.ERROR
                await publish_agent_task_status(
                    agent_id=context.agent_id,
                    plan_run_id=context.plan_run_id,
                    tasks=tasks,
                    logger=logger,
                    db=async_db,
                )

                if await check_cancelled(
                    db=db,
                    agent_id=context.agent_id,
                    plan_id=context.plan_id,
                    plan_run_id=context.plan_run_id,
                ):
                    await publish_agent_execution_status(
                        agent_id=context.agent_id,
                        plan_run_id=context.plan_run_id,
                        plan_id=context.plan_id,
                        status=Status.CANCELLED,
                        logger=logger,
                        db=async_db,
                    )

                    # NEVER replan if the plan is already cancelled.
                    raise Exception("Execution plan has been cancelled")

                await publish_agent_execution_status(
                    agent_id=context.agent_id,
                    plan_run_id=context.plan_run_id,
                    plan_id=context.plan_id,
                    status=Status.ERROR,
                    logger=logger,
                    db=async_db,
                )

                retrying = False
                if replan_execution_error:
                    retrying = await handle_error_in_execution(context, e, step, do_chat)

                if retrying:
                    raise RuntimeError("Plan run attempt failed, retrying")

                raise RuntimeError("All retry attempts failed")

        split_outputs = await split_io_type_into_components(tool_output)
        # This will only be used if the step is an output node
        outputs_with_ids = [
            OutputWithID(output=output, output_id=str(uuid4())) for output in split_outputs
        ]

        if not step.is_output_node:
            if step.store_output:
                db.write_tool_split_outputs(split_outputs=split_outputs, context=context)
        else:
            # We have an output node
            live_plan_output = False
            if scheduled_by_automation or (
                override_task_output_lookup and step.tool_task_id in override_task_output_lookup
            ):
                live_plan_output = True

            # Publish the output
            await publish_agent_output(
                outputs_with_ids=outputs_with_ids,
                live_plan_output=live_plan_output,
                context=context,
                db=db,
            )

        if log_all_outputs:
            logger.info(f"Output of step '{step.tool_name}': {output_for_log(tool_output)}")

        # Store the output in the associated variable
        if step.output_variable_name:
            variable_lookup[step.output_variable_name] = tool_output

        # Update the chat context in case of new messages
        if not context.skip_db_commit and not scheduled_by_automation:
            context.chat = db.get_chats_history_for_agent(agent_id=context.agent_id)
        if step.is_output_node:
            final_outputs.extend(split_outputs)
            final_outputs_with_ids.extend(outputs_with_ids)

        current_task_logs = await get_agent_task_logs(
            agent_id=context.agent_id, task_id=context.task_id, db=db
        )
        tasks[i].logs = current_task_logs
        tasks[i].status = Status.COMPLETE
        tasks[i].has_output = step.store_output

        # publish finish task execution
        await publish_agent_task_status(
            agent_id=context.agent_id,
            plan_run_id=context.plan_run_id,
            tasks=tasks,
            logger=logger,
            db=async_db,
        )
        logger.info(f"Finished step '{step.tool_name}'")

    ###########################################
    # PLAN RUN ENDS, RUN POSTPROCESSING BEGINS
    ###########################################
    if await check_cancelled(
        db=db, agent_id=context.agent_id, plan_id=context.plan_id, plan_run_id=context.plan_run_id
    ):
        await publish_agent_execution_status(
            agent_id=context.agent_id,
            plan_run_id=context.plan_run_id,
            plan_id=context.plan_id,
            status=Status.CANCELLED,
            logger=logger,
            db=async_db,
        )
        raise Exception("Execution plan has been cancelled")

    logger.info(f"Finished running {context.agent_id=}, {context.plan_id=}, {context.plan_run_id=}")

    updated_output_ids = []
    full_diff_summary = None
    full_diff_summary_output = None
    short_diff_summary = "Summary generation in progress..."
    if not scheduled_by_automation and do_chat:
        logger.info("Generating chat message...")
        chat_context = db.get_chats_history_for_agent(agent_id=context.agent_id)

        # generate short diff summary
        gpt_context = create_gpt_context(
            GptJobType.AGENT_CHATBOT, context.agent_id, GptJobIdType.AGENT_ID
        )
        llm = GPT(gpt_context, GPT4_O)
        latest_report_list = [await io_type_to_gpt_input(output) for output in final_outputs]
        main_prompt = SHORT_SUMMARY_WORKLOG_MAIN_PROMPT.format(
            chat_context=chat_context,
            latest_report="\n".join(latest_report_list),
        )
        short_diff_summary = await llm.do_chat_w_sys_prompt(
            main_prompt=main_prompt,
            sys_prompt=NO_PROMPT,
            output_json=True,
        )

        message = await chatbot.generate_execution_complete_response(
            chat_context=chat_context,
            execution_plan=plan,
            outputs=final_outputs,
        )
        await send_chat_message(
            message=Message(agent_id=context.agent_id, message=message, is_user_message=False),
            db=db,
        )
    elif scheduled_by_automation:
        logger.info("Generating diff message...")
        custom_notifications = await async_db.get_all_agent_custom_notifications(context.agent_id)
        custom_notification_str = "\n".join((cn.notification_prompt for cn in custom_notifications))
        output_differ = OutputDiffer(
            plan=plan, context=context, custom_notifications=custom_notification_str
        )

        output_diffs = await output_differ.diff_outputs(
            latest_outputs_with_ids=final_outputs_with_ids,
            db=SyncBoostedPG(skip_commit=context.skip_db_commit),
        )
        logger.info(f"Got output diffs: {output_diffs}")
        should_notify = any([diff.should_notify for diff in output_diffs])
        updated_output_ids = [
            diff.output_id for diff in output_diffs if diff.should_notify and diff.output_id
        ]

        full_diff_summary = generate_full_diff_summary(output_diffs)
        full_diff_summary_output = (
            (await full_diff_summary.get()).val if full_diff_summary else None
        )
        if isinstance(full_diff_summary, Text):
            full_diff_summary_output = await full_diff_summary.to_rich_output(pg=async_db.pg)  # type: ignore

        if not should_notify:
            logger.info("No notification necessary")
            short_diff_summary = NO_CHANGE_MESSAGE
            await send_chat_message(
                message=Message(
                    agent_id=context.agent_id,
                    message=NO_CHANGE_MESSAGE,
                    is_user_message=False,
                    visible_to_llm=False,
                ),
                db=db,
                send_notification=False,
            )
        else:
            filtered_diff_summary = "\n".join(
                (
                    f"- {diff.title}: {diff.diff_summary_message}"
                    if diff.title
                    else f"- {diff.diff_summary_message}"
                )
                for diff in output_diffs
                if diff.should_notify
            )
            logger.info("Generating and sending notification")
            short_diff_summary = await output_differ.generate_short_diff_summary(
                filtered_diff_summary, custom_notification_str
            )
            await send_chat_message(
                message=Message(
                    agent_id=context.agent_id,
                    message=CHAT_DIFF_TEMPLATE.format(diff=short_diff_summary),
                    is_user_message=False,
                    visible_to_llm=False,
                ),
                db=db,
            )

            # Don't send email if agent is draft
            is_agent_draft = await check_draft(db=db, agent_id=context.agent_id)
            if (
                get_ld_flag(
                    flag_name="agent-email-notification",
                    default=False,
                    user_context=get_user_context(user_id=context.user_id),
                )
                and not is_agent_draft
            ):
                logger.info(
                    f"Sending Email notification for agent: {context.agent_id}, plan run: {context.plan_run_id}"
                )
                await send_agent_emails(
                    pg=async_db,
                    agent_id=context.agent_id,
                    plan_run_id=context.plan_run_id,
                    run_summary_short=short_diff_summary if short_diff_summary else "",
                    run_summary_long=filtered_diff_summary if filtered_diff_summary else "",
                )

    await async_db.set_plan_run_metadata(
        context=context,
        metadata=RunMetadata(
            run_summary_long=full_diff_summary,
            run_summary_short=short_diff_summary,
            updated_output_ids=updated_output_ids,
        ),
    )

    # publish finish plan run task execution
    await publish_agent_execution_status(
        agent_id=context.agent_id,
        plan_run_id=context.plan_run_id,
        plan_id=context.plan_id,
        status=Status.COMPLETE,
        logger=logger,
        updated_output_ids=updated_output_ids,
        run_summary_long=full_diff_summary_output,
        run_summary_short=short_diff_summary,
        db=async_db,
    )
    logger.info("Finished run!")
    return final_outputs


@flow(name=CREATE_EXECUTION_PLAN_FLOW_NAME, flow_run_name="{plan_id}")
async def create_execution_plan(
    agent_id: str,
    plan_id: str,
    user_id: str,
    action: FollowupAction = FollowupAction.CREATE,
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
    if action != FollowupAction.CREATE:
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
    context = plan_create_context(agent_id, user_id, plan_id)
    planner = Planner(
        agent_id=agent_id, context=context, skip_db_commit=skip_db_commit, send_chat=do_chat
    )

    logger.info(f"Starting creation of execution plan for {agent_id=}...")
    await publish_agent_plan_status(
        agent_id=agent_id, plan_id=plan_id, status=PlanStatus.CREATING, db=db
    )

    chat_context = chat_context or await get_chat_history_from_db(agent_id, db)
    if await check_cancelled(db=db, agent_id=agent_id, plan_id=plan_id):
        await publish_agent_plan_status(
            agent_id=agent_id, plan_id=plan_id, status=PlanStatus.CANCELLED, db=db
        )
        raise RuntimeError("Plan creation has been cancelled.")
    plan = await planner.create_initial_plan(
        chat_context=chat_context, use_sample_plans=use_sample_plans, plan_id=plan_id
    )
    if await check_cancelled(db=db, agent_id=agent_id, plan_id=plan_id):
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

    # Write complete plan to db and let FE know the plan is ready
    # FE cancellation button will show up after this point
    await publish_agent_execution_plan(plan, ctx, db)

    logger.info(f"Finished creating execution plan for {agent_id=}")
    logger.info(
        "\n"
        "\n===============================================\n"
        f"Execution Plan:\n{plan.get_formatted_plan(numbered=True)}"
        "\n===============================================\n"
        "\n"
    )

    if do_chat:
        if await check_cancelled(db=db, agent_id=agent_id, plan_id=plan_id):
            await publish_agent_plan_status(
                agent_id=agent_id, plan_id=plan_id, status=PlanStatus.CANCELLED, db=db
            )
            raise RuntimeError("Plan creation has been cancelled.")

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
    if await check_cancelled(db=db, agent_id=agent_id, plan_id=plan_id):
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


@async_perf_logger
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
) -> Optional[Tuple[str, ExecutionPlan, FollowupAction]]:
    logger = get_prefect_logger(__name__)

    db = get_psql(skip_commit=skip_db_commit)

    flow_run = await prefect_pause_current_agent_flow(agent_id=agent_id)

    chat_context = chat_context or db.get_chats_history_for_agent(agent_id=agent_id)

    chatbot = Chatbot(agent_id=agent_id)

    latest_plan_id, latest_plan, _, _, _ = db.get_latest_execution_plan(agent_id)
    # decider will be either FirstActionDecider or FollowupActionDecider depending on whether plan exists
    action: Union[FirstAction, FollowupAction]
    if latest_plan and latest_plan_id:
        followupdecider = FollowupActionDecider(agent_id=agent_id, skip_db_commit=skip_db_commit)
        action = await followupdecider.decide_action(chat_context, latest_plan)
    else:
        firstdecider = FirstActionDecider(agent_id=agent_id, skip_db_commit=skip_db_commit)
        action = await firstdecider.decide_action(chat_context)
        # for mypy
        latest_plan = ExecutionPlan(nodes=[])
        latest_plan_id = ""

    logger.info(f"Decided on action: {action} for {agent_id=}, {latest_plan_id=}")

    # For now, no layout metatool
    if action == FollowupAction.LAYOUT:
        action = FollowupAction.NONE

    if isinstance(action, FirstAction):
        if action == FirstAction.NONE:
            if do_chat:
                message = await chatbot.generate_first_response_none(chat_context)
                await send_chat_message(
                    message=Message(agent_id=agent_id, message=message, is_user_message=False),
                    db=db,
                    send_notification=False,
                )
            if flow_run:
                await prefect_resume_agent_flow(flow_run)
            return None
        elif action == FirstAction.NOTIFICATION:
            if do_chat:
                message = await chatbot.generate_first_response_notification(chat_context)
                await send_chat_message(
                    message=Message(
                        agent_id=agent_id,
                        message=message,
                        is_user_message=False,
                        visible_to_llm=False,
                    ),
                    db=db,
                    send_notification=False,
                )
            if flow_run:
                await prefect_resume_agent_flow(flow_run)
            return None
        elif action == FirstAction.REFER:
            if do_chat:
                message = await chatbot.generate_first_response_refer(chat_context)
                await send_chat_message(
                    message=Message(agent_id=agent_id, message=message, is_user_message=False),
                    db=db,
                    send_notification=False,
                )
            if flow_run:
                await prefect_resume_agent_flow(flow_run)
            return None
        elif action == FirstAction.PLAN:
            # set the action to CREATE so that we can create a plan
            action = FollowupAction.CREATE

    if isinstance(action, FollowupAction):
        if action == FollowupAction.CREATE:
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

        elif action == FollowupAction.NONE or (
            action == FollowupAction.RERUN
            and (flow_run and flow_run.flow_run_type == FlowRunType.PLAN_CREATION)
        ):
            if do_chat:
                message = await chatbot.generate_input_update_no_action_response(chat_context)
                await send_chat_message(
                    message=Message(agent_id=agent_id, message=message, is_user_message=False),
                    db=db,
                    send_notification=False,
                )
            if flow_run:
                await prefect_resume_agent_flow(flow_run)
            return None

        elif action == action.NOTIFICATION:
            if do_chat:
                message = await chatbot.generate_first_response_notification(chat_context)
                await send_chat_message(
                    message=Message(
                        agent_id=agent_id,
                        message=message,
                        is_user_message=False,
                        visible_to_llm=False,
                    ),
                    db=db,
                    send_notification=False,
                )
            if flow_run:
                await prefect_resume_agent_flow(flow_run)
            return None

        elif action == FollowupAction.RERUN:
            # In this case, we know that the flow_run_type is PLAN_EXECUTION (or there no flow_run),
            # otherwise we'd have run the above block instead.
            if do_chat:
                message = await chatbot.generate_input_update_rerun_response(
                    chat_context,
                    latest_plan,
                    str(
                        [
                            node.tool_name
                            for node in latest_plan.nodes
                            if ToolRegistry.get_tool(node.tool_name).reads_chat
                        ]
                    ),
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

            logger.info(
                f"Rerunning execution plan for {agent_id=}, {latest_plan_id=}, {plan_run_id=}"
            )
            db = get_psql(skip_commit=ctx.skip_db_commit)
            db.insert_plan_run(
                agent_id=ctx.agent_id, plan_id=ctx.plan_id, plan_run_id=ctx.plan_run_id
            )
            await prefect_run_execution_plan(plan=latest_plan, context=ctx, do_chat=do_chat)

        else:
            # This handles the cases for REPLAN, APPEND, and CREATE
            if do_chat:
                message = await chatbot.generate_input_update_replan_preplan_response(chat_context)
                await send_chat_message(
                    message=Message(agent_id=agent_id, message=message, is_user_message=False),
                    db=db,
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
                    print("plan created: ", new_plan_id, plan)
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
    action: FollowupAction,
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
    context = plan_create_context(agent_id, user_id, plan_id)
    planner = Planner(agent_id=agent_id, context=context)
    db = db or AsyncDB(pg=SyncBoostedPG(skip_commit=skip_db_commit))
    chatbot = Chatbot(agent_id=agent_id)
    override_task_output_id_lookup = None
    replan_execution_error = True
    logger.info(f"Starting rewrite of execution plan for {agent_id=}...")
    chat_context = chat_context or await get_chat_history_from_db(agent_id, db)
    automation_enabled = await db.get_agent_automation_enabled(agent_id=agent_id)
    old_plan_id, old_plan, plan_timestamp, _, _ = await db.get_latest_execution_plan(agent_id)
    if automation_enabled and action == FollowupAction.APPEND:
        pg = get_psql()
        old_plan_id, live_plan = pg.get_agent_live_execution_plan(agent_id=agent_id)
        if live_plan:
            old_plan = live_plan
        else:
            raise RuntimeError(f"No live plan found for agent {agent_id}!")
        replan_execution_error = False

    if not old_plan or not old_plan_id or not plan_timestamp:  # shouldn't happen, just for mypy
        raise RuntimeError("Cannot rewrite a plan that does not exist!")

    await publish_agent_plan_status(
        agent_id=agent_id, plan_id=plan_id, status=PlanStatus.CREATING, db=db
    )

    if await check_cancelled(db=db, agent_id=agent_id, plan_id=plan_id):
        await publish_agent_plan_status(
            agent_id=agent_id, plan_id=plan_id, status=PlanStatus.CANCELLED, db=db
        )
        raise RuntimeError("Plan rewrite has been cancelled.")
    if error_info:
        new_plan = await planner.rewrite_plan_after_error(
            error_info,
            chat_context,
            old_plan,
            action=action,
            use_sample_plans=use_sample_plans,
            plan_id=plan_id,
        )
    else:
        new_plan = await planner.rewrite_plan_after_input(
            chat_context,
            old_plan,
            plan_timestamp,
            action=action,
            use_sample_plans=use_sample_plans,
            plan_id=plan_id,
        )

        if automation_enabled and action == FollowupAction.APPEND and new_plan:
            task_ids = planner.replicate_plan_set_for_automated_run(old_plan, new_plan)
            override_task_output_id_lookup = await Clickhouse().get_task_replay_ids(
                agent_id=agent_id, task_ids=task_ids, plan_id=old_plan_id
            )

    if await check_cancelled(db=db, agent_id=agent_id, plan_id=plan_id):
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
    if await check_cancelled(db=db, agent_id=agent_id, plan_id=plan_id):
        await publish_agent_plan_status(
            agent_id=agent_id, plan_id=plan_id, status=PlanStatus.CANCELLED, db=db
        )
        raise RuntimeError("Plan rewrite has been cancelled.")

    await prefect_run_execution_plan(
        plan=new_plan,
        context=ctx,
        do_chat=do_chat,
        override_task_output_id_lookup=override_task_output_id_lookup,
        replan_execution_error=replan_execution_error,
    )

    return new_plan


async def handle_error_in_execution(
    context: PlanRunContext, error: Exception, step: ToolExecutionNode, do_chat: bool = True
) -> bool:
    """
    Handles an error, and returns a boolean. Returns True if the plan is being
    retried, and false if not.
    """
    logger = get_prefect_logger(__name__)

    db = get_psql(skip_commit=context.skip_db_commit)
    plans, plan_times, _ = db.get_all_execution_plans(context.agent_id)
    chat_context = db.get_chats_history_for_agent(context.agent_id)
    # check to see if EXECUTION_TRIES (3) plans have been written since last user message
    # if so, we should give up, otherwise could retry forever...
    last_user_message_time = max(
        message.message_time for message in chat_context.messages if message.is_user_message
    )

    if sum([last_user_message_time < plan_time for plan_time in plan_times]) >= EXECUTION_TRIES:
        logger.info("Too many retries, giving up. Set action to None")
        action = FollowupAction.NONE
    else:
        logger.info("Deciding on action after error...")
        followup_decider = ErrorActionDecider(context.agent_id)
        action, change = await followup_decider.decide_action(error, step, plans, chat_context)
        logger.info(
            f"Decided on action after error: {action} for {context.agent_id=}, {context.plan_id=}"
        )

    if action == FollowupAction.NONE:
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
        logger.info("Generating error response...")
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
            db.insert_plan_run(
                agent_id=context.agent_id, plan_id=context.plan_id, plan_run_id=context.plan_run_id
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
        logger.info("Submitting a new creation job to Prefect through SQS")
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
    execution_log: Optional[DefaultDict[str, List[dict]]] = None,
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
        execution_log=execution_log,
    )


async def create_execution_plan_local(
    agent_id: str,
    plan_id: str,
    user_id: str,
    action: FollowupAction = FollowupAction.CREATE,
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
