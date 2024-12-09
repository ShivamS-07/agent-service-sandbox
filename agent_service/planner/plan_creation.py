import asyncio
import logging
from typing import Optional, Tuple, Union
from uuid import uuid4

from pydantic import validate_call

from agent_service.chatbot.chatbot import Chatbot
from agent_service.endpoints.models import QuickThoughts
from agent_service.external.gemini_client import GeminiClient
from agent_service.planner.action_decide import (
    FirstActionDecider,
    FollowupActionDecider,
)
from agent_service.planner.constants import (
    PASS_CHECK_OUTPUT,
    FirstAction,
    FollowupAction,
)
from agent_service.planner.errors import AgentCancelledError
from agent_service.planner.planner import Planner
from agent_service.planner.planner_types import ErrorInfo, ExecutionPlan, PlanStatus
from agent_service.planner.prompts import QUICK_THOUGHTS_PROMPT
from agent_service.planner.utils import check_cancelled
from agent_service.tool import default_tool_registry
from agent_service.types import (
    ChatContext,
    Message,
    MessageMetadata,
    MessageSpecialFormatting,
    PlanRunContext,
)
from agent_service.utils.agent_event_utils import (
    publish_agent_execution_plan,
    publish_agent_plan_status,
    publish_agent_quick_thoughts,
    send_chat_message,
    update_agent_help_requested,
)
from agent_service.utils.async_db import AsyncDB, get_chat_history_from_db
from agent_service.utils.async_utils import run_async_background
from agent_service.utils.clickhouse import Clickhouse
from agent_service.utils.feature_flags import get_ld_flag, get_ld_flag_async
from agent_service.utils.gpt_logging import plan_create_context
from agent_service.utils.logs import async_perf_logger
from agent_service.utils.postgres import SyncBoostedPG, get_psql
from agent_service.utils.prefect import (
    cancel_agent_flow,
    get_prefect_logger,
    kick_off_create_execution_plan,
    kick_off_run_execution_plan,
)

plan_create_deco = validate_call

logger = logging.getLogger(__name__)


# Update logic
# First, pause current plan
# Then check to see what action we should take
# If no action, then just simple chatbot prompt (TODO)
# If rerun, then check to see if current step is before step that requires
# rerun, if so, can continue, otherwise cancel current run and restart run
# if append or replan, update the plan and rerun
@plan_create_deco  # type: ignore
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
    plan_run_id: Optional[str] = None,
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

    chat_context = chat_context or await get_chat_history_from_db(agent_id, db)

    # Generate quick thoughts in the background
    run_async_background(
        generate_quick_thoughts(
            agent_id=agent_id, user_id=user_id, db=db, chat_context=chat_context
        )
    )

    logger = get_prefect_logger(__name__)
    chatbot = Chatbot(agent_id=agent_id)
    context = plan_create_context(agent_id, user_id, plan_id)
    user_settings = await db.get_user_agent_settings(user_id=user_id)
    planner = Planner(
        agent_id=agent_id,
        user_id=user_id,
        context=context,
        skip_db_commit=skip_db_commit,
        send_chat=do_chat,
        user_settings=user_settings,
    )

    await publish_agent_plan_status(
        agent_id=agent_id, plan_id=plan_id, status=PlanStatus.CREATING, db=db
    )

    if await check_cancelled(db=db, agent_id=agent_id, plan_id=plan_id):
        await publish_agent_plan_status(
            agent_id=agent_id, plan_id=plan_id, status=PlanStatus.CANCELLED, db=db
        )
        raise AgentCancelledError("Plan creation has been cancelled.")
    plan = await planner.create_initial_plan(
        chat_context=chat_context, use_sample_plans=use_sample_plans, plan_id=plan_id
    )
    if await check_cancelled(db=db, agent_id=agent_id, plan_id=plan_id):
        await publish_agent_plan_status(
            agent_id=agent_id, plan_id=plan_id, status=PlanStatus.CANCELLED, db=db
        )
        raise AgentCancelledError("Plan creation has been cancelled.")
    if plan is None:
        if do_chat:
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
        if await get_ld_flag_async(
            flag_name="get-analyst-help", user_id=user_id, default=False, async_db=db
        ) and await get_ld_flag_async(
            flag_name="auto-request-help-woz", user_id=user_id, default=False, async_db=db
        ):
            await update_agent_help_requested(
                agent_id=agent_id,
                user_id=user_id,
                help_requested=True,
                db=db,
                send_message=Message(
                    agent_id=agent_id,
                    message=("Having trouble coming up with a plan, I've alerted a human to help."),
                    is_user_message=False,
                    visible_to_llm=False,
                    message_metadata=MessageMetadata(
                        formatting=MessageSpecialFormatting.HELP_REQUESTED
                    ),
                ),
            )
        raise RuntimeError("Failed to create execution plan!")

    if not plan_run_id:
        plan_run_id = str(uuid4())

    incomplete_plan = False

    if get_ld_flag("plan_completeness_check_enabled", default=False, user_context=user_id) and plan:
        try:  # this is all optional so keep it in a try/except
            first_check_result = await planner.plan_completeness_check(chat_context, plan)
            if PASS_CHECK_OUTPUT not in first_check_result:
                logger.warning(f"Plan failed completeness check:\n{first_check_result}")
                if do_chat:
                    chatbot = Chatbot(agent_id=agent_id)
                    message = await chatbot.generate_initial_midplan_response(
                        chat_context=await get_chat_history_from_db(agent_id, db)
                    )
                    await send_chat_message(
                        message=Message(
                            agent_id=agent_id,
                            message=message,
                            is_user_message=False,
                            plan_run_id=plan_run_id,
                        ),
                        db=db,
                    )
                new_plan = await planner.rewrite_plan_for_completeness(
                    chat_context, plan, first_check_result, plan_id
                )
                if new_plan:
                    second_check_result = await planner.plan_completeness_check(
                        chat_context, new_plan
                    )
                    success = PASS_CHECK_OUTPUT in second_check_result
                else:
                    logger.info("Plan completeness replan failed to parse")
                    success = False
                    second_check_result = None
                if success and new_plan:
                    logger.info("Plan completeness replan succeeded")
                    plan = new_plan
                else:
                    if new_plan and second_check_result:
                        logger.info(
                            "Plan completeness replan failed to address all incompleteness:\n"
                            + second_check_result
                        )
                        plan = new_plan
                        final_missing = second_check_result
                    else:
                        final_missing = first_check_result
                    if do_chat and plan:
                        message = await chatbot.generate_initial_postplan_incomplete_response(
                            chat_context=await get_chat_history_from_db(agent_id, db),
                            execution_plan=plan,
                            missing_str=final_missing,
                        )
                        await send_chat_message(
                            message=Message(
                                agent_id=agent_id,
                                message=message,
                                is_user_message=False,
                                plan_run_id=plan_run_id,
                            ),
                            db=db,
                        )
        except Exception as e:
            logger.warning(f"Pass for completeness failed: {e}")

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

    if await check_cancelled(db=db, agent_id=agent_id, plan_id=plan_id):
        await publish_agent_plan_status(
            agent_id=agent_id, plan_id=plan_id, status=PlanStatus.CANCELLED, db=db
        )
        raise AgentCancelledError("Plan creation has been cancelled.")

    if do_chat and not incomplete_plan:
        logger.info("Generating initial postplan response...")
        chatbot = Chatbot(agent_id=agent_id)
        message = await chatbot.generate_initial_postplan_response(
            chat_context=await get_chat_history_from_db(agent_id, db), execution_plan=plan
        )
        await send_chat_message(
            message=Message(
                agent_id=agent_id, message=message, is_user_message=False, plan_run_id=plan_run_id
            ),
            db=db,
        )

    if not run_plan_in_prefect_immediately:
        return plan

    # Check one more time for cancellation
    if await check_cancelled(db=db, agent_id=agent_id, plan_id=plan_id):
        await publish_agent_plan_status(
            agent_id=agent_id, plan_id=plan_id, status=PlanStatus.CANCELLED, db=db
        )
        raise AgentCancelledError("Plan creation has been cancelled.")

    logger.info(f"Submitting execution plan to Prefect for {agent_id=}, {plan_id=}, {plan_run_id=}")
    # Replan on error only because it's the first run
    await kick_off_run_execution_plan(
        plan=plan, context=ctx, do_chat=do_chat, replan_execution_error=True
    )
    return plan


# Update logic
# First, pause current plan
# Then check to see what action we should take
# If no action, then just simple chatbot prompt (TODO)
# If rerun, then check to see if current step is before step that requires
# rerun, if so, can continue, otherwise cancel current run and restart run
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
    async_db: Optional[AsyncDB] = None,
) -> Optional[Tuple[str, ExecutionPlan, FollowupAction]]:
    logger = get_prefect_logger(__name__)

    db = get_psql(skip_commit=skip_db_commit)

    # Using `asyncio.to_thread` to run a blocking synchronous function in a separate thread so
    # it won't block the main event loop (FastAPI server)
    logger.info(
        f"Pausing current agent flow for {agent_id=}, getting running plan run from DB "
        "and getting latest execution plan from DB..."
    )
    results = await asyncio.gather(
        asyncio.to_thread(db.get_running_plan_run, agent_id=agent_id),
        asyncio.to_thread(db.get_latest_execution_plan, agent_id=agent_id),
    )
    if not async_db:
        async_db = AsyncDB(SyncBoostedPG(skip_commit=skip_db_commit))

    plan_run_db = results[0]
    latest_plan_id, latest_plan, _, plan_status, _ = results[1]

    if plan_run_db:
        running_plan_id = plan_run_db["plan_id"]
        running_plan_run_id = plan_run_db["plan_run_id"]
    else:
        running_plan_run_id, running_plan_id = None, None

    if not chat_context:
        logger.info(f"Getting chat context for {agent_id=}")
        chat_context = await asyncio.to_thread(db.get_chats_history_for_agent, agent_id=agent_id)

    # decider will be either FirstActionDecider or FollowupActionDecider depending on whether plan exists
    logger.info(f"Deciding on action for {agent_id=}, {latest_plan_id=}")
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
    elif action == FirstAction.PLAN:
        # set the action to CREATE so that we can create a plan
        action = FollowupAction.CREATE

    chatbot = Chatbot(agent_id=agent_id)
    if isinstance(action, FirstAction):
        if action == FirstAction.NONE:
            if do_chat:
                message = await chatbot.generate_first_response_none(chat_context)
                await send_chat_message(
                    message=Message(
                        agent_id=agent_id,
                        message=message,
                        is_user_message=False,
                        plan_run_id=running_plan_run_id,
                    ),
                    db=db,
                    send_notification=False,
                )
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
                        plan_run_id=running_plan_run_id,
                    ),
                    db=db,
                    send_notification=False,
                )
            return None
        elif action == FirstAction.REFER:
            if do_chat:
                message = await chatbot.generate_first_response_refer(chat_context)
                await send_chat_message(
                    message=Message(
                        agent_id=agent_id,
                        message=message,
                        is_user_message=False,
                        plan_run_id=running_plan_run_id,
                    ),
                    db=db,
                    send_notification=False,
                )
            return None

    elif isinstance(action, FollowupAction):
        if action == FollowupAction.CREATE:
            await cancel_agent_flow(db, agent_id, running_plan_id, running_plan_run_id)

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

            await asyncio.gather(
                publish_agent_plan_status(
                    agent_id=agent_id, plan_id=new_plan_id, status=PlanStatus.CREATING, db=async_db
                ),
                kick_off_create_execution_plan(
                    agent_id=agent_id,
                    user_id=user_id,
                    plan_id=new_plan_id,
                    action=action,
                    skip_db_commit=skip_db_commit,
                    skip_task_cache=skip_task_cache,
                    run_plan_in_prefect_immediately=run_plan_in_prefect_immediately,
                ),
            )

        elif action == FollowupAction.NONE or (
            action == FollowupAction.RERUN and (plan_status == PlanStatus.CREATING)
        ):
            if do_chat:
                message = await chatbot.generate_input_update_no_action_response(chat_context)
                await send_chat_message(
                    message=Message(
                        agent_id=agent_id,
                        message=message,
                        is_user_message=False,
                        plan_run_id=running_plan_run_id,
                    ),
                    db=db,
                    send_notification=False,
                )

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
                        plan_run_id=running_plan_run_id,
                    ),
                    db=db,
                    send_notification=False,
                )

            return None

        elif action == FollowupAction.RERUN:
            # cancel the running agent flow first (DO NOT cancel this plan)
            await cancel_agent_flow(db, agent_id, plan_id=None, plan_run_id=running_plan_run_id)

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
                            if default_tool_registry().get_tool(node.tool_name).reads_chat
                        ]
                    ),
                )
                await send_chat_message(
                    message=Message(
                        agent_id=agent_id,
                        message=message,
                        is_user_message=False,
                        plan_run_id=running_plan_run_id,
                    ),
                    db=db,
                )

            if run_tasks_without_prefect:
                return latest_plan_id, latest_plan, action

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
            await kick_off_run_execution_plan(plan=latest_plan, context=ctx, do_chat=do_chat)

        else:
            # This handles the cases for REPLAN, APPEND, and CREATE
            await cancel_agent_flow(
                db,
                agent_id,
                running_plan_id,
                running_plan_run_id,
            )

            if do_chat:
                message = await chatbot.generate_input_update_replan_preplan_response(chat_context)
                await send_chat_message(
                    message=Message(
                        agent_id=agent_id,
                        message=message,
                        is_user_message=False,
                        plan_run_id=running_plan_run_id,
                    ),
                    db=db,
                )

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
                    logger.info(f"plan created: {new_plan_id}, {plan}")
                    return new_plan_id, plan, action
            await asyncio.gather(
                publish_agent_plan_status(
                    agent_id=agent_id, plan_id=new_plan_id, status=PlanStatus.CREATING, db=async_db
                ),
                kick_off_create_execution_plan(
                    agent_id=agent_id,
                    user_id=user_id,
                    plan_id=new_plan_id,
                    action=action,
                    skip_db_commit=skip_db_commit,
                    skip_task_cache=skip_task_cache,
                    run_plan_in_prefect_immediately=run_plan_in_prefect_immediately,
                ),
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
    db = db or AsyncDB(pg=SyncBoostedPG(skip_commit=skip_db_commit))
    user_settings = await db.get_user_agent_settings(user_id=user_id)
    planner = Planner(
        agent_id=agent_id, context=context, user_id=user_id, user_settings=user_settings
    )
    chatbot = Chatbot(agent_id=agent_id)
    override_task_output_id_lookup = None
    override_task_work_log_id_lookup = None
    logger.info(f"Starting rewrite of execution plan for {agent_id=}...")
    chat_context = chat_context or await get_chat_history_from_db(agent_id, db)
    automation_enabled = await db.get_agent_automation_enabled(agent_id=agent_id)
    old_plan_id, old_plan, plan_timestamp, _, _ = await db.get_latest_execution_plan(
        agent_id, only_finished_plans=True
    )
    if automation_enabled and action == FollowupAction.APPEND:
        pg = get_psql()
        old_plan_id, live_plan = pg.get_agent_live_execution_plan(agent_id=agent_id)
        if live_plan:
            old_plan = live_plan
        else:
            raise RuntimeError(f"No live plan found for agent {agent_id}!")

    if not old_plan or not old_plan_id or not plan_timestamp:  # shouldn't happen, just for mypy
        raise RuntimeError("Cannot rewrite a plan that does not exist!")

    if action == FollowupAction.REPLAN:
        run_async_background(
            generate_quick_thoughts(
                agent_id=agent_id, user_id=user_id, db=db, chat_context=chat_context
            )
        )

    await publish_agent_plan_status(
        agent_id=agent_id, plan_id=plan_id, status=PlanStatus.CREATING, db=db
    )

    if await check_cancelled(db=db, agent_id=agent_id, plan_id=plan_id):
        await publish_agent_plan_status(
            agent_id=agent_id, plan_id=plan_id, status=PlanStatus.CANCELLED, db=db
        )
        raise AgentCancelledError("Plan rewrite has been cancelled.")
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
        replan_with_locked_tasks = False
        if old_plan.locked_task_ids and action == FollowupAction.REPLAN:
            # When a replan is going to happen, but we have locked widgets, we
            # retain then by doing the following:
            # 1. Remove ALL outputs nodes that are NOT locked
            # 2. Switch into APPEND mode and create a new plan
            # This ensures we retain locked widgets while potentially recreating
            # non-locked ones.
            logger.info(
                (
                    "Switching from REPLAN mode to APPEND mode because of locked tasks in old plan."
                    f"{old_plan.locked_task_ids=}"
                )
            )
            replan_with_locked_tasks = True
            old_plan = old_plan.remove_non_locked_output_nodes()
            action = FollowupAction.APPEND

        new_plan = await planner.rewrite_plan_after_input(
            chat_context,
            old_plan,
            plan_timestamp,
            action=action,
            use_sample_plans=use_sample_plans,
            plan_id=plan_id,
            replan_with_locked_tasks=replan_with_locked_tasks,
        )

        if new_plan:
            task_ids = []
            if action == FollowupAction.APPEND:
                # If we're appending to an existing plan, we want to re-use the
                # outputs from the old plan so that the new results will appear
                # faster (and more efficiently).
                task_ids = planner.copy_task_ids_to_new_plan(old_plan, new_plan)
            if task_ids:
                # These maps will be used to look up prior outputs when the plan
                # runs. (Eventually will remove this clickhouse part, leaving it in
                # for redundancy.)
                override_task_output_id_lookup = await Clickhouse().get_task_replay_ids(
                    agent_id=agent_id, task_ids=task_ids, plan_id=old_plan_id
                )
                override_task_work_log_id_lookup = await db.get_task_work_log_ids(
                    agent_id=agent_id, task_ids=task_ids, plan_id=old_plan_id
                )

    if await check_cancelled(db=db, agent_id=agent_id, plan_id=plan_id):
        await publish_agent_plan_status(
            agent_id=agent_id, plan_id=plan_id, status=PlanStatus.CANCELLED, db=db
        )
        raise AgentCancelledError("Plan rewrite has been cancelled.")
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
        if await get_ld_flag_async(
            flag_name="get-analyst-help", user_id=user_id, default=False, async_db=db
        ) and await get_ld_flag_async(
            flag_name="auto-request-help-woz", user_id=user_id, default=False, async_db=db
        ):
            await update_agent_help_requested(
                agent_id=agent_id,
                user_id=user_id,
                help_requested=True,
                db=db,
                send_message=Message(
                    agent_id=agent_id,
                    message=(
                        "Having trouble coming up with a new plan, I've alerted a human to help."
                    ),
                    is_user_message=False,
                    visible_to_llm=False,
                    message_metadata=MessageMetadata(
                        formatting=MessageSpecialFormatting.HELP_REQUESTED
                    ),
                ),
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

    # Make sure any new plan maintains locking if possible
    new_plan.inherit_locked_task_ids_from(old_plan)
    await publish_agent_execution_plan(new_plan, ctx, db)

    logger.info(
        f"Submitting new execution plan to Prefect for {agent_id=}, {plan_id=}, {plan_run_id=}"
    )
    if await check_cancelled(db=db, agent_id=agent_id, plan_id=plan_id):
        await publish_agent_plan_status(
            agent_id=agent_id, plan_id=plan_id, status=PlanStatus.CANCELLED, db=db
        )
        raise AgentCancelledError("Plan rewrite has been cancelled.")

    await kick_off_run_execution_plan(
        plan=new_plan,
        context=ctx,
        do_chat=do_chat,
        override_task_output_id_lookup=override_task_output_id_lookup,
        override_task_work_log_id_lookup=override_task_work_log_id_lookup,
    )

    return new_plan


@async_perf_logger
async def generate_quick_thoughts(
    agent_id: str, user_id: str, db: AsyncDB, chat_context: Optional[ChatContext] = None
) -> None:

    if not get_ld_flag(
        flag_name="enable-quick-thoughts-generation", default=False, user_context=user_id
    ):
        return
    try:
        if not chat_context:
            chat_context = await db.get_chats_history_for_agent(agent_id=agent_id)
        # TODO handle logging and context
        message = chat_context.get_latest_user_message()

        # check if message can be handled by quickthoughts
        # run_quick_thoughts = await is_relevant_for_quick_thoughts(message) if message else False
        run_quick_thoughts = True

        if not message or not run_quick_thoughts:
            return
        gemini = GeminiClient()
        prompt = QUICK_THOUGHTS_PROMPT.format(chat=message.get_gpt_input())
        grounding_result = await gemini.query_google_grounding(query=prompt.filled_prompt)
        await publish_agent_quick_thoughts(
            agent_id=agent_id, quick_thoughts=QuickThoughts(summary=grounding_result), db=db
        )
    except Exception:
        logger.exception(f"Unable to generate quick thoughts for {agent_id=}")


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
    plan_run_id: Optional[str] = None,
) -> Optional[ExecutionPlan]:
    return await create_execution_plan(
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
        plan_run_id=plan_run_id,
    )
