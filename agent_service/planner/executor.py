import asyncio
import os
import pprint
import time
import traceback
from typing import Any, DefaultDict, Dict, List, Optional, Set, Tuple, Union
from uuid import uuid4

from gbi_common_py_utils.utils.environment import PROD_TAG, get_environment_tag
from pydantic import validate_call

from agent_service.chatbot.chatbot import Chatbot
from agent_service.endpoints.models import Status, TaskStatus
from agent_service.GPT.constants import GPT4_O, NO_PROMPT
from agent_service.GPT.requests import GPT, set_plan_run_context
from agent_service.GPT.tokens import GPTTokenizer
from agent_service.io_type_utils import (
    IOType,
    dump_io_type,
    split_io_type_into_components,
)
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
    EXECUTION_TRIES,
    NO_CHANGE_MESSAGE,
)
from agent_service.planner.errors import (
    AgentCancelledError,
    AgentExecutionError,
    AgentRetryError,
    NonRetriableError,
)
from agent_service.planner.planner import Planner
from agent_service.planner.planner_types import (
    ErrorInfo,
    ExecutionPlan,
    OutputWithID,
    PlanStatus,
    RunMetadata,
    ToolExecutionNode,
)
from agent_service.slack.slack_sender import SlackSender, get_user_info_slack_string
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
from agent_service.utils.cache_utils import get_redis_cache_backend_for_output
from agent_service.utils.clickhouse import Clickhouse
from agent_service.utils.date_utils import get_now_utc
from agent_service.utils.event_logging import log_event
from agent_service.utils.feature_flags import agent_output_cache_enabled
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
from agent_service.utils.output_utils.prompts import (
    EMAIL_SUBJECT_MAIN_PROMPT,
    SHORT_SUMMARY_WORKLOG_MAIN_PROMPT,
)
from agent_service.utils.output_utils.utils import io_type_to_gpt_input, output_for_log
from agent_service.utils.postgres import Postgres, SyncBoostedPG, get_psql
from agent_service.utils.prefect import (
    cancel_agent_flow,
    get_prefect_logger,
    kick_off_create_execution_plan,
    kick_off_run_execution_plan,
)

plan_run_deco = validate_call
plan_create_deco = validate_call


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


async def send_notification_slack_message(
    pg: AsyncDB, agent_id: str, message: str, user_id: str, chat: Optional[ChatContext]
) -> None:
    try:
        env = get_environment_tag()
        channel = f"agent-notifications-{'prod' if env == PROD_TAG else 'dev'}"
        base_url = f"{'alfa' if env == PROD_TAG else 'agent-dev'}.boosted.ai"
        user_email, user_info_slack_string = await get_user_info_slack_string(pg, user_id)
        if env != PROD_TAG or (
            not user_email.endswith("@boosted.ai")
            and not user_email.endswith("@gradientboostedinvestments.com")
        ):
            message_text = (
                f"initial_prompt: {chat.messages[0].message if chat and chat.messages else ''}\n"
                f"difference: {message}\n"
                f"link: {base_url}/chat/{agent_id}\n"
                f"{user_info_slack_string}"
            )

            SlackSender(channel).send_message_at(message_text, int(time.time()) + 60)

    except Exception:
        log_event(
            "notifications-slack-message-error",
            event_data={
                "agent_id": agent_id,
                "error_msg": f"Unable to send slack message for agent_id={agent_id}, error: {traceback.format_exc()}",
            },
        )


@plan_run_deco  # type: ignore
async def run_execution_plan(
    plan: ExecutionPlan,
    context: PlanRunContext,
    do_chat: bool = True,
    log_all_outputs: bool = False,
    replan_execution_error: bool = False,
    run_plan_in_prefect_immediately: bool = True,
    # This is meant for testing, basically we can fill in the lookup table to
    # make sure we only run the plan starting from a certain point while passing
    # in precomputed outputs for prior tasks.
    override_task_output_lookup: Optional[Dict[str, IOType]] = None,
    scheduled_by_automation: bool = False,
    execution_log: Optional[DefaultDict[str, List[dict]]] = None,
    # Map task ID's to "replay ID's", which uniquely identify rows in
    # clickhouse's tool_calls table.
    override_task_output_id_lookup: Optional[Dict[str, str]] = None,
    # Map task ID's to log_ids, which uniquely identify rows in in the work_logs
    # table. This is a more reliable alternative to the clickhouse version
    # above, but they can be used together, with this map taking precedence over
    # the above map.
    override_task_work_log_id_lookup: Optional[Dict[str, str]] = None,
) -> Tuple[List[IOType], Optional[DefaultDict[str, List[dict]]]]:
    logger = get_prefect_logger(__name__)
    async_db = AsyncDB(pg=SyncBoostedPG(skip_commit=context.skip_db_commit))
    try:
        result_to_return = await _run_execution_plan_impl(
            plan=plan,
            context=context,
            do_chat=do_chat,
            log_all_outputs=log_all_outputs,
            replan_execution_error=replan_execution_error,
            run_plan_in_prefect_immediately=run_plan_in_prefect_immediately,
            override_task_output_lookup=override_task_output_lookup,
            override_task_output_id_lookup=override_task_output_id_lookup,
            execution_log=execution_log,
            scheduled_by_automation=scheduled_by_automation,
            override_task_work_log_id_lookup=override_task_work_log_id_lookup,
        )
        return result_to_return, execution_log
    except Exception as e:
        status = Status.ERROR
        if isinstance(e, AgentExecutionError):
            status = e.result_status
        await publish_agent_execution_status(
            agent_id=context.agent_id,
            plan_run_id=context.plan_run_id,
            plan_id=context.plan_id,
            status=status,
            logger=logger,
            db=async_db,
        )
        raise


async def _run_execution_plan_impl(
    plan: ExecutionPlan,
    context: PlanRunContext,
    do_chat: bool = True,
    log_all_outputs: bool = False,
    replan_execution_error: bool = False,
    run_plan_in_prefect_immediately: bool = True,
    override_task_output_lookup: Optional[Dict[str, IOType]] = None,
    scheduled_by_automation: bool = False,
    execution_log: Optional[DefaultDict[str, List[dict]]] = None,
    override_task_output_id_lookup: Optional[Dict[str, str]] = None,
    override_task_work_log_id_lookup: Optional[Dict[str, str]] = None,
) -> List[IOType]:
    ###########################################
    # PLAN RUN SETUP
    ###########################################
    logger = get_prefect_logger(__name__)
    logger.info(f"PLAN RUN SETUP {context.plan_run_id=}")
    # Maps variables to their resolved values
    variable_lookup: Dict[str, IOType] = {}
    db = get_psql(skip_commit=context.skip_db_commit)
    async_db = AsyncDB(pg=SyncBoostedPG(skip_commit=context.skip_db_commit))
    existing_run = db.get_plan_run(plan_run_id=context.plan_run_id)
    skip_tasks_with_existing_outputs = False

    if existing_run and existing_run.get("status") in (
        Status.ERROR.value,
        Status.NO_RESULTS_FOUND.value,
    ):
        # If there's an existing errored run with the same ID, that means that
        # we want to retry from the errored task.
        skip_tasks_with_existing_outputs = True
        logger.info(
            (
                f"{context.plan_run_id=} already exists with status {existing_run['status']},"
                " retrying run from latest non-complete step!"
            )
        )
        complete_task_ids = []
        task_statuses = await async_db.get_task_run_statuses(plan_run_ids=[context.plan_run_id])
        for (_, task_id), status_info in task_statuses.items():
            if status_info.status == Status.COMPLETE:
                complete_task_ids.append(task_id)
        override_task_work_log_id_lookup = await async_db.get_task_work_log_ids(
            agent_id=context.agent_id, task_ids=complete_task_ids, plan_id=context.plan_id
        )
    elif existing_run and existing_run.get("status") != Status.NOT_STARTED.value:
        # Not allowed to run with the same ID if the run wasn't
        # errored. NOT_STARTED is acceptable since the run may have been
        # inserted before it started.
        raise RuntimeError(
            f"Unable to retry a run that is in status={existing_run['status']}!!! {context=}"
        )
    else:
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

    if agent_output_cache_enabled() and os.getenv("REDIS_HOST") and not context.skip_db_commit:
        logger.info(f"Using redis output cache. Connecting to {os.getenv('REDIS_HOST')}")
        redis_cache_backend = get_redis_cache_backend_for_output()
    else:
        redis_cache_backend = None

    if scheduled_by_automation:
        context.diff_info = {}  # this will be populated during the run

    worklog_task_id_output_map = {}
    if override_task_work_log_id_lookup:
        try:
            worklog_task_id_output_map = await async_db.get_task_outputs_from_work_log_ids(
                log_ids=list(override_task_work_log_id_lookup.values())
            )
        except Exception:
            logger.exception("Unable to fetch task outputs using work_logs table")

    clickhouse_task_id_output_map = {}
    if override_task_output_id_lookup:
        try:
            clickhouse_task_id_output_map = await Clickhouse().get_task_outputs_from_replay_ids(
                replay_ids=list(override_task_output_id_lookup.values())
            )
        except Exception:
            logger.exception("Unable to fetch task outputs using replay ID's from clickhouse")

    # Override clickhouse values with postgres values if both are present
    override_task_output_lookup = {
        key: val for key, val in clickhouse_task_id_output_map.items() if val is not None
    }
    worklog_task_id_output_map = {
        key: val for key, val in worklog_task_id_output_map.items() if val is not None
    }
    override_task_output_lookup.update(worklog_task_id_output_map)

    locked_task_ids = set(plan.locked_task_ids)

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
    # Maps each node to its direct child nodes
    node_dependency_map: Dict[ToolExecutionNode, Set[ToolExecutionNode]] = (
        plan.get_node_dependency_map()
    )
    node_parent_map: Dict[ToolExecutionNode, Set[ToolExecutionNode]] = plan.get_node_parent_map()
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
            raise AgentCancelledError("Execution plan has been cancelled")

        logger.warning(
            f"Running step '{step.tool_name}' (Task ID: {step.tool_task_id}),"
            f" {context.plan_id=}"
        )

        if (
            skip_tasks_with_existing_outputs
            and override_task_output_lookup
            and step.tool_task_id in override_task_output_lookup
            and override_task_output_lookup[step.tool_task_id] is not None
        ):
            # If the skip_tasks_with_existing_outputs flag is set, we want to
            # FULLY skip these steps, don't even publish statuses or anything.
            tool_output = override_task_output_lookup[step.tool_task_id]
            if step.output_variable_name:
                variable_lookup[step.output_variable_name] = tool_output
            logger.info(f"Fully skipping step '{step.tool_name}' with id={step.tool_task_id}")
            tasks[i].status = Status.COMPLETE
            tasks[i].has_output = step.store_output
            continue

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
        if (
            override_task_output_lookup
            and step.tool_task_id in override_task_output_lookup
            and override_task_output_lookup[step.tool_task_id] is not None
        ):
            logger.info(f"Step '{step.tool_name}' already in task lookup, using existing value...")
            step_args = None
            try:
                step_args = tool.input_type(**resolved_args)
            except Exception:
                logger.exception("Failed to validate tool args on cached run")
            event_data: Dict[str, Any] = {
                "tool_name": step.tool_name,
                "args": step_args.model_dump_json(serialize_as_any=True) if step_args else None,
                "context": context.model_dump_json(),
                "result": dump_io_type(override_task_output_lookup[step.tool_task_id]),
                "end_time_utc": get_now_utc().isoformat(),
                "start_time_utc": get_now_utc().isoformat(),
            }
            log_event(event_name="agent-service-tool-call", event_data=event_data)
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
                tasks[i].status = nre.result_status
                await publish_agent_task_status(
                    agent_id=context.agent_id,
                    plan_run_id=context.plan_run_id,
                    tasks=tasks,
                    logger=logger,
                    db=async_db,
                )

                await publish_agent_execution_status(
                    agent_id=context.agent_id,
                    plan_run_id=context.plan_run_id,
                    plan_id=context.plan_id,
                    status=nre.result_status,
                    logger=logger,
                    db=async_db,
                )

                response = await chatbot.generate_non_retriable_error_response(
                    chat_context=db.get_chats_history_for_agent(agent_id=context.agent_id),
                    plan=plan,
                    step=step,
                    error=nre.get_message_for_llm(),
                )
                msg = Message(
                    agent_id=context.agent_id,
                    message=response,
                    is_user_message=False,
                    plan_run_id=context.plan_run_id,
                )
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
                    raise AgentCancelledError("Execution plan has been cancelled")

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
                    raise AgentRetryError("Plan run attempt failed, retrying")

                logger.error("All retry attempts failed!")
                raise e

        split_outputs = await split_io_type_into_components(tool_output)
        # This will only be used if the step is an output node
        dependent_nodes = node_dependency_map.get(step, set())
        parent_nodes = node_parent_map.get(step, set())
        outputs_with_ids = [
            OutputWithID(
                output=output,
                task_id=step.tool_task_id,
                output_id=str(uuid4()),
                dependent_task_ids=[node.tool_task_id for node in dependent_nodes],
                parent_task_ids=[node.tool_task_id for node in parent_nodes if node.is_output_node],
            )
            for output in split_outputs
        ]

        if not step.is_output_node:
            if step.store_output:
                db.write_tool_split_outputs(outputs_with_ids=outputs_with_ids, context=context)
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
                is_locked=step.tool_task_id in locked_task_ids,
                cache_backend=redis_cache_backend,
            )

            final_outputs.extend(split_outputs)
            final_outputs_with_ids.extend(outputs_with_ids)

        if log_all_outputs:
            logger.info(f"Output of step '{step.tool_name}': {output_for_log(tool_output)}")

        # Store the output in the associated variable
        if step.output_variable_name:
            variable_lookup[step.output_variable_name] = tool_output

        # Update the chat context in case of new messages
        if not context.skip_db_commit and not scheduled_by_automation:
            context.chat = db.get_chats_history_for_agent(agent_id=context.agent_id)

        current_task_logs = await get_agent_task_logs(
            agent_id=context.agent_id,
            plan_run_id=context.plan_run_id,
            task_id=context.task_id,
            db=db,
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
        raise AgentCancelledError("Execution plan has been cancelled")

    logger.info(f"Finished running {context.agent_id=}, {context.plan_id=}, {context.plan_run_id=}")

    updated_output_ids = []
    full_diff_summary = None
    full_diff_summary_output = None
    short_diff_summary = "Summary generation in progress..."

    gpt_context = create_gpt_context(
        GptJobType.AGENT_CHATBOT, context.agent_id, GptJobIdType.AGENT_ID
    )
    llm = GPT(gpt_context, GPT4_O)
    if not scheduled_by_automation and do_chat:
        logger.info("Generating chat message...")
        chat_context = db.get_chats_history_for_agent(agent_id=context.agent_id)

        # generate short diff summary
        latest_report_list = [await io_type_to_gpt_input(output) for output in final_outputs]
        latest_report = "\n".join(latest_report_list)
        chat_text = chat_context.get_gpt_input()
        latest_report = GPTTokenizer(GPT4_O).do_truncation_if_needed(
            latest_report,
            [SHORT_SUMMARY_WORKLOG_MAIN_PROMPT.template, chat_text],
        )
        main_prompt = SHORT_SUMMARY_WORKLOG_MAIN_PROMPT.format(
            chat_context=chat_text,
            latest_report=latest_report,
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
            message=Message(
                agent_id=context.agent_id,
                message=message,
                is_user_message=False,
                plan_run_id=context.plan_run_id,
            ),
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
        whats_new_summary = full_diff_summary_output
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
                    plan_run_id=context.plan_run_id,
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
                    plan_run_id=context.plan_run_id,
                ),
                db=db,
            )
            await send_notification_slack_message(
                pg=async_db,
                agent_id=context.agent_id,
                message=short_diff_summary,
                user_id=context.user_id,
                chat=context.chat,
            )

            # Don't send email if agent is draft
            is_agent_draft = await check_draft(db=db, agent_id=context.agent_id)
            if not is_agent_draft:
                logger.info(
                    f"Sending Email notification for agent: {context.agent_id}, plan run: {context.plan_run_id}"
                )

                email_subject_prompt = EMAIL_SUBJECT_MAIN_PROMPT.format(
                    email_content=(
                        short_diff_summary + "\n" + whats_new_summary if whats_new_summary else ""
                    )
                )
                email_subject = await llm.do_chat_w_sys_prompt(
                    main_prompt=email_subject_prompt,
                    sys_prompt=NO_PROMPT,
                )

                await send_agent_emails(
                    pg=async_db,
                    agent_id=context.agent_id,
                    email_subject=email_subject,
                    plan_run_id=context.plan_run_id,
                    run_summary_short=short_diff_summary if short_diff_summary else "",
                    run_summary_long=whats_new_summary if whats_new_summary else "",
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
        agent_id=agent_id,
        user_id=user_id,
        context=context,
        skip_db_commit=skip_db_commit,
        send_chat=do_chat,
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
            raise AgentCancelledError("Plan creation has been cancelled.")

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
            await kick_off_create_execution_plan(
                agent_id=agent_id,
                user_id=user_id,
                plan_id=new_plan_id,
                action=action,
                skip_db_commit=skip_db_commit,
                skip_task_cache=skip_task_cache,
                run_plan_in_prefect_immediately=run_plan_in_prefect_immediately,
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
                            if ToolRegistry.get_tool(node.tool_name).reads_chat
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
            await kick_off_create_execution_plan(
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
    planner = Planner(agent_id=agent_id, context=context, user_id=user_id)
    db = db or AsyncDB(pg=SyncBoostedPG(skip_commit=skip_db_commit))
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
                message=Message(
                    agent_id=context.agent_id,
                    message=message,
                    is_user_message=False,
                    plan_run_id=context.plan_run_id,
                ),
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
            message=Message(
                agent_id=context.agent_id,
                message=message,
                is_user_message=False,
                plan_run_id=context.plan_run_id,
            ),
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
            logger.info("Got replan after error output:")
            out_str = pprint.pformat(output)
            logger.info(out_str)
            return True
        else:
            logger.warning("replan failed")
            return False

    else:
        logger.info("Submitting a new creation job to Prefect through SQS")
        await kick_off_create_execution_plan(
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
) -> Tuple[List[IOType], Optional[Dict[str, List[Dict]]]]:
    context.run_tasks_without_prefect = True
    result_to_return, new_execution_log = await run_execution_plan(
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
    return result_to_return, new_execution_log


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
    )
