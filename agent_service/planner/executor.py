import asyncio
import logging
import os
import pprint
import time
import traceback
from copy import deepcopy
from typing import Any, DefaultDict, Dict, List, Optional, Set, Tuple, Union
from uuid import uuid4

from gbi_common_py_utils.utils.environment import PROD_TAG, get_environment_tag
from pydantic import BaseModel, validate_call

from agent_service.chatbot.chatbot import Chatbot
from agent_service.endpoints.models import PlanRunTaskLog, Status, TaskStatus
from agent_service.GPT.constants import GPT4_O, NO_PROMPT
from agent_service.GPT.requests import GPT, set_plan_run_context
from agent_service.GPT.tokens import GPTTokenizer
from agent_service.io_type_utils import (
    IOType,
    load_io_type,
    safe_dump_io_type,
    split_io_type_into_components,
)
from agent_service.io_types.text import Text
from agent_service.planner.action_decide import ErrorActionDecider, FollowupAction
from agent_service.planner.constants import (
    CHAT_DIFF_TEMPLATE,
    EXECUTION_TRIES,
    NO_CHANGE_MESSAGE,
    PLAN_RUN_EMAIL_THRESHOLD_SECONDS,
)
from agent_service.planner.errors import (
    AgentCancelledError,
    AgentExecutionError,
    AgentExitEarlyError,
    AgentRetryError,
    NonRetriableError,
)
from agent_service.planner.plan_creation import create_execution_plan_local
from agent_service.planner.planner_types import (
    ErrorInfo,
    ExecutionPlan,
    OutputWithID,
    RunMetadata,
    ToolExecutionNode,
)
from agent_service.planner.utils import check_cancelled
from agent_service.slack.slack_sender import SlackSender, get_user_info_slack_string
from agent_service.tool import default_tool_registry, log_tool_call_event
from agent_service.types import (
    ChatContext,
    Message,
    MessageMetadata,
    MessageSpecialFormatting,
    PlanRunContext,
)
from agent_service.utils.agent_event_utils import (
    get_agent_task_logs,
    publish_agent_execution_status,
    publish_agent_output,
    publish_agent_task_status,
    send_chat_message,
    update_agent_help_requested,
)
from agent_service.utils.async_db import AsyncDB
from agent_service.utils.async_utils import gather_with_concurrency
from agent_service.utils.cache_utils import (
    RedisCacheBackend,
    get_redis_cache_backend_for_output,
)
from agent_service.utils.clickhouse import Clickhouse
from agent_service.utils.date_utils import get_now_utc
from agent_service.utils.email_utils import AgentEmail
from agent_service.utils.event_logging import log_event
from agent_service.utils.feature_flags import (
    agent_output_cache_enabled,
    get_ld_flag_async,
)
from agent_service.utils.gpt_logging import GptJobIdType, GptJobType, create_gpt_context
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
    get_prefect_logger,
    kick_off_create_execution_plan,
)

logger = logging.getLogger(__name__)

plan_run_deco = validate_call


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

            SlackSender(channel).send_message(message_text, int(time.time()) + 60)

    except Exception:
        log_event(
            "notifications-slack-message-error",
            event_data={
                "agent_id": agent_id,
                "error_msg": f"Unable to send slack message for agent_id={agent_id}, error: {traceback.format_exc()}",
            },
        )


async def send_slow_execution_message(
    agent_id: str,
    plan_run_id: str,
    db: Union[Postgres, AsyncDB],
) -> None:
    await asyncio.sleep(PLAN_RUN_EMAIL_THRESHOLD_SECONDS)
    await send_chat_message(
        message=Message(
            agent_id=agent_id,
            message="I'm sorry this is taking so long. I'll send an email to you when your report is ready.",
            is_user_message=False,
            visible_to_llm=False,
            plan_run_id=plan_run_id,
        ),
        db=db,
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
    use_new_executor_impl=False,
) -> Tuple[List[IOType], Optional[DefaultDict[str, List[dict]]]]:
    logger = get_prefect_logger(__name__)
    async_db = AsyncDB(pg=SyncBoostedPG(skip_commit=context.skip_db_commit))
    if await get_ld_flag_async(
        flag_name="use-new-plan-executor", default=False, user_id=context.user_id, async_db=async_db
    ):
        use_new_executor_impl = True
    try:
        if not use_new_executor_impl:
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
        else:
            logger.info(f"Using the NEW executor for {context.plan_run_id=}")
            result_to_return = await _run_execution_plan_impl_new(
                plan=plan,
                context=context,
                do_chat=do_chat,
                log_all_outputs=log_all_outputs,
                replan_execution_error=replan_execution_error,
                override_task_output_lookup=override_task_output_lookup,
                override_task_output_id_lookup=override_task_output_id_lookup,
                scheduled_by_automation=scheduled_by_automation,
            )
        return result_to_return, execution_log
    except Exception as e:
        status = Status.ERROR
        publish_result = True
        if isinstance(e, AgentExecutionError):
            status = e.result_status
            publish_result = e.publish_result_status
        if publish_result:
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
    replaying_plan_run = False
    complete_tasks_from_prior_run = set()
    plan_run_start_time = time.time()

    # Schedule message to be sent after 10 minutes
    send_plan_run_finish_email = await get_ld_flag_async(
        flag_name="send-plan-run-finish-email",
        user_id=context.user_id,
        default=False,
        async_db=async_db,
    )
    message_task = None
    if send_plan_run_finish_email and not scheduled_by_automation:
        message_task = asyncio.create_task(
            send_slow_execution_message(
                agent_id=context.agent_id,
                plan_run_id=context.plan_run_id,
                db=async_db,
            )
        )

    if existing_run and existing_run.get("status") in (
        Status.ERROR.value,
        Status.NO_RESULTS_FOUND.value,
        Status.CANCELLED.value,
    ):
        # If there's an existing errored run with the same ID, that means that
        # we want to retry from the errored task.
        replaying_plan_run = True
        logger.info(
            (
                f"{context.plan_run_id=} already exists with status {existing_run['status']},"
                " retrying run from latest non-complete step!"
            )
        )
        task_statuses = await async_db.get_task_run_statuses(plan_run_ids=[context.plan_run_id])
        for (_, task_id), status_info in task_statuses.items():
            if status_info.status == Status.COMPLETE:
                complete_tasks_from_prior_run.add(task_id)
    elif existing_run and existing_run.get("status") != Status.NOT_STARTED.value:
        # Not allowed to run with the same ID if the run wasn't
        # errored. NOT_STARTED is acceptable since the run may have been
        # inserted before it started.
        raise AgentExitEarlyError(
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

    # Load the user agent settings
    context.user_settings = await async_db.get_user_agent_settings(user_id=context.user_id)

    if agent_output_cache_enabled() and os.getenv("REDIS_HOST") and not context.skip_db_commit:
        logger.info(f"Using redis output cache. Connecting to {os.getenv('REDIS_HOST')}")
        redis_cache_backend = get_redis_cache_backend_for_output(auto_close_connection=True)
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

    replay_id_task_id_output_map = {}
    if override_task_output_id_lookup:
        try:
            replay_id_task_id_output_map = await Clickhouse().get_task_outputs_from_replay_ids(
                replay_ids=list(override_task_output_id_lookup.values())
            )
        except Exception:
            logger.exception("Unable to fetch task outputs using replay ID's from clickhouse")

        try:
            pg_task_id_output_map = await async_db.get_task_outputs_from_replay_ids(
                replay_ids=list(override_task_output_id_lookup.values())
            )
            replay_id_task_id_output_map.update(pg_task_id_output_map)
        except Exception:
            logger.exception("Unable to fetch task outputs using replay ID's from postgres")

    # Override work log values with those from clickhouse/task_run_info since
    # they are more accurate. We should only use work log values if we have no
    # other choice (no other data exists).
    override_task_output_lookup = {
        key: val for key, val in worklog_task_id_output_map.items() if val is not None
    }
    better_task_id_output_map = {
        key: val for key, val in replay_id_task_id_output_map.items() if val is not None
    }
    override_task_output_lookup.update(better_task_id_output_map)

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

    existing_output_titles = db.get_agent_output_titles(agent_id=context.agent_id)
    new_output_titles = {}
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

        if replaying_plan_run and step.tool_task_id in complete_tasks_from_prior_run:
            # If this is a replay, try to just look up any task that's already
            # had output stored
            task_run_info = await async_db.get_task_run_info(
                plan_run_id=context.plan_run_id, task_id=step.tool_task_id, tool_name=step.tool_name
            )
            if task_run_info:
                logger.info(f"{step=} , {str(task_run_info)[:1000]=}")
                _, tool_output_str, _, _ = task_run_info
                if step.output_variable_name and tool_output_str:
                    tool_output = load_io_type(tool_output_str)
                    logger.info(
                        f"{step=} , {str(tool_output_str)[:1000]=} , {repr(tool_output)[:1000]=}"
                    )
                    if tool_output:
                        variable_lookup[step.output_variable_name] = tool_output
                        logger.info(
                            f"Fully skipping step '{step.tool_name}' with id={step.tool_task_id}"
                        )
                        tasks[i].status = Status.COMPLETE
                        tasks[i].has_output = step.store_output
                        continue

        tool = default_tool_registry().get_tool(step.tool_name)
        # First, resolve the variables
        resolved_args = step.resolve_arguments(variable_lookup=variable_lookup)

        # Create the context
        context.task_id = step.tool_task_id
        context.tool_name = step.tool_name
        context.skip_task_logging = False

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

        logger.info(f"{repr(tool_output)[:1000]=}")
        # if the tool output already exists in the map, just use that
        if (
            override_task_output_lookup
            and step.tool_task_id in override_task_output_lookup
            and override_task_output_lookup[step.tool_task_id] is not None
        ):
            logger.info(f"Step '{step.tool_name}' already in task lookup, using existing value...")
            tool_output = override_task_output_lookup[step.tool_task_id]
            step_args = None
            try:
                step_args = tool.input_type(**resolved_args)
            except Exception:
                logger.exception("Failed to validate tool args on cached run")

            errmsg = "Failed to serialize the result on cached run"
            result_dump = safe_dump_io_type(override_task_output_lookup[step.tool_task_id], errmsg)

            event_data: Dict[str, Any] = {
                "tool_name": step.tool_name,
                "args": step_args.model_dump_json(serialize_as_any=True) if step_args else None,
                "context": context.model_dump_json(),
                "result": result_dump,
                "end_time_utc": get_now_utc().isoformat(),
                "start_time_utc": get_now_utc().isoformat(),
            }
            await log_tool_call_event(context=context, event_data=event_data)
        else:
            # Run the tool, store its output, errors and replan
            try:
                try:
                    step_args = tool.input_type(**resolved_args)
                except Exception as e:
                    logger.error(f"Could not create args: {tool.input_type=}")
                    logger.error(f"{resolved_args=}")
                    logger.error(f"{variable_lookup=}")
                    logger.error(f"{step=}")
                    raise e

                if execution_log is not None:
                    execution_log[step.tool_name].append(resolved_args)
                tool_output = await tool.func(args=step_args, context=context)
                logger.info(f"{step.tool_name=} {repr(tool_output)[:1000]=}")
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
                logger.info("agent cancelled")
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

                    logger.info("Execution plan has been cancelled")
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
                    logger.warning("Plan run attempt failed, retrying")
                    raise AgentRetryError("Plan run attempt failed, retrying")

                if (
                    not scheduled_by_automation
                    and await get_ld_flag_async(
                        flag_name="get-analyst-help",
                        user_id=context.user_id,
                        default=False,
                        async_db=async_db,
                    )
                    and await get_ld_flag_async(
                        flag_name="auto-request-help-woz",
                        user_id=context.user_id,
                        default=False,
                        async_db=async_db,
                    )
                ):
                    await update_agent_help_requested(
                        agent_id=context.agent_id,
                        user_id=context.user_id,
                        help_requested=True,
                        db=async_db,
                        send_message=Message(
                            agent_id=context.agent_id,
                            message=(
                                "I ran into an issue while running the plan, "
                                "I've alerted a human to help."
                            ),
                            is_user_message=False,
                            visible_to_llm=False,
                            message_metadata=MessageMetadata(
                                formatting=MessageSpecialFormatting.HELP_REQUESTED
                            ),
                        ),
                    )

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
                await async_db.write_tool_split_outputs(
                    outputs_with_ids=outputs_with_ids, context=context
                )
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

            new_output_titles.update(
                {output.output.title: output.output_id for output in outputs_with_ids}  # type: ignore
            )

        if log_all_outputs:
            logger.info(f"Output of step '{step.tool_name}': {output_for_log(tool_output)}")

        # Store the output in the associated variable
        if step.output_variable_name:
            logger.info(
                f"Storing output of {step.tool_name=} into {step.output_variable_name=}"
                f" value: {repr(tool_output)[:1000]}"
            )
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
    agent_email_handler = AgentEmail(db=async_db)

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

        # add widget chip to the chat message to link to the new widgets
        message = chatbot.generate_execution_complete_response(
            existing_output_titles=existing_output_titles,
            new_output_titles=new_output_titles,
            plan_run_id=context.plan_run_id,
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

        # send reminder email if plan run took longer than 10 minutes (600 seconds)
        if (
            time.time() - plan_run_start_time > PLAN_RUN_EMAIL_THRESHOLD_SECONDS
            and send_plan_run_finish_email
        ):
            await agent_email_handler.send_plan_run_finish_email(
                agent_id=context.agent_id,
                short_summary=short_diff_summary,
                output_titles=[output.title for output in final_outputs],  # type: ignore
            )
        else:
            # Cancel the scheduled low execution message if execution completes before 10 minutes
            if message_task and not message_task.done():
                message_task.cancel()
                try:
                    # ensuring the task is properly cancelled
                    await message_task
                except asyncio.CancelledError:
                    pass

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
                        short_diff_summary + "\n" + (whats_new_summary if whats_new_summary else "")
                    )
                )
                email_subject = await llm.do_chat_w_sys_prompt(
                    main_prompt=email_subject_prompt,
                    sys_prompt=NO_PROMPT,
                )

                await agent_email_handler.send_agent_emails(
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
    use_new_executor_impl: bool = False,
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
        use_new_executor_impl=use_new_executor_impl,
    )
    return result_to_return, new_execution_log


######################################################
# NEW STUFF HERE
######################################################
class TaskRunResult(BaseModel):
    result: IOType
    task_step_index: int
    output_variable: Optional[str]
    has_output: bool
    task_logs: List[PlanRunTaskLog] = []


class RunDiffs(BaseModel):
    short_summary: str
    long_summary: Optional[Union[str, Text]] = None
    updated_output_ids: Optional[List[str]] = None


async def _exit_if_cancelled(db: AsyncDB, context: PlanRunContext) -> None:
    if not context.agent_id and not context.plan_id and not context.plan_run_id:
        return

    ids = [val for val in (context.plan_id, context.plan_run_id) if val is not None]

    res = await gather_with_concurrency(
        [db.is_cancelled(ids_to_check=ids), db.is_agent_deleted(agent_id=context.agent_id)]
    )
    should_cancel = res[0] or res[1]
    if should_cancel:
        await publish_agent_execution_status(
            agent_id=context.agent_id,
            plan_run_id=context.plan_run_id,
            plan_id=context.plan_id,
            status=Status.CANCELLED,
            logger=logger,
            db=db,
        )
        logger.info(f"{context.plan_run_id=} has been cancelled")
        raise AgentCancelledError("Execution plan has been cancelled")


async def _schedule_email_task(
    context: PlanRunContext, async_db: AsyncDB
) -> Optional[asyncio.Task]:
    # Schedule message to be sent after 10 minutes
    message_task = None
    message_task = asyncio.create_task(
        send_slow_execution_message(
            agent_id=context.agent_id,
            plan_run_id=context.plan_run_id,
            db=async_db,
        )
    )
    return message_task


async def _handle_prior_run_replay(
    context: PlanRunContext, async_db: AsyncDB
) -> Tuple[bool, Set[str]]:
    """
    If this run is a replay of a prior run, return True as well as a set of task
    ID's that are complete in the prior run. Otherwise, return false and an
    empty set.
    """
    complete_tasks_from_prior_run = set()
    existing_run = await async_db.get_plan_run_info(plan_run_id=context.plan_run_id)
    replaying_plan_run = False
    if existing_run and existing_run.status in (
        Status.ERROR,
        Status.NO_RESULTS_FOUND,
        Status.CANCELLED,
    ):
        # If there's an existing errored run with the same ID, that means that
        # we want to retry from the errored task.
        replaying_plan_run = True
        logger.info(
            (
                f"{context.plan_run_id=} already exists with status {existing_run.status},"
                " retrying run from latest non-complete step!"
            )
        )
        task_statuses = await async_db.get_task_run_statuses(plan_run_ids=[context.plan_run_id])
        for (_, task_id), status_info in task_statuses.items():
            if status_info.status == Status.COMPLETE:
                complete_tasks_from_prior_run.add(task_id)
    elif existing_run and existing_run.status != Status.NOT_STARTED.value:
        # Not allowed to run with the same ID if the run wasn't
        # errored. NOT_STARTED is acceptable since the run may have been
        # inserted before it started.
        raise AgentExitEarlyError(
            f"Unable to retry a run that is in status={existing_run.status}!!! {context=}"
        )
    return replaying_plan_run, complete_tasks_from_prior_run


def _get_redis_cache_backend(context: PlanRunContext) -> Optional[RedisCacheBackend]:
    if agent_output_cache_enabled() and os.getenv("REDIS_HOST") and not context.skip_db_commit:
        logger.info(f"Using redis output cache. Connecting to {os.getenv('REDIS_HOST')}")
        return get_redis_cache_backend_for_output(auto_close_connection=True)
    return None


async def _get_override_task_output_lookup(
    async_db: AsyncDB,
    override_task_output_id_lookup: Optional[Dict[str, str]],
    override_task_output_lookup: Optional[Dict[str, IOType]],
) -> Optional[Dict[str, IOType]]:
    if not override_task_output_id_lookup and not override_task_output_lookup:
        return None
    replay_id_task_id_output_map = {}
    if override_task_output_id_lookup:
        try:
            replay_id_task_id_output_map = await Clickhouse().get_task_outputs_from_replay_ids(
                replay_ids=list(override_task_output_id_lookup.values())
            )
        except Exception:
            logger.exception("Unable to fetch task outputs using replay ID's from clickhouse")

        try:
            pg_task_id_output_map = await async_db.get_task_outputs_from_replay_ids(
                replay_ids=list(override_task_output_id_lookup.values())
            )
            replay_id_task_id_output_map.update(pg_task_id_output_map)
        except Exception:
            logger.exception("Unable to fetch task outputs using replay ID's from postgres")

    better_task_id_output_map = {
        key: val for key, val in replay_id_task_id_output_map.items() if val is not None
    }
    override_task_output_lookup = override_task_output_lookup or {}
    override_task_output_lookup.update(better_task_id_output_map)
    return override_task_output_lookup


async def _run_plan_step(
    step: ToolExecutionNode,
    context: PlanRunContext,
    async_db: AsyncDB,
    tasks: List[TaskStatus],
    task_step_index: int,
    tasks_lock: asyncio.Lock,
    replaying_plan_run: bool,
    complete_tasks_from_prior_run: Set[str],
    variable_lookup: Dict[str, IOType],
    override_task_output_lookup: Optional[Dict[str, IOType]],
    scheduled_by_automation: bool,
) -> TaskRunResult:
    await _exit_if_cancelled(db=async_db, context=context)
    logger.warning(
        f"Running step '{step.tool_name}' (Task ID: {step.tool_task_id})," f" {context.plan_id=}"
    )

    set_plan_run_context(context, scheduled_by_automation)

    if replaying_plan_run and step.tool_task_id in complete_tasks_from_prior_run:
        # If this is a replay, try to just look up any task that's already
        # had output stored
        task_run_info = await async_db.get_task_run_info(
            plan_run_id=context.plan_run_id, task_id=step.tool_task_id, tool_name=step.tool_name
        )
        if task_run_info:
            logger.info(f"{step=} , {str(task_run_info)[:1000]=}")
            _, tool_output_str, _, _ = task_run_info
            if step.output_variable_name and tool_output_str:
                tool_output = load_io_type(tool_output_str)
                logger.info(
                    f"{step=} , {str(tool_output_str)[:1000]=} , {repr(tool_output)[:1000]=}"
                )
                if tool_output:
                    logger.info(
                        f"Fully skipping step '{step.tool_name}' with id={step.tool_task_id}"
                    )
                    return TaskRunResult(
                        result=tool_output,
                        task_step_index=task_step_index,
                        output_variable=step.output_variable_name,
                        has_output=tool_output is not None and step.store_output,
                    )

    tool = default_tool_registry().get_tool(step.tool_name)
    # First, resolve the variables
    resolved_args = step.resolve_arguments(variable_lookup=variable_lookup)

    # Create the context
    context.task_id = step.tool_task_id
    context.tool_name = step.tool_name
    context.skip_task_logging = False

    async with tasks_lock:
        # update current task to running
        tasks[task_step_index].status = Status.RUNNING

        # publish start task execution
        await publish_agent_task_status(
            agent_id=context.agent_id,
            plan_run_id=context.plan_run_id,
            tasks=tasks,
            logger=logger,
            db=async_db,
        )

    # if the tool output already exists in the map, just use that
    task_logs = []
    if (
        override_task_output_lookup
        and step.tool_task_id in override_task_output_lookup
        and override_task_output_lookup[step.tool_task_id] is not None
    ):
        logger.info(
            f"{step.tool_task_id=}: '{step.tool_name}' already in task lookup, using existing value..."
        )
        tool_output = override_task_output_lookup[step.tool_task_id]
        step_args = None
        try:
            step_args = tool.input_type(**resolved_args)
        except Exception:
            logger.exception("Failed to validate tool args on cached run")

        errmsg = "Failed to serialize the result on cached run"
        result_dump = safe_dump_io_type(override_task_output_lookup[step.tool_task_id], errmsg)

        event_data: Dict[str, Any] = {
            "tool_name": step.tool_name,
            "args": step_args.model_dump_json(serialize_as_any=True) if step_args else None,
            "context": context.model_dump_json(),
            "result": result_dump,
            "end_time_utc": get_now_utc().isoformat(),
            "start_time_utc": get_now_utc().isoformat(),
        }
        await log_tool_call_event(context=context, event_data=event_data)

    else:
        # Run the tool, store its output, errors and replan
        step_args = tool.input_type(**resolved_args)
        tool_output = await tool.func(args=step_args, context=context)
        logger.info(f"{step.tool_name=} {repr(tool_output)[:1000]=}")
        task_logs = await async_db.get_task_logs(
            agent_id=context.agent_id, plan_run_id=context.plan_run_id, task_id=context.task_id
        )

    return TaskRunResult(
        result=tool_output,
        task_step_index=task_step_index,
        output_variable=step.output_variable_name,
        has_output=tool_output is not None and step.store_output,
        task_logs=task_logs,
    )


async def _handle_task_non_retriable_error(
    nre: NonRetriableError,
    context: PlanRunContext,
    async_db: AsyncDB,
    plan: ExecutionPlan,
    step: ToolExecutionNode,
    chatbot: Chatbot,
    tasks: List[TaskStatus],
    task_index: int,
) -> None:
    # Publish task error status to FE
    tasks[task_index].status = nre.result_status
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
        chat_context=await async_db.get_chats_history_for_agent(agent_id=context.agent_id),
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
    await send_chat_message(message=msg, db=async_db)


async def _handle_task_other_error(
    e: Exception,
    context: PlanRunContext,
    async_db: AsyncDB,
    replan_execution_error: bool,
    failed_step: ToolExecutionNode,
    do_chat: bool,
    scheduled_by_automation: bool,
    tasks: List[TaskStatus],
    task_index: int,
) -> None:
    # Publish task error status to FE
    tasks[task_index].status = Status.ERROR
    await publish_agent_task_status(
        agent_id=context.agent_id,
        plan_run_id=context.plan_run_id,
        tasks=tasks,
        logger=logger,
        db=async_db,
    )

    await _exit_if_cancelled(db=async_db, context=context)

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
        retrying = await handle_error_in_execution(context, e, failed_step, do_chat)

    if retrying:
        logger.warning("Plan run attempt failed, retrying")
        raise AgentRetryError("Plan run attempt failed, retrying")

    if (
        not scheduled_by_automation
        and await get_ld_flag_async(
            flag_name="get-analyst-help",
            user_id=context.user_id,
            default=False,
            async_db=async_db,
        )
        and await get_ld_flag_async(
            flag_name="auto-request-help-woz",
            user_id=context.user_id,
            default=False,
            async_db=async_db,
        )
    ):
        await update_agent_help_requested(
            agent_id=context.agent_id,
            user_id=context.user_id,
            help_requested=True,
            db=async_db,
            send_message=Message(
                agent_id=context.agent_id,
                message=(
                    "I ran into an issue while running the plan, " "I've alerted a human to help."
                ),
                is_user_message=False,
                visible_to_llm=False,
                message_metadata=MessageMetadata(
                    formatting=MessageSpecialFormatting.HELP_REQUESTED
                ),
            ),
        )

    logger.error("All retry attempts failed!")


async def _postprocess_non_live(
    async_db: AsyncDB,
    context: PlanRunContext,
    chatbot: Chatbot,
    llm: GPT,
    final_outputs: List[IOType],
    new_output_titles: Dict[str, str],
    agent_email_handler: AgentEmail,
    plan_run_start_time: float,
    send_email: bool,
    message_task: asyncio.Task | None,
) -> RunDiffs:
    logger.info("Generating chat message...")
    chat_context = await async_db.get_chats_history_for_agent(agent_id=context.agent_id)

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

    # add widget chip to the chat message to link to the new widgets
    existing_output_titles = await async_db.get_agent_output_titles(agent_id=context.agent_id)
    message = chatbot.generate_execution_complete_response(
        existing_output_titles=existing_output_titles,
        new_output_titles=new_output_titles,
        plan_run_id=context.plan_run_id,
    )
    await send_chat_message(
        message=Message(
            agent_id=context.agent_id,
            message=message,
            is_user_message=False,
            plan_run_id=context.plan_run_id,
        ),
        db=async_db,
    )

    # send reminder email if plan run took longer than 10 minutes (600 seconds)
    if time.time() - plan_run_start_time > PLAN_RUN_EMAIL_THRESHOLD_SECONDS and send_email:
        await agent_email_handler.send_plan_run_finish_email(
            agent_id=context.agent_id,
            short_summary=short_diff_summary,
            output_titles=[output.title for output in final_outputs],  # type: ignore
        )
    else:
        # Cancel the scheduled slow execution message if execution completes before 10 minutes
        if message_task and not message_task.done():
            message_task.cancel()
            try:
                # ensuring the task is properly cancelled
                await message_task
            except asyncio.CancelledError:
                pass

    return RunDiffs(short_summary=short_diff_summary)


async def _postprocess_live(
    async_db: AsyncDB,
    context: PlanRunContext,
    final_outputs_with_ids: List[OutputWithID],
    agent_email_handler: AgentEmail,
    plan: ExecutionPlan,
    llm: GPT,
) -> RunDiffs:
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
    full_diff_summary_output = (await full_diff_summary.get()).val if full_diff_summary else None
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
            db=async_db,
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
            db=async_db,
        )
        await send_notification_slack_message(
            pg=async_db,
            agent_id=context.agent_id,
            message=short_diff_summary,
            user_id=context.user_id,
            chat=context.chat,
        )

        # Don't send email if agent is draft
        is_agent_draft = await check_draft(db=async_db, agent_id=context.agent_id)
        if not is_agent_draft:
            logger.info(
                f"Sending Email notification for agent: {context.agent_id}, plan run: {context.plan_run_id}"
            )

            email_subject_prompt = EMAIL_SUBJECT_MAIN_PROMPT.format(
                email_content=(
                    short_diff_summary + "\n" + (whats_new_summary if whats_new_summary else "")
                )
            )
            email_subject = await llm.do_chat_w_sys_prompt(
                main_prompt=email_subject_prompt,
                sys_prompt=NO_PROMPT,
            )

            await agent_email_handler.send_agent_emails(
                agent_id=context.agent_id,
                email_subject=email_subject,
                plan_run_id=context.plan_run_id,
                run_summary_short=short_diff_summary if short_diff_summary else "",
                run_summary_long=whats_new_summary if whats_new_summary else "",
            )

    return RunDiffs(
        short_summary=short_diff_summary,
        long_summary=whats_new_summary,
        updated_output_ids=updated_output_ids,
    )


async def _run_execution_plan_impl_new(
    plan: ExecutionPlan,
    context: PlanRunContext,
    do_chat: bool = True,
    log_all_outputs: bool = False,
    replan_execution_error: bool = False,
    override_task_output_lookup: Optional[Dict[str, IOType]] = None,
    override_task_output_id_lookup: Optional[Dict[str, str]] = None,
    scheduled_by_automation: bool = False,
    send_email: bool = True,
) -> List[IOType]:
    ###########################################
    # PLAN RUN SETUP
    ###########################################
    logger = get_prefect_logger(__name__)
    logger.info(f"PLAN RUN SETUP {context.plan_run_id=}")
    # Maps variables to their resolved values
    variable_lookup: Dict[str, IOType] = {}
    async_db = AsyncDB(pg=SyncBoostedPG(skip_commit=context.skip_db_commit))
    replaying_plan_run = False
    complete_tasks_from_prior_run: Set[str] = set()
    plan_run_start_time = time.time()

    send_email = send_email and await get_ld_flag_async(
        flag_name="send-plan-run-finish-email",
        user_id=context.user_id,
        default=False,
        async_db=async_db,
    )

    message_task = None
    if send_email:
        message_task = await _schedule_email_task(context=context, async_db=async_db)

    replaying_plan_run, complete_tasks_from_prior_run = await _handle_prior_run_replay(
        context=context, async_db=async_db
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

    # Load the user agent settings
    context.user_settings = await async_db.get_user_agent_settings(user_id=context.user_id)

    # Used for updating cached outputs
    redis_cache_backend = _get_redis_cache_backend(context=context)

    if scheduled_by_automation:
        context.diff_info = {}  # this will be populated during the run

    # Resolve the outputs that are cached, so that we can skip them later.
    override_task_output_lookup = await _get_override_task_output_lookup(
        async_db=async_db,
        override_task_output_id_lookup=override_task_output_id_lookup,
        override_task_output_lookup=override_task_output_lookup,
    )

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
    tasks_lock = asyncio.Lock()

    # Maps each node to its direct child nodes
    node_dependency_map: Dict[ToolExecutionNode, Set[ToolExecutionNode]] = (
        plan.get_node_dependency_map()
    )
    node_parent_map: Dict[ToolExecutionNode, Set[ToolExecutionNode]] = plan.get_node_parent_map()
    chatbot = Chatbot(agent_id=context.agent_id)

    new_output_titles: Dict[str, str] = {}

    ###########################################
    # PLAN RUN BEGINS
    ###########################################
    for i, step in enumerate(plan.nodes):
        # Update the chat context in case of new messages
        if not context.skip_db_commit and not scheduled_by_automation:
            context.chat = await async_db.get_chats_history_for_agent(agent_id=context.agent_id)

        try:
            task_result = await _run_plan_step(
                step=step,
                context=deepcopy(context),
                async_db=async_db,
                tasks=tasks,
                task_step_index=i,
                tasks_lock=tasks_lock,
                replaying_plan_run=replaying_plan_run,
                complete_tasks_from_prior_run=complete_tasks_from_prior_run,
                variable_lookup=variable_lookup,
                override_task_output_lookup=override_task_output_lookup,
                scheduled_by_automation=scheduled_by_automation,
            )
        except NonRetriableError as nre:
            logger.exception(f"Step '{step.tool_name}' failed due to {nre}")
            await _handle_task_non_retriable_error(
                nre,
                context=context,
                async_db=async_db,
                plan=plan,
                step=step,
                chatbot=chatbot,
                tasks=tasks,
                task_index=i,
            )
            raise nre
        except AgentCancelledError as ace:
            logger.info("agent cancelled")
            await publish_agent_execution_status(
                agent_id=context.agent_id,
                plan_run_id=context.plan_run_id,
                plan_id=context.plan_id,
                status=Status.CANCELLED,
                logger=logger,
                db=async_db,
            )
            raise ace
        except Exception as e:
            logger.exception(f"Step '{step.tool_name}' failed due to {e}")
            await _handle_task_other_error(
                e,
                context=context,
                async_db=async_db,
                replan_execution_error=replan_execution_error,
                failed_step=step,
                do_chat=do_chat,
                scheduled_by_automation=scheduled_by_automation,
                tasks=tasks,
                task_index=i,
            )
            raise e

        tool_output = task_result.result
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
                await async_db.write_tool_split_outputs(
                    outputs_with_ids=outputs_with_ids, context=context
                )
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
                db=async_db,
                is_locked=step.tool_task_id in locked_task_ids,
                cache_backend=redis_cache_backend,
            )

            final_outputs.extend(split_outputs)
            final_outputs_with_ids.extend(outputs_with_ids)

            new_output_titles.update(
                {output.output.title: output.output_id for output in outputs_with_ids}  # type: ignore
            )

        if log_all_outputs:
            logger.info(f"Output of step '{step.tool_name}': {output_for_log(tool_output)}")

        # Store the output in the associated variable
        if step.output_variable_name:
            logger.info(
                f"Storing output of {step.tool_name=} into {step.output_variable_name=}"
                f" value: {repr(tool_output)[:1000]}"
            )
            variable_lookup[step.output_variable_name] = tool_output

        tasks[i].logs = task_result.task_logs
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
    await _exit_if_cancelled(db=async_db, context=context)

    logger.info(
        (
            f"Finished running {context.agent_id=}, {context.plan_id=}, {context.plan_run_id=},"
            " moving on to postprocessing"
        )
    )

    agent_email_handler = AgentEmail(db=async_db)
    gpt_context = create_gpt_context(
        GptJobType.AGENT_CHATBOT, context.agent_id, GptJobIdType.AGENT_ID
    )
    llm = GPT(gpt_context, GPT4_O)
    if scheduled_by_automation:
        run_diffs = await _postprocess_live(
            async_db=async_db,
            context=context,
            final_outputs_with_ids=final_outputs_with_ids,
            agent_email_handler=agent_email_handler,
            plan=plan,
            llm=llm,
        )
    else:
        run_diffs = await _postprocess_non_live(
            async_db=async_db,
            context=context,
            llm=llm,
            chatbot=chatbot,
            final_outputs=final_outputs,
            new_output_titles=new_output_titles,
            agent_email_handler=agent_email_handler,
            plan_run_start_time=plan_run_start_time,
            send_email=send_email,
            message_task=message_task,
        )

    await async_db.set_plan_run_metadata(
        context=context,
        metadata=RunMetadata(
            run_summary_long=run_diffs.long_summary,
            run_summary_short=run_diffs.short_summary,
            updated_output_ids=run_diffs.updated_output_ids,
        ),
    )

    full_diff_summary_output = run_diffs.long_summary
    if isinstance(run_diffs.long_summary, Text):
        full_diff_summary_output = await run_diffs.long_summary.to_rich_output(pg=async_db.pg)  # type: ignore

    # publish finish plan run task execution
    await publish_agent_execution_status(
        agent_id=context.agent_id,
        plan_run_id=context.plan_run_id,
        plan_id=context.plan_id,
        status=Status.COMPLETE,
        logger=logger,
        run_summary_long=full_diff_summary_output,  # type: ignore
        run_summary_short=run_diffs.short_summary,
        updated_output_ids=run_diffs.updated_output_ids,
        db=async_db,
    )
    logger.info(f"Finished run {context.plan_run_id=}")
    return final_outputs
