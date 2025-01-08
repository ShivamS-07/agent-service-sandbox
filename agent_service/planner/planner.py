import ast
import asyncio
import datetime
import json
import os
import traceback
from typing import Any, Dict, List, Optional, Tuple, Type, Union, get_args, get_origin
from uuid import uuid4

from agent_service.chatbot.chatbot import Chatbot
from agent_service.GPT.constants import DEFAULT_SMART_MODEL, GPT4_O, NO_PROMPT
from agent_service.GPT.requests import GPT
from agent_service.io_type_utils import IOType, PrimitiveType, check_type_is_valid
from agent_service.planner.constants import (
    ALWAYS_AVAILABLE_TOOL_CATEGORIES,
    ARGUMENT_RE,
    ASSIGNMENT_RE,
    INITIAL_PLAN_TRIES,
    MIN_SUCCESSFUL_FOR_STOP,
    PARSING_FAIL_LINE,
    PASS_CHECK_OUTPUT,
    FollowupAction,
)
from agent_service.planner.planner_types import (
    ErrorInfo,
    ExecutionPlan,
    ExecutionPlanParsingError,
    ParsedStep,
    PartialToolArgs,
    ToolExecutionNode,
    Variable,
    get_types_from_tool_args,
)
from agent_service.planner.prompts import (
    BREAKDOWN_NEED_MAIN_PROMPT,
    BREAKDOWN_NEED_SYS_PROMPT,
    COMMENTER_MAIN_PROMPT,
    COMMENTER_SYS_PROMPT,
    COMPLETENESS_CHECK_PROMPT,
    COMPLETENESS_REPLAN_MAIN_PROMPT,
    COMPLETENESS_REPLAN_SYS_PROMPT,
    ERROR_REPLAN_MAIN_PROMPT,
    ERROR_REPLAN_SYS_PROMPT,
    PARSING_ERROR_REPLAN_MAIN_PROMPT,
    PARSING_ERROR_REPLAN_SYS_PROMPT,
    PICK_BEST_PLAN_MAIN_PROMPT,
    PICK_BEST_PLAN_SYS_PROMPT,
    PLAN_EXAMPLE,
    PLAN_EXAMPLE_NO_COMMENT,
    PLAN_GUIDELINES,
    PLAN_RULES,
    PLAN_SAMPLE_TEMPLATE,
    PLANNER_MAIN_PROMPT,
    PLANNER_SYS_PROMPT,
    RULE_COMPLEMENT,
    RULE_COMPLEMENT_NO_COMMENT,
    SELECT_TOOLS_MAIN_PROMPT,
    SELECT_TOOLS_SYS_PROMPT,
    SUBPLANNER_MAIN_PROMPT,
    SUBPLANNER_SYS_PROMPT,
    USER_INPUT_APPEND_MAIN_PROMPT,
    USER_INPUT_APPEND_SYS_PROMPT,
    USER_INPUT_REPLAN_MAIN_PROMPT,
    USER_INPUT_REPLAN_SYS_PROMPT,
)
from agent_service.planner.utils import add_comments_to_plan, get_similar_sample_plans
from agent_service.tool import Tool, ToolCategory, ToolRegistry, default_tool_registry

# Make sure all tools are imported for the planner
from agent_service.tools import *  # noqa
from agent_service.types import AgentUserSettings, ChatContext, Message
from agent_service.utils.agent_event_utils import send_chat_message
from agent_service.utils.async_utils import gather_with_concurrency, gather_with_stop
from agent_service.utils.date_utils import get_now_utc
from agent_service.utils.event_logging import log_event
from agent_service.utils.feature_flags import get_ld_flag
from agent_service.utils.logs import async_perf_logger
from agent_service.utils.postgres import get_psql
from agent_service.utils.prefect import get_prefect_logger


def get_agent_id_from_chat_context(context: ChatContext) -> str:
    if context.messages:
        return context.messages[0].agent_id
    return ""


def plan_to_json(plan: ExecutionPlan) -> str:
    json_list = []
    for node in plan.nodes:
        json_list.append(node.model_dump())
    return json.dumps(json_list)


def get_arg_dict(arg_str: str) -> Dict[str, str]:
    if len(arg_str) == 0:
        return {}
    arg_indicies = [0, *[match.start() for match in ARGUMENT_RE.finditer(arg_str)], len(arg_str)]
    arg_dict = {}
    for i in range(len(arg_indicies) - 1):
        key, value = (
            arg_str[arg_indicies[i] : arg_indicies[i + 1]].strip(" ,").split("=", maxsplit=1)
        )
        arg_dict[key.strip()] = value.strip()
    return arg_dict


class Planner:
    def __init__(
        self,
        agent_id: Optional[str] = None,
        user_id: Optional[str] = None,
        context: Optional[Dict[str, str]] = None,
        tool_registry: Optional[ToolRegistry] = None,
        send_chat: bool = True,
        skip_db_commit: bool = False,
        user_settings: Optional[AgentUserSettings] = None,
        is_subplanner: bool = False,
    ) -> None:
        self.agent_id = agent_id
        self.user_id = user_id
        self.context = context
        self.smart_llm = GPT(self.context, DEFAULT_SMART_MODEL)
        self.fast_llm = GPT(self.context, GPT4_O)
        tool_registry = tool_registry or default_tool_registry()
        self.tool_registry = tool_registry
        self.user_settings = user_settings
        self.tool_string_function_only = tool_registry.get_tool_str(
            self.user_id,
            filter_input=True,
            skip_list=ALWAYS_AVAILABLE_TOOL_CATEGORIES,
            user_settings=self.user_settings,
            using_subplanner=is_subplanner,
        )
        self.full_tool_string = tool_registry.get_tool_str(
            self.user_id, user_settings=self.user_settings, using_subplanner=is_subplanner
        )
        self.send_chat = send_chat
        self.skip_db_commit = skip_db_commit
        self.db = get_psql(skip_commit=skip_db_commit)
        self.is_subplanner = is_subplanner

    @async_perf_logger
    async def create_initial_plan(
        self,
        chat_context: ChatContext,
        plan_id: Optional[str] = None,
        use_sample_plans: bool = True,
    ) -> Optional[ExecutionPlan]:
        logger = get_prefect_logger(__name__)
        if use_sample_plans:
            logger.info("Getting matching sample plans")
            sample_plans_str = await self._get_sample_plan_str(
                input=chat_context.get_gpt_input(client_only=True)
            )
        else:
            sample_plans_str = ""

        logger.info("Writing plan")
        first_round_tasks = [
            self._create_initial_plan(
                chat_context,
                plan_id=plan_id,
                llm=self.fast_llm,
                sample_plans=sample_plans_str,
                filter_tools=True,
            )
            for _ in range(INITIAL_PLAN_TRIES)
        ]
        first_round_results = await gather_with_stop(
            first_round_tasks, MIN_SUCCESSFUL_FOR_STOP, include_exceptions=True
        )

        successful_results = [
            result for result in first_round_results if not isinstance(result, Exception)
        ]

        if successful_results:
            logger.info(
                f"{len(successful_results)} of {INITIAL_PLAN_TRIES} initial plan runs succeeded"
            )
            successful_results = list(set(successful_results))  # get rid of complete duplicates
            # GPT 4O seems to have done it now need to just pick
            if len(successful_results) > 1:
                best_plan = await self._pick_best_plan(
                    chat_context, successful_results, plan_id=plan_id
                )
            else:
                best_plan = successful_results[0]

            return best_plan

        if self.send_chat:
            await self._send_delayed_planning_message(chat_context)

        logger.warning(f"All of {INITIAL_PLAN_TRIES} initial plan runs failed, trying round 2")

        second_round_tasks = [
            self._create_initial_plan(chat_context, plan_id, self.smart_llm),
            self._get_plan_from_breakdown(chat_context, plan_id=plan_id),
        ]

        if first_round_results:
            error = first_round_results[0]  # Just use first one, TODO: consider picking?
            second_round_tasks.append(
                self._rewrite_plan_after_parsing_error(
                    chat_context, error, plan_id=plan_id, filter_tools=True
                )
            )

        turbo_plan, breakdown_plan, parsing_fix_plan = await gather_with_concurrency(
            second_round_tasks
        )

        if isinstance(turbo_plan, ExecutionPlan):
            logger.info("Round 2 turbo run succeeded, using that plan")
            # if we were able to get a working version with turbo, use that
            logger.info(f"New Plan:\n{turbo_plan.get_formatted_plan()}")
            return turbo_plan

        if isinstance(parsing_fix_plan, ExecutionPlan):
            logger.info("Round 2 parsing fix succeeded, using that plan")
            # if we were able to get a working version with turbo, use that
            logger.info(f"New Plan:\n{parsing_fix_plan.get_formatted_plan()}")
            return parsing_fix_plan

        if breakdown_plan:
            logger.info("Round 2 breakdown run succeeded, using best breakdown plan")
            logger.info(f"New Plan:\n{breakdown_plan.get_formatted_plan()}")
        else:
            logger.warning("Round 2 planning failed, giving up")

        # otherwise, return the best breakdown version if any
        return breakdown_plan

    async def _get_plan_from_breakdown(
        self,
        chat_context: ChatContext,
        plan_id: Optional[str] = None,
    ) -> Optional[ExecutionPlan]:
        request_breakdown = await self._get_request_breakdown(chat_context)
        breakdown_tasks = [
            self._create_initial_plan(
                ChatContext(messages=[Message(message=subneed, is_user_message=True)]),
                plan_id,
            )
            for subneed in request_breakdown
        ]
        breakdown_results = [
            plan
            for plan in await gather_with_concurrency(breakdown_tasks)
            if isinstance(plan, ExecutionPlan)
        ]
        if breakdown_results:
            if len(breakdown_results) > 1:
                # TODO: Combine best plans instead of just picking the best one?
                best_plan = await self._pick_best_plan(
                    chat_context, breakdown_results, plan_id=plan_id
                )
            else:
                best_plan = breakdown_results[0]
        else:
            best_plan = None

        return best_plan

    @async_perf_logger
    async def _create_initial_plan(
        self,
        chat_context: ChatContext,
        plan_id: Optional[str] = None,
        llm: Optional[GPT] = None,
        sample_plans: str = "",
        filter_tools: bool = False,
    ) -> Union[ExecutionPlan, ExecutionPlanParsingError]:
        logger = get_prefect_logger(__name__)
        execution_plan_start = get_now_utc().isoformat()
        if llm is None:
            llm = self.fast_llm

        prompt = chat_context.get_gpt_input()
        plan_str = await self._query_GPT_for_initial_plan(
            chat_context.get_gpt_input(),
            llm=llm,
            sample_plans=sample_plans,
            filter_tools=filter_tools,
        )
        agent_id = get_agent_id_from_chat_context(context=chat_context)

        try:
            steps = self._parse_plan_str(plan_str)
            plan = self._validate_and_construct_plan(steps)
        except ExecutionPlanParsingError as e:
            logger.warning(
                f"Failed to parse and validate plan with original LLM output string:\n{plan_str}"
            )
            logger.warning(
                f"Failed to parse and validate plan due to exception:\n{repr(e)}\non line:\n{e.line}"
            )
            log_event(
                event_name="agent_plan_generated",
                event_data={
                    "started_at_utc": execution_plan_start,
                    "finished_at_utc": get_now_utc().isoformat(),
                    "error_message": traceback.format_exc(),
                    "plan_str": plan_str,
                    "agent_id": agent_id,
                    "model_id": llm.model,
                    "prompt": prompt,
                    "action": FollowupAction.CREATE,
                    "sample_plans": sample_plans,
                    "plan_id": plan_id,
                },
            )
            return e  # we return this now so we can use it for updates
        log_event(
            event_name="agent_plan_generated",
            event_data={
                "started_at_utc": execution_plan_start,
                "finished_at_utc": get_now_utc().isoformat(),
                "execution_plan": plan_to_json(plan=plan),
                "plan_str": plan_str,
                "agent_id": agent_id,
                "model_id": llm.model,
                "prompt": prompt,
                "action": FollowupAction.CREATE,
                "sample_plans": sample_plans,
                "plan_id": plan_id,
            },
        )
        return plan

    @async_perf_logger
    async def _pick_best_plan(
        self, chat_context: ChatContext, plans: List[ExecutionPlan], plan_id: Optional[str] = None
    ) -> ExecutionPlan:
        plan_pick_started_at = get_now_utc().isoformat()
        plans_str = "\n\n".join(
            f"Plan {n}:\n{plan.get_formatted_plan()}" for n, plan in enumerate(plans)
        )
        sys_prompt = PICK_BEST_PLAN_SYS_PROMPT.format(guidelines=PLAN_GUIDELINES)

        main_prompt = PICK_BEST_PLAN_MAIN_PROMPT.format(
            message=chat_context.get_gpt_input(), plans=plans_str
        )
        result = await self.fast_llm.do_chat_w_sys_prompt(main_prompt, sys_prompt, no_cache=True)
        plan_selection = int(result.split("\n")[-1])
        log_event(
            event_name="agent_plan_selected",
            event_data={
                "plans": [plan_to_json(plan) for plan in plans],
                "selection": plan_selection,
                "selection_str": result,
                "agent_id": get_agent_id_from_chat_context(context=chat_context),
                "started_at_utc": plan_pick_started_at,
                "finished_at_utc": get_now_utc().isoformat(),
                "plan_id": plan_id,
            },
        )
        return plans[plan_selection]

    @async_perf_logger
    async def _get_request_breakdown(self, chat_context: ChatContext) -> List[str]:
        sys_prompt = BREAKDOWN_NEED_SYS_PROMPT.format(tools=self.tool_registry.get_tool_str())

        main_prompt = BREAKDOWN_NEED_MAIN_PROMPT.format(message=chat_context.get_gpt_input())
        result = await self.fast_llm.do_chat_w_sys_prompt(main_prompt, sys_prompt, no_cache=True)
        return result.strip("\n").replace("\n\n", "\n").split("\n")

    @async_perf_logger
    async def _send_delayed_planning_message(self, chat_context: ChatContext) -> None:
        if self.agent_id:
            chatbot = Chatbot(agent_id=self.agent_id)
            message = await chatbot.generate_initial_midplan_response(chat_context=chat_context)

            await send_chat_message(
                message=Message(agent_id=self.agent_id, message=message, is_user_message=False),
                db=self.db,
            )

    @async_perf_logger
    async def rewrite_plan_after_input(
        self,
        chat_context: ChatContext,
        last_plan: ExecutionPlan,
        last_plan_timestamp: datetime.datetime,
        action: FollowupAction,
        plan_id: str,
        use_sample_plans: bool = True,
        replan_with_locked_tasks: bool = False,
    ) -> Optional[ExecutionPlan]:
        logger = get_prefect_logger(__name__)
        if use_sample_plans:
            logger.info("Getting matching sample plans")
            sample_plans_str = await self._get_sample_plan_str(
                input=chat_context.get_gpt_input(client_only=True)
            )
        else:
            sample_plans_str = ""

        latest_user_messages = [
            message
            for message in chat_context.messages
            if message.is_user_message and message.message_time > last_plan_timestamp
        ]
        logger.info(
            f"Rewriting plan after user input, action={action}, input={latest_user_messages}"
        )
        tasks = [
            self._rewrite_plan_after_input(
                chat_context,
                last_plan,
                latest_user_messages,
                sample_plans_str,
                action=action,
                plan_id=plan_id,
                replan_with_locked_tasks=replan_with_locked_tasks,
            )
            for _ in range(INITIAL_PLAN_TRIES)
        ]
        results = await gather_with_stop(tasks, stop_count=MIN_SUCCESSFUL_FOR_STOP)
        if results:
            logger.info(f"{len(results)} of {INITIAL_PLAN_TRIES} replan runs succeeded")
            results = list(set(results))  # get rid of complete duplicates and avoid best pick
            # GPT 4O seems to have done it now need to just pick
            if len(results) > 1:
                best_plan = await self._pick_best_plan(chat_context, results, plan_id=plan_id)
            else:
                best_plan = results[0]

            return best_plan

        logger.warning(f"All of {INITIAL_PLAN_TRIES} replan runs failed")

        return None

    @staticmethod
    def copy_task_ids_to_new_plan(last_plan: ExecutionPlan, new_plan: ExecutionPlan) -> List[str]:
        task_ids = []
        for i, _ in enumerate(last_plan.nodes):
            new_plan.nodes[i].tool_task_id = last_plan.nodes[i].tool_task_id
            task_ids.append(new_plan.nodes[i].tool_task_id)
        return task_ids

    @async_perf_logger
    async def _rewrite_plan_after_input(
        self,
        chat_context: ChatContext,
        last_plan: ExecutionPlan,
        latest_messages: List[Message],
        sample_plans_str: str,
        action: FollowupAction,
        plan_id: str,
        replan_with_locked_tasks: bool = False,
    ) -> Optional[ExecutionPlan]:
        # for now, just assume last step is what is new
        # TODO: multiple old plans, flexible chat cutoff
        execution_plan_start = get_now_utc().isoformat()
        agent_id = get_agent_id_from_chat_context(context=chat_context)
        logger = get_prefect_logger(__name__)
        main_chat_str = chat_context.get_gpt_input()
        new_messages = "\n".join([message.get_gpt_input() for message in latest_messages])
        append = action == FollowupAction.APPEND
        old_plan_str = last_plan.get_formatted_plan()
        new_plan_str = await self._query_GPT_for_new_plan_after_input(
            new_messages, main_chat_str, old_plan_str, sample_plans_str, action=action
        )

        if append:
            plan_str = old_plan_str + "\n" + new_plan_str
        else:
            plan_str = new_plan_str
        try:
            steps = self._parse_plan_str(plan_str)
            new_plan = self._validate_and_construct_plan(steps)
            log_event(
                event_name="agent_plan_generated",
                event_data={
                    "started_at_utc": execution_plan_start,
                    "finished_at_utc": get_now_utc().isoformat(),
                    "execution_plan": plan_to_json(plan=new_plan),
                    "plan_str": plan_str,
                    "agent_id": agent_id,
                    "model_id": self.fast_llm.model,
                    "prompt": main_chat_str,
                    "action": action,
                    "sample_plans": sample_plans_str,
                    "plan_id": plan_id,
                    "prior_locked_tasks": last_plan.locked_task_ids,
                    "replan_with_locked_tasks": replan_with_locked_tasks,
                    "old_plan_str": old_plan_str,
                },
            )
        except Exception as e:
            log_event(
                event_name="agent_plan_generated",
                event_data={
                    "started_at_utc": execution_plan_start,
                    "finished_at_utc": get_now_utc().isoformat(),
                    "error_message": traceback.format_exc(),
                    "plan_str": plan_str,
                    "agent_id": agent_id,
                    "model_id": self.fast_llm.model,
                    "prompt": main_chat_str,
                    "action": action,
                    "sample_plans": sample_plans_str,
                    "plan_id": plan_id,
                    "prior_locked_tasks": last_plan.locked_task_ids,
                    "replan_with_locked_tasks": replan_with_locked_tasks,
                    "old_plan_str": old_plan_str,
                },
            )
            logger.warning(
                f"Failed to validate replan with original LLM output string: {plan_str}"
                f"\nException: {e}"
            )
            return None

        return new_plan

    @async_perf_logger
    async def rewrite_plan_after_error(
        self,
        error_info: ErrorInfo,
        chat_context: ChatContext,
        last_plan: ExecutionPlan,
        action: FollowupAction,
        plan_id: str,
        use_sample_plans: bool = True,
    ) -> Optional[ExecutionPlan]:
        logger = get_prefect_logger(__name__)
        if use_sample_plans:
            logger.info("Getting matching sample plans")
            sample_plans_str = await self._get_sample_plan_str(
                input=chat_context.get_gpt_input(client_only=True)
            )
        else:
            sample_plans_str = ""

        logger.info(f"Rewriting plan after error, error={error_info}")
        tasks = [
            self._rewrite_plan_after_error(
                error_info,
                chat_context,
                last_plan,
                sample_plans_str,
                action=action,
                plan_id=plan_id,
            )
            for _ in range(INITIAL_PLAN_TRIES)
        ]
        results = await gather_with_stop(tasks, stop_count=MIN_SUCCESSFUL_FOR_STOP)
        if results:
            logger.info(f"{len(results)} of {INITIAL_PLAN_TRIES} replan runs succeeded")
            results = list(set(results))  # get rid of complete duplicates
            # GPT 4O seems to have done it now need to just pick
            if len(results) > 1:
                best_plan = await self._pick_best_plan(chat_context, results, plan_id=plan_id)
            else:
                best_plan = results[0]

            return best_plan

        logger.warning(f"All of {INITIAL_PLAN_TRIES} replan runs failed")

        return None

    @async_perf_logger
    async def _rewrite_plan_after_error(
        self,
        error_info: ErrorInfo,
        chat_context: ChatContext,
        last_plan: ExecutionPlan,
        sample_plans_str: str,
        action: FollowupAction,
        plan_id: str,
    ) -> Optional[ExecutionPlan]:
        execution_plan_start = get_now_utc().isoformat()
        agent_id = get_agent_id_from_chat_context(context=chat_context)
        logger = get_prefect_logger(__name__)
        old_plan_str = last_plan.get_formatted_plan()
        chat_str = chat_context.get_gpt_input()
        error_str = error_info.error
        step_str = error_info.step.get_plan_step_str()
        new_plan_str = await self._query_GPT_for_new_plan_after_error(
            error_str, step_str, chat_str, old_plan_str, sample_plans_str
        )

        try:
            steps = self._parse_plan_str(new_plan_str)
            new_plan = self._validate_and_construct_plan(steps)
            log_event(
                event_name="agent_plan_generated",
                event_data={
                    "started_at_utc": execution_plan_start,
                    "finished_at_utc": get_now_utc().isoformat(),
                    "execution_plan": plan_to_json(plan=new_plan),
                    "plan_str": new_plan_str,
                    "agent_id": agent_id,
                    "model_id": self.smart_llm.model,
                    "prompt": error_str,
                    "action": action,
                    "sample_plans": sample_plans_str,
                    "plan_id": plan_id,
                    "old_plan_str": old_plan_str,
                },
            )
        except Exception as e:
            log_event(
                event_name="agent_plan_generated",
                event_data={
                    "started_at_utc": execution_plan_start,
                    "finished_at_utc": get_now_utc().isoformat(),
                    "error_message": traceback.format_exc(),
                    "plan_str": new_plan_str,
                    "agent_id": agent_id,
                    "model_id": self.smart_llm.model,
                    "prompt": error_str,
                    "action": action,
                    "sample_plans": sample_plans_str,
                    "plan_id": plan_id,
                    "old_plan_str": old_plan_str,
                },
            )
            logger.warning(
                f"Failed to validate replan with original LLM output string: {new_plan_str}"
                f"\nException: {e}"
            )
            return None

        return new_plan

    @async_perf_logger
    async def _rewrite_plan_after_parsing_error(
        self,
        chat_context: ChatContext,
        exception: ExecutionPlanParsingError,
        plan_id: Optional[str] = None,
        filter_tools: bool = False,
    ) -> Optional[ExecutionPlan]:
        execution_plan_start = get_now_utc().isoformat()
        agent_id = get_agent_id_from_chat_context(context=chat_context)
        logger = get_prefect_logger(__name__)
        if not exception.raw_plan:
            return None  # shouldn't happen, but if we don't have a plan to fix this doesn't work
        old_plan_str = exception.raw_plan
        chat_str = chat_context.get_gpt_input()
        error_str = exception.message
        if exception.line:
            step_str = PARSING_FAIL_LINE.format(line=exception.line)
        else:
            step_str = ""
        new_plan_str = await self._query_GPT_for_new_plan_after_parsing_error(
            error_str, step_str, chat_str, old_plan_str, filter_tools=filter_tools
        )

        try:
            steps = self._parse_plan_str(new_plan_str)
            new_plan = self._validate_and_construct_plan(steps)
            log_event(
                event_name="agent_plan_generated",
                event_data={
                    "started_at_utc": execution_plan_start,
                    "finished_at_utc": get_now_utc().isoformat(),
                    "execution_plan": plan_to_json(plan=new_plan),
                    "plan_str": new_plan_str,
                    "agent_id": agent_id,
                    "model_id": self.smart_llm.model,
                    "prompt": error_str,
                    "plan_id": plan_id,
                    "old_plan_str": old_plan_str,
                },
            )
        except Exception as e:
            log_event(
                event_name="agent_plan_generated",
                event_data={
                    "started_at_utc": execution_plan_start,
                    "finished_at_utc": get_now_utc().isoformat(),
                    "error_message": traceback.format_exc(),
                    "plan_str": new_plan_str,
                    "agent_id": agent_id,
                    "model_id": self.smart_llm.model,
                    "prompt": error_str,
                    "plan_id": plan_id,
                    "old_plan_str": old_plan_str,
                },
            )
            logger.warning(
                f"Failed to validate replan with original LLM output string: {new_plan_str}"
                f"\nException: {e}"
            )
            return None

        return new_plan

    @async_perf_logger
    async def plan_completeness_check(
        self, chat_context: ChatContext, last_plan: ExecutionPlan
    ) -> str:
        return await self.fast_llm.do_chat_w_sys_prompt(
            COMPLETENESS_CHECK_PROMPT.format(
                input=chat_context.get_gpt_input(),
                plan=last_plan.get_formatted_plan(numbered=True, include_descriptions=False),
                pass_phrase=PASS_CHECK_OUTPUT,
            ),
            NO_PROMPT,
        )

    @async_perf_logger
    async def rewrite_plan_for_completeness(
        self,
        chat_context: ChatContext,
        last_plan: ExecutionPlan,
        missing_str: str,
        plan_id: str,
    ) -> Optional[ExecutionPlan]:
        logger = get_prefect_logger(__name__)

        logger.info("Rewriting plan for completness")
        first_round_tasks = [
            self._rewrite_plan_for_completeness(
                chat_context,
                missing_str,
                last_plan,
                plan_id=plan_id,
            )
            for _ in range(INITIAL_PLAN_TRIES)
        ]
        first_round_results = await gather_with_stop(first_round_tasks, MIN_SUCCESSFUL_FOR_STOP)

        if first_round_results:
            logger.info(
                f"{len(first_round_results)} of {INITIAL_PLAN_TRIES} rewrite runs succeeded"
            )
            first_round_results = list(set(first_round_results))  # get rid of complete duplicates
            # GPT 4O seems to have done it now need to just pick
            if len(first_round_results) > 1:
                best_plan = await self._pick_best_plan(
                    chat_context, first_round_results, plan_id=plan_id
                )
            else:
                best_plan = first_round_results[0]

            return best_plan
        else:
            return None

    @async_perf_logger
    async def _rewrite_plan_for_completeness(
        self,
        chat_context: ChatContext,
        missing_str: str,
        last_plan: ExecutionPlan,
        plan_id: str,
    ) -> Optional[ExecutionPlan]:
        execution_plan_start = get_now_utc().isoformat()
        agent_id = get_agent_id_from_chat_context(context=chat_context)
        logger = get_prefect_logger(__name__)
        old_plan_str = last_plan.get_formatted_plan()
        chat_str = chat_context.get_gpt_input()
        new_plan_str = await self._query_GPT_for_new_plan_for_completeness(
            chat_str, missing_str, old_plan_str
        )

        try:
            steps = self._parse_plan_str(new_plan_str)
            new_plan = self._validate_and_construct_plan(steps)
            log_event(
                event_name="agent_plan_generated",
                event_data={
                    "started_at_utc": execution_plan_start,
                    "finished_at_utc": get_now_utc().isoformat(),
                    "execution_plan": plan_to_json(plan=new_plan),
                    "plan_str": new_plan_str,
                    "agent_id": agent_id,
                    "model_id": self.smart_llm.model,
                    "missing_str": missing_str,
                    "plan_id": plan_id,
                    "old_plan_str": old_plan_str,
                },
            )
        except Exception as e:
            log_event(
                event_name="agent_plan_generated",
                event_data={
                    "started_at_utc": execution_plan_start,
                    "finished_at_utc": get_now_utc().isoformat(),
                    "error_message": traceback.format_exc(),
                    "plan_str": new_plan_str,
                    "agent_id": agent_id,
                    "model_id": self.smart_llm.model,
                    "missing_str": missing_str,
                    "plan_id": plan_id,
                    "old_plan_str": old_plan_str,
                },
            )
            logger.warning(
                f"Failed to validate replan with original LLM output string: {new_plan_str}"
                f"\nException: {e}"
            )
            return None

        return new_plan

    @async_perf_logger
    async def create_subplan(
        self,
        directions: str,
        variables: Dict[str, Type[IOType]],
        use_sample_plans: bool = True,
    ) -> Optional[ExecutionPlan]:
        logger = get_prefect_logger(__name__)
        if use_sample_plans:
            logger.info("Getting matching sample plans")
            sample_plans_str = await self._get_sample_plan_str(input=directions)
        else:
            sample_plans_str = ""

        logger.info("Writing subplan")
        plan = await self._create_subplan(
            directions,
            variables,
            llm=self.fast_llm,
            sample_plans_str=sample_plans_str,
            filter_tools=True,
        )

        return plan

    @async_perf_logger
    async def _create_subplan(
        self,
        directions: str,
        variables: Dict[str, Type[IOType]],
        llm: Optional[GPT] = None,
        sample_plans_str: str = "",
        filter_tools: bool = False,
    ) -> Optional[ExecutionPlan]:
        if llm is None:
            llm = self.fast_llm

        plan_str = await self._query_GPT_for_new_subplan(
            directions,
            variables,
            sample_plans_str=sample_plans_str,
            filter_tools=filter_tools,
        )

        # for now, just let the tool fail if the parse fails
        steps = self._parse_plan_str(plan_str)
        plan = self._validate_and_construct_plan(steps, variable_lookup=variables)
        return plan

    async def _query_GPT_for_initial_plan(
        self, user_input: str, llm: GPT, sample_plans: str = "", filter_tools: bool = False
    ) -> str:
        logger = get_prefect_logger(__name__)

        if filter_tools and get_ld_flag(
            "planner-tool-filtering-enabled", default=False, user_context=self.user_id
        ):
            tool_str = await self.get_filtered_tool_str(user_input, sample_plans)
        else:
            tool_str = self.full_tool_string

        # fast planner
        if get_ld_flag("fast-planner-enabled", default=False, user_context=self.user_id):
            logger.info("Using fast planner")
            try:
                # get the python script
                sys_prompt = PLANNER_SYS_PROMPT.format(
                    guidelines=PLAN_GUIDELINES,
                    example=PLAN_EXAMPLE_NO_COMMENT,
                )
                main_prompt = PLANNER_MAIN_PROMPT.format(
                    message=user_input,
                    sample_plans=sample_plans,
                    rules=PLAN_RULES.format(
                        rule_complement=RULE_COMPLEMENT_NO_COMMENT,
                    ),
                    tools=tool_str,
                )
                script = await llm.do_chat_w_sys_prompt(main_prompt, sys_prompt, no_cache=True)
                plan_str = await add_comments_to_plan(
                    script=script,
                    commenter_main_prompt=COMMENTER_MAIN_PROMPT,
                    commenter_sys_prompt=COMMENTER_SYS_PROMPT,
                    sample_plans=sample_plans,
                    llm=self.fast_llm,
                )
                return plan_str
            except Exception as e:
                logger.warning(f"Failed to get initial plan with fast planner: {e}")

        # normal approach
        sys_prompt = PLANNER_SYS_PROMPT.format(
            guidelines=PLAN_GUIDELINES,
            example=PLAN_EXAMPLE,
        )
        main_prompt = PLANNER_MAIN_PROMPT.format(
            message=user_input,
            sample_plans=sample_plans,
            rules=PLAN_RULES.format(
                rule_complement=RULE_COMPLEMENT,
            ),
            tools=tool_str,
        )
        # save main prompt as txt
        plan_str = await llm.do_chat_w_sys_prompt(main_prompt, sys_prompt, no_cache=True)
        return plan_str

    async def _query_GPT_for_new_plan_after_input(
        self,
        new_user_input: str,
        existing_context: str,
        old_plan: str,
        sample_plans_str: str,
        action: FollowupAction,
    ) -> str:
        if action == FollowupAction.APPEND:
            sys_prompt_template = USER_INPUT_APPEND_SYS_PROMPT
            main_prompt_template = USER_INPUT_APPEND_MAIN_PROMPT
        else:
            sys_prompt_template = USER_INPUT_REPLAN_SYS_PROMPT
            main_prompt_template = USER_INPUT_REPLAN_MAIN_PROMPT

        sys_prompt = sys_prompt_template.format(
            rules=PLAN_RULES.format(
                rule_complement=RULE_COMPLEMENT,
            ),
            guidelines=PLAN_GUIDELINES,
            example=PLAN_EXAMPLE,
            tools=self.full_tool_string,
        )

        main_prompt = main_prompt_template.format(
            new_message=new_user_input,
            chat_context=existing_context,
            old_plan=old_plan,
            sample_plans=sample_plans_str,
        )
        plan_str = await self.smart_llm.do_chat_w_sys_prompt(main_prompt, sys_prompt, no_cache=True)
        return plan_str

    async def _query_GPT_for_new_plan_after_error(
        self, error: str, failed_step: str, chat_context: str, old_plan: str, sample_plans_str: str
    ) -> str:
        sys_prompt = ERROR_REPLAN_SYS_PROMPT.format(
            rules=PLAN_RULES.format(
                rule_complement=RULE_COMPLEMENT,
            ),
            guidelines=PLAN_GUIDELINES,
            example=PLAN_EXAMPLE,
            tools=self.full_tool_string,
        )

        main_prompt = ERROR_REPLAN_MAIN_PROMPT.format(
            error=error,
            failed_step=failed_step,
            chat_context=chat_context,
            old_plan=old_plan,
            sample_plans=sample_plans_str,
        )

        return await self.smart_llm.do_chat_w_sys_prompt(main_prompt, sys_prompt, no_cache=True)

    async def _query_GPT_for_new_plan_after_parsing_error(
        self,
        error: str,
        error_step_str: str,
        chat_context: str,
        old_plan: str,
        filter_tools: bool = False,
    ) -> str:
        if filter_tools and get_ld_flag(
            "planner-tool-filtering-enabled", default=False, user_context=self.user_id
        ):
            tool_str = await self.get_filtered_tool_str(chat_context, "")
        else:
            tool_str = self.full_tool_string

        sys_prompt = PARSING_ERROR_REPLAN_SYS_PROMPT.format(
            guidelines=PLAN_GUIDELINES,
            example=PLAN_EXAMPLE,
        )

        main_prompt = PARSING_ERROR_REPLAN_MAIN_PROMPT.format(
            error=error,
            error_step_str=error_step_str,
            chat_context=chat_context,
            old_plan=old_plan,
            rules=PLAN_RULES.format(
                rule_complement=RULE_COMPLEMENT,
            ),
            tools=tool_str,
        )

        return await self.smart_llm.do_chat_w_sys_prompt(main_prompt, sys_prompt, no_cache=True)

    async def _query_GPT_for_new_plan_for_completeness(
        self,
        chat_str: str,
        missing_str: str,
        old_plan: str,
    ) -> str:
        sys_prompt_template = COMPLETENESS_REPLAN_SYS_PROMPT
        main_prompt_template = COMPLETENESS_REPLAN_MAIN_PROMPT
        sys_prompt = sys_prompt_template.format(
            guidelines=PLAN_GUIDELINES,
            example=PLAN_EXAMPLE,
        )

        main_prompt = main_prompt_template.format(
            rules=PLAN_RULES.format(
                rule_complement=RULE_COMPLEMENT,
            ),
            tools=self.full_tool_string,
            missing=missing_str,
            chat_context=chat_str,
            old_plan=old_plan,
        )

        return await self.fast_llm.do_chat_w_sys_prompt(main_prompt, sys_prompt, no_cache=True)

    async def _query_GPT_for_new_subplan(
        self,
        directions: str,
        variables: Dict[str, Type[IOType]],
        sample_plans_str: str = "",
        filter_tools: bool = False,
    ) -> str:
        sys_prompt_template = SUBPLANNER_SYS_PROMPT
        main_prompt_template = SUBPLANNER_MAIN_PROMPT
        sys_prompt = sys_prompt_template.format(
            guidelines=PLAN_GUIDELINES,
            example=PLAN_EXAMPLE,
        )

        if filter_tools and get_ld_flag(
            "planner-tool-filtering-enabled", default=False, user_context=self.user_id
        ):
            tool_str = await self.get_filtered_tool_str(directions, sample_plans_str)
        else:
            tool_str = self.full_tool_string

        main_prompt = main_prompt_template.format(
            rules=PLAN_RULES.format(
                rule_complement=RULE_COMPLEMENT,
            ),
            tools=tool_str,
            directions=directions,
            sample_plans=sample_plans_str,
            variables="\n".join([f"{name}: {type}" for name, type in variables.items()]),
        )

        return await self.fast_llm.do_chat_w_sys_prompt(main_prompt, sys_prompt, no_cache=True)

    async def _get_sample_plan_str(self, input: str) -> str:
        logger = get_prefect_logger(__name__)

        # Check for sample plan override file.
        override_sample_plan_file = os.environ.get("AGENT_SAMPLE_PLAN_OVERRIDE_FILE", "")
        if len(override_sample_plan_file) > 0:
            try:
                with open(override_sample_plan_file, "r") as f:
                    sample_plans_str = f.read()
                    logger.warning("Only using override sample plan!!!")
                    logger.info(f"Found relevant sample plan(s):\n{sample_plans_str}")
                    sample_plans_str = PLAN_SAMPLE_TEMPLATE.format(sample_plans=sample_plans_str)
                    return sample_plans_str
            except Exception as e:
                logger.warning(
                    f"Sample plan override failed to load: {e}.  Continuing with normal operation."
                )

        # Normal operation, no overrides.
        sample_plans = await get_similar_sample_plans(
            input, context=self.context, user_id=self.user_id
        )

        if sample_plans:
            sample_plans_str = "\n\n".join(
                [
                    f"{chr(65 + i)}. {sample_plan.get_formatted_plan()}"
                    for i, sample_plan in enumerate(sample_plans)
                ]
            )
            logger.info(f"Found relevant sample plan(s):\n{sample_plans_str}")
            sample_plans_str = PLAN_SAMPLE_TEMPLATE.format(sample_plans=sample_plans_str)
        else:
            logger.info("No relevant sample plan(s) found")
            sample_plans_str = ""

        return sample_plans_str

    @async_perf_logger
    async def get_filtered_tool_str(self, input_str: str, sample_plans: str) -> str:
        main_prompt = SELECT_TOOLS_MAIN_PROMPT.format(
            request=input_str, tools=self.tool_string_function_only, sample_plans=sample_plans
        )
        result = await self.fast_llm.do_chat_w_sys_prompt(
            main_prompt, SELECT_TOOLS_SYS_PROMPT.format(), no_cache=True
        )
        not_wanted_categories = []
        for cat in result.split("\n"):
            try:
                not_wanted_cat = ToolCategory(cat.strip())
            except ValueError:
                continue
            not_wanted_categories.append(not_wanted_cat)
        tool_str = self.tool_registry.get_tool_str(
            self.user_id,
            skip_list=not_wanted_categories,
            user_settings=self.user_settings,
            using_subplanner=self.is_subplanner,
        )
        return tool_str

    def _try_parse_str_literal(self, val: str) -> Optional[str]:
        if (val.startswith('"') and val.endswith('"')) or (
            val.startswith("'") and val.endswith("'")
        ):
            # Strip off the quotes, return a string
            return val[1:-1]
        return None

    def _try_parse_bool_literal(self, val: str) -> Optional[bool]:
        if val == "True":
            return True
        if val == "False":
            return False
        return None

    def _try_parse_int_literal(self, val: str) -> Optional[int]:
        try:
            return int(val)
        except ValueError:
            return None

    def _try_parse_float_literal(self, val: str) -> Optional[float]:
        try:
            return float(val)
        except ValueError:
            return None

    def _try_parse_indexer(
        self, var_name: str, variable_lookup: Dict[str, Type[IOType]]
    ) -> Optional[Variable]:
        components = var_name.split("[")
        if len(components) != 2:
            return None
        list_var_name = components[0]
        index_constant = components[1]
        if not index_constant.endswith("]"):
            return None
        if list_var_name not in variable_lookup:
            return None
        list_type = variable_lookup[list_var_name]
        if get_origin(list_type) not in (List, list):
            raise ExecutionPlanParsingError(
                f"Variable '{list_var_name}' is not a list so cannot be indexed!"
            )

        # This gets the generic type out of the list (e.g. List[int] -> int).
        # This is the type of the value that is indexed, so it is the type we want to return.
        inner_type = get_args(list_type)[0]
        index_constant = index_constant.removesuffix("]")
        # At this point, the list variable is stored in `list_var_name`, and the
        # index is stored in `index_constant`
        index_int = self._try_parse_int_literal(index_constant)
        if index_int is None:
            raise ExecutionPlanParsingError(
                (
                    f"List variable '{list_var_name}' has invalid indexer "
                    f"'{index_constant}', must be integer constant!"
                )
            )
        return Variable(var_name=list_var_name, var_type=inner_type, index=index_int)

    def _try_parse_variable(
        self, var_name: str, variable_lookup: Dict[str, Type[IOType]]
    ) -> Optional[Variable]:
        if "[" in var_name:
            return self._try_parse_indexer(var_name=var_name, variable_lookup=variable_lookup)
        var_type = variable_lookup.get(var_name)
        if not var_type:
            return None
        return Variable(var_name=var_name, var_type=var_type)

    def _try_parse_primitive_literal(
        self, val: str, expected_type: Optional[Type]
    ) -> Optional[IOType]:
        parsed_val: Optional[PrimitiveType] = None
        if expected_type is float:
            parsed_val = self._try_parse_float_literal(val)
        elif expected_type is int:
            parsed_val = self._try_parse_int_literal(val)
        elif expected_type is str:
            parsed_val = self._try_parse_str_literal(val)
        elif expected_type is bool:
            parsed_val = self._try_parse_bool_literal(val)
        else:
            # Try them all!
            parsed_val = self._try_parse_int_literal(val)
            if parsed_val is None:
                parsed_val = self._try_parse_float_literal(val)
            if parsed_val is None:
                parsed_val = self._try_parse_str_literal(val)
            if parsed_val is None:
                parsed_val = self._try_parse_bool_literal(val)

        return parsed_val

    def _try_parse_variable_or_literal(
        self,
        val_or_name: str,
        expected_type: Optional[Type],
        variable_lookup: Dict[str, Type[IOType]],
    ) -> Optional[Union[IOType, Variable]]:
        # First try to parse as a literal
        output = self._try_parse_primitive_literal(val=val_or_name, expected_type=expected_type)
        if output is None:
            variable = self._try_parse_variable(
                var_name=val_or_name, variable_lookup=variable_lookup
            )
            if variable is None:
                return None
            if not check_type_is_valid(variable.var_type, expected_type):
                raise ExecutionPlanParsingError(
                    f"Variable {val_or_name} is invalid type, got {variable.var_type}, expecting {expected_type}."
                )
            output = variable
        elif expected_type not in (str, int, float, bool):
            if not check_type_is_valid(type(output), expected_type):
                raise ExecutionPlanParsingError(
                    f"Variable {val_or_name} is invalid type, got {type(output)}, expecting {expected_type}."
                )

        return output

    def _parse_ast_elem(self, item: Any) -> Any:
        if isinstance(item, ast.Constant):
            val = item.value
            if isinstance(val, str):
                return f'"{val}"'
            else:
                return str(val)
        elif isinstance(item, ast.Name):
            return item.id
        elif isinstance(item, ast.Subscript):
            # We're going to parse this later
            return f"{item.value.id}[{item.slice.value}]"  # type: ignore
        elif isinstance(item, ast.Attribute):
            # We're going to parse this later (TODO)
            return f"{item.value.id}.{item.attr}"  # type: ignore
        else:
            raise ExecutionPlanParsingError(f"Value '{item}' has list with non supported syntax")

    def _split_dict_literal_with_ast(self, val: str) -> Dict[str, str]:
        expr = ast.parse(val).body[0]
        assert isinstance(expr, ast.Expr)
        dict_expr = expr.value
        assert isinstance(dict_expr, ast.Dict)
        output = {}
        keys = dict_expr.keys
        values = dict_expr.values
        assert len(keys) == len(values)
        for key_elem, item_elem in zip(keys, values):
            key = self._parse_ast_elem(key_elem)
            item = self._parse_ast_elem(item_elem)
            output[key] = item
        return output

    def _split_list_literal_with_ast(self, val: str) -> List[str]:
        """
        A smarter version of string.split(",") that handles commas inside of
        list components.
        """
        try:
            expr = ast.parse(val).body[0]
            assert isinstance(expr, ast.Expr)
            list_expr = expr.value
            assert isinstance(list_expr, ast.List)
            items = list_expr.elts
            output = []
            for item in items:
                val = self._parse_ast_elem(item)
                output.append(val)
            return output
        except Exception:
            logger = get_prefect_logger(__name__)
            logger.exception(f"Failed to parse list with ast: {val}")
            contents_str = val[1:-1]
            return contents_str.split(",")

    def _try_parse_dict_literal(
        self, val: str, expected_type: Optional[Type], variable_lookup: Dict[str, Type[IOType]]
    ) -> Optional[Dict[IOType, Union[IOType, Variable]]]:
        if (not val.startswith("{")) or (not val.endswith("}")):
            return None

        type_args = get_args(expected_type)
        if len(type_args) != 2:
            return None
        key_type = type_args[0]
        val_type = type_args[1]
        # TODO add better type checking, right now not really necessary
        elements = self._split_dict_literal_with_ast(val)
        output = {}
        for key, elem in elements.items():
            key = key.strip()
            elem = elem.strip()
            parsed_key = self._try_parse_variable_or_literal(
                key, expected_type=key_type, variable_lookup=variable_lookup
            )
            parsed_elem = self._try_parse_variable_or_literal(
                elem, expected_type=val_type, variable_lookup=variable_lookup
            )
            if parsed_key is None or parsed_elem is None:
                raise ExecutionPlanParsingError(
                    f"Element ({key}, {elem}) of dict {val} is invalid type, expecting {expected_type}"
                )
            output[parsed_key] = parsed_elem

        return output

    def _try_parse_list_literal(
        self,
        val: str,
        expected_type: Optional[Type],
        variable_lookup: Dict[str, Type[IOType]],
    ) -> Optional[List[Union[IOType, Variable]]]:
        if (not val.startswith("[")) or (not val.endswith("]")):
            return None
        list_type_tup: Tuple[Type] = get_args(expected_type)
        if list_type_tup[-1] is type(None):  # special Optional case, need to go an extra level in
            list_type_tup = get_args(list_type_tup[0])
        list_type: Type
        if not list_type_tup:
            list_type = Any
        else:
            list_type = list_type_tup[0]
        elements = self._split_list_literal_with_ast(val)
        output = []
        for i, elem in enumerate(elements):
            elem = elem.strip()
            parsed = self._try_parse_variable_or_literal(
                elem, expected_type=list_type, variable_lookup=variable_lookup
            )
            if parsed is None:
                raise ExecutionPlanParsingError(
                    f"Element {i} of list {val} is invalid type, expecting {expected_type}"
                )
            output.append(parsed)

        return output

    def _validate_tool_arguments(
        self, tool: Tool, args: Dict[str, str], variable_lookup: Dict[str, Type[IOType]]
    ) -> PartialToolArgs:
        logger = get_prefect_logger(__name__)
        """
        Given a tool, a set of arguments, and a variable lookup table, validate
        all the arguments for the tool given by GPT. If validation is
        successful, return a PartialToolArgs, which represents both literals and
        variables that must be bound to values at execution time.
        """
        expected_args = tool.input_type.model_fields
        num_required_args = len([arg for arg in expected_args.values() if arg.is_required()])
        if len(args) < num_required_args:
            raise ExecutionPlanParsingError(f"Tool '{tool.name}' has missing required arguments!")
        parsed_args: PartialToolArgs = {}

        for arg, val in args.items():
            if not val:
                raise ExecutionPlanParsingError(
                    f"Tool '{tool.name} has invalid empty argument for '{arg}'"
                )
            if arg not in expected_args:
                raise ExecutionPlanParsingError(f"Tool '{tool.name}' has invalid argument '{arg}'")

            arg_info = expected_args[arg]

            # GPT isn't supposed to put Nones in the input but in case it does, we can just skip it
            # (Optional variables must always default to None!)
            if val == "None":
                if arg_info.default is None:
                    # valid when type is `x: some_type = None`
                    continue

                # now, check if the type has `Optional`
                inner_args = get_args(arg_info.annotation)
                if not inner_args or type(None) not in inner_args:
                    raise ExecutionPlanParsingError(
                        f"Tool '{tool.name}' has invalid None argument for '{arg}'"
                    )
                else:
                    continue

            # For literals, we can parse out an actual value at "compile" time

            parsed_val = self._try_parse_primitive_literal(val, expected_type=arg_info.annotation)
            if parsed_val is None:
                parsed_val = self._try_parse_list_literal(
                    val, arg_info.annotation, variable_lookup=variable_lookup
                )

            if parsed_val is None:
                parsed_val = self._try_parse_dict_literal(
                    val, arg_info.annotation, variable_lookup=variable_lookup
                )

            if parsed_val is not None:
                literal_typ: Type = type(parsed_val)
                expected = arg_info.annotation
                if not check_type_is_valid(actual=literal_typ, expected=expected):
                    raise ExecutionPlanParsingError(
                        (
                            f"Tool '{tool.name}' has incorrectly typed literal argument for '{arg}'. "
                            f"Expected '{arg_info.annotation}' but found '{val}' ({literal_typ})"
                        )
                    )
                # We have found a literal. Add it to the map,
                # and move on to the next argument.
                parsed_args[arg] = parsed_val
                continue

            variable = self._try_parse_variable(val, variable_lookup=variable_lookup)
            # First check for an undefined variable
            if not variable:
                logger.warning(
                    f"{tool.name}' has undefined variable argument '{val}]' for arg '{arg}"
                    f", {variable_lookup=}"
                )
                raise ExecutionPlanParsingError(
                    f"Tool '{tool.name}' has undefined variable argument '{val}' for arg '{arg}'."
                )
            # Next, check if the variable's type matches the expected type for the argument
            if not check_type_is_valid(actual=variable.var_type, expected=arg_info.annotation):
                raise ExecutionPlanParsingError(
                    (
                        f"Tool '{tool.name}' has a type mismatch for arg '{str(arg)}',"
                        f" expected '{arg_info.annotation}' but found '{val}: {variable.var_type}'"
                    )
                )
            parsed_args[arg] = variable

        return parsed_args

    def _validate_and_construct_plan(
        self, steps: List[ParsedStep], variable_lookup: Optional[Dict[str, Type[IOType]]] = None
    ) -> ExecutionPlan:
        if variable_lookup is None:
            variable_lookup = {}
        plan_nodes: List[ToolExecutionNode] = []
        has_output_tool = False
        for step in steps:
            try:
                if not self.tool_registry.is_tool_registered(step.function):
                    raise ExecutionPlanParsingError(f"Invalid function '{step.function}'")

                tool = self.tool_registry.get_tool(step.function)
                if tool.is_output_tool:
                    has_output_tool = True
                partial_args = self._validate_tool_arguments(
                    tool, args=step.arguments, variable_lookup=variable_lookup
                )
                if tool.output_type_transformation:
                    arg_type_dict = get_types_from_tool_args(
                        args=partial_args, var_type_lookup=variable_lookup
                    )
                    variable_lookup[step.output_var] = tool.output_type_transformation(
                        arg_type_dict
                    )
                else:
                    variable_lookup[step.output_var] = tool.return_type

                node = ToolExecutionNode(
                    tool_name=step.function,
                    args=partial_args,
                    description=step.description,
                    output_variable_name=step.output_var,
                    tool_task_id=str(uuid4()),
                    is_output_node=tool.is_output_tool,
                    store_output=tool.store_output,
                )
                plan_nodes.append(node)
            except Exception as e:
                raise ExecutionPlanParsingError(
                    str(e),
                    raw_plan="\n".join([temp_step.original_str for temp_step in steps]),
                    line=step.original_str,
                )

        if not has_output_tool:
            raise ExecutionPlanParsingError(
                "No call to `prepare_output` found!",
                raw_plan="\n".join([temp_step.original_str for temp_step in steps]),
            )

        return ExecutionPlan(nodes=plan_nodes)

    def _parse_plan_str(self, plan_str: str) -> List[ParsedStep]:
        plan_steps: List[ParsedStep] = []
        stripped_plan = [
            line
            for line in plan_str.split("\n")
            if not (line.startswith("#") or line.startswith("```") or not line.strip())
        ]
        for line in stripped_plan:
            try:
                match = ASSIGNMENT_RE.match(line)

                if match is None:
                    raise ExecutionPlanParsingError(
                        "Basic formatting error in plan, could not match regex"
                    )

                output_var, function, arguments, description = match.groups()
                arg_dict = get_arg_dict(arguments)
                plan_steps.append(
                    ParsedStep(
                        output_var=output_var,
                        function=function,
                        arguments=arg_dict,
                        description=description,
                        original_str=line,
                    )
                )
            except Exception as e:
                raise ExecutionPlanParsingError(
                    str(e), raw_plan="\n".join(stripped_plan), line=line
                )

        if len(plan_steps) == 0:
            raise RuntimeError("No steps in the plan")

        return plan_steps


async def main() -> None:
    input_text = "Can you give me a single summary of news published in the last week about machine learning at Meta, Apple, and Microsoft?"  # noqa: E501
    user_message = Message(message=input_text, is_user_message=True, message_time=get_now_utc())
    AI_response = "Okay, I'm doing that summary for you."
    AI_message = Message(message=AI_response, is_user_message=True, message_time=get_now_utc())
    chat_context = ChatContext(messages=[user_message, AI_message])
    planner = Planner("123")
    plan = await planner.create_initial_plan(chat_context, plan_id=str(uuid4()))

    print(plan)


if __name__ == "__main__":
    asyncio.run(main())
