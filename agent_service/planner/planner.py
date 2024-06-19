import asyncio
import datetime
import json
import traceback
from typing import Any, Dict, List, Optional, Tuple, Type, Union, get_args
from uuid import uuid4

from agent_service.chatbot.chatbot import Chatbot
from agent_service.GPT.constants import DEFAULT_SMART_MODEL, GPT4_O
from agent_service.GPT.requests import GPT
from agent_service.io_type_utils import IOType, PrimitiveType, check_type_is_valid
from agent_service.planner.constants import (
    ARGUMENT_RE,
    ASSIGNMENT_RE,
    INITIAL_PLAN_TRIES,
    MIN_SUCCESSFUL_FOR_STOP,
    Action,
)
from agent_service.planner.planner_types import (
    ErrorInfo,
    ExecutionPlan,
    ExecutionPlanParsingError,
    ParsedStep,
    PartialToolArgs,
    ToolExecutionNode,
    Variable,
)
from agent_service.planner.prompts import (
    BREAKDOWN_NEED_MAIN_PROMPT,
    BREAKDOWN_NEED_SYS_PROMPT,
    ERROR_REPLAN_MAIN_PROMPT,
    ERROR_REPLAN_SYS_PROMPT,
    PICK_BEST_PLAN_MAIN_PROMPT,
    PICK_BEST_PLAN_SYS_PROMPT,
    PLAN_EXAMPLE,
    PLAN_GUIDELINES,
    PLAN_RULES,
    PLAN_SAMPLE_TEMPLATE,
    PLANNER_MAIN_PROMPT,
    PLANNER_SYS_PROMPT,
    USER_INPUT_APPEND_MAIN_PROMPT,
    USER_INPUT_APPEND_SYS_PROMPT,
    USER_INPUT_REPLAN_MAIN_PROMPT,
    USER_INPUT_REPLAN_SYS_PROMPT,
)
from agent_service.planner.utils import get_similar_sample_plans
from agent_service.tool import Tool, ToolRegistry

# Make sure all tools are imported for the planner
from agent_service.tools import *  # noqa
from agent_service.types import ChatContext, Message
from agent_service.utils.agent_event_utils import send_chat_message
from agent_service.utils.async_utils import gather_with_concurrency, gather_with_stop
from agent_service.utils.date_utils import get_now_utc
from agent_service.utils.event_logging import log_event
from agent_service.utils.gpt_logging import GptJobIdType, GptJobType, create_gpt_context
from agent_service.utils.logs import async_perf_logger
from agent_service.utils.postgres import get_psql
from agent_service.utils.prefect import get_prefect_logger

logger = get_prefect_logger(__name__)


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
        key, value = arg_str[arg_indicies[i] : arg_indicies[i + 1]].strip(" ,").split("=")
        arg_dict[key.strip()] = value.strip()
    return arg_dict


class Planner:
    def __init__(
        self,
        agent_id: str,
        tool_registry: Type[ToolRegistry] = ToolRegistry,
        send_chat: bool = True,
        skip_db_commit: bool = False,
    ) -> None:
        self.agent_id = agent_id
        self.context = create_gpt_context(GptJobType.AGENT_PLANNER, agent_id, GptJobIdType.AGENT_ID)
        self.smart_llm = GPT(self.context, DEFAULT_SMART_MODEL)
        self.fast_llm = GPT(self.context, GPT4_O)
        self.tool_registry = tool_registry
        self.tool_string = tool_registry.get_tool_str()
        self.send_chat = send_chat
        self.skip_db_commit = skip_db_commit
        self.db = get_psql(skip_commit=skip_db_commit)

    @async_perf_logger
    async def create_initial_plan(
        self, chat_context: ChatContext, use_sample_plans: bool = True
    ) -> Optional[ExecutionPlan]:

        if use_sample_plans:
            logger.info("Getting matching sample plans")
            sample_plans_str = await self._get_sample_plan_str(
                input=chat_context.get_gpt_input(client_only=True)
            )
        else:
            sample_plans_str = ""

        logger.info("Writing plan")
        first_round_tasks = [
            self._create_initial_plan(chat_context, self.fast_llm, sample_plans=sample_plans_str)
            for _ in range(INITIAL_PLAN_TRIES)
        ]
        first_round_results = await gather_with_stop(first_round_tasks, MIN_SUCCESSFUL_FOR_STOP)

        if first_round_results:
            logger.info(
                f"{len(first_round_results)} of {INITIAL_PLAN_TRIES} initial plan runs succeeded"
            )
            first_round_results = list(set(first_round_results))  # get rid of complete duplicates
            # GPT 4O seems to have done it now need to just pick
            if len(first_round_results) > 1:
                best_plan = await self._pick_best_plan(chat_context, first_round_results)
            else:
                best_plan = first_round_results[0]

            return best_plan

        if self.send_chat:
            await self._send_delayed_planning_message(chat_context)

        logger.warning(f"All of {INITIAL_PLAN_TRIES} initial plan runs failed, trying round 2")

        second_round_tasks = [
            self._create_initial_plan(chat_context, self.smart_llm),
            self._get_plan_from_breakdown(chat_context),
        ]

        turbo_plan, breakdown_plan = await gather_with_concurrency(second_round_tasks)

        if turbo_plan is not None:
            logger.info("Round 2 turbo run succeeded, using that plan")
            # if we were able to get a working version with turbo, use that
            logger.info(f"New Plan:\n{turbo_plan.get_formatted_plan()}")
            return turbo_plan

        if breakdown_plan:
            logger.info("Round 2 breakdown run succeeded, using best breakdown plan")
        else:
            logger.warning("Round 2 planning failed, giving up")

        # otherwise, return the best breakdown version if any
        logger.info(f"New Plan:\n{breakdown_plan.get_formatted_plan()}")
        return breakdown_plan

    async def _get_plan_from_breakdown(self, chat_context: ChatContext) -> Optional[ExecutionPlan]:
        request_breakdown = await self._get_request_breakdown(chat_context)
        breakdown_tasks = [
            self._create_initial_plan(
                ChatContext(messages=[Message(message=subneed, is_user_message=True)])
            )
            for subneed in request_breakdown
        ]
        breakdown_results = [
            plan for plan in await gather_with_concurrency(breakdown_tasks) if plan
        ]
        if breakdown_results:
            if len(breakdown_results) > 1:
                # TODO: Combine best plans instead of just picking the best one?
                best_plan = await self._pick_best_plan(chat_context, breakdown_results)
            else:
                best_plan = breakdown_results[0]
        else:
            best_plan = None

        return best_plan

    @async_perf_logger
    async def _create_initial_plan(
        self, chat_context: ChatContext, llm: Optional[GPT] = None, sample_plans: str = ""
    ) -> Optional[ExecutionPlan]:
        execution_plan_start = datetime.datetime.utcnow().isoformat()
        if llm is None:
            llm = self.fast_llm

        prompt = chat_context.get_gpt_input()
        plan_str = await self._query_GPT_for_initial_plan(
            chat_context.get_gpt_input(), llm=llm, sample_plans=sample_plans
        )
        agent_id = get_agent_id_from_chat_context(context=chat_context)

        try:
            steps = self._parse_plan_str(plan_str)
            plan = self._validate_and_construct_plan(steps)
        except Exception as e:
            logger.warning(
                f"Failed to parse and validate plan with original LLM output string:\n{plan_str}"
            )
            logger.warning(f"Failed to parse and validate plan due to exception: {repr(e)}")
            log_event(
                event_name="agent_plan_generated",
                event_data={
                    "started_at_utc": execution_plan_start,
                    "finished_at_utc": datetime.datetime.utcnow().isoformat(),
                    "error_message": traceback.format_exc(),
                    "plan_str": plan_str,
                    "agent_id": agent_id,
                    "model_id": llm.model,
                    "prompt": prompt,
                    "action": Action.CREATE,
                },
            )
            return None
        log_event(
            event_name="agent_plan_generated",
            event_data={
                "started_at_utc": execution_plan_start,
                "finished_at_utc": datetime.datetime.utcnow().isoformat(),
                "execution_plan": plan_to_json(plan=plan),
                "plan_str": plan_str,
                "agent_id": agent_id,
                "model_id": llm.model,
                "prompt": prompt,
                "action": Action.CREATE,
            },
        )
        return plan

    @async_perf_logger
    async def _pick_best_plan(
        self, chat_context: ChatContext, plans: List[ExecutionPlan]
    ) -> ExecutionPlan:
        plan_pick_started_at = datetime.datetime.utcnow().isoformat()
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
                "finished_at_utc": datetime.datetime.utcnow().isoformat(),
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
        action: Action,
        use_sample_plans: bool = True,
    ) -> Optional[ExecutionPlan]:

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
                chat_context, last_plan, latest_user_messages, sample_plans_str, action=action
            )
            for _ in range(INITIAL_PLAN_TRIES)
        ]
        results = await gather_with_stop(tasks, stop_count=MIN_SUCCESSFUL_FOR_STOP)
        if results:
            logger.info(f"{len(results)} of {INITIAL_PLAN_TRIES} replan runs succeeded")
            results = list(set(results))  # get rid of complete duplicates and avoid best pick
            # GPT 4O seems to have done it now need to just pick
            if len(results) > 1:
                best_plan = await self._pick_best_plan(chat_context, results)
            else:
                best_plan = results[0]

            return best_plan

        logger.warning(f"All of {INITIAL_PLAN_TRIES} replan runs failed")

        return None

    @async_perf_logger
    async def _rewrite_plan_after_input(
        self,
        chat_context: ChatContext,
        last_plan: ExecutionPlan,
        latest_messages: List[Message],
        sample_plans_str: str,
        action: Action,
    ) -> Optional[ExecutionPlan]:
        # for now, just assume last step is what is new
        # TODO: multiple old plans, flexible chat cutoff
        execution_plan_start = datetime.datetime.utcnow().isoformat()
        agent_id = get_agent_id_from_chat_context(context=chat_context)
        logger = get_prefect_logger(__name__)
        main_chat_str = chat_context.get_gpt_input()
        new_messages = "\n".join([message.get_gpt_input() for message in latest_messages])
        old_plan_str = last_plan.get_formatted_plan()
        new_plan_str = await self._query_GPT_for_new_plan_after_input(
            new_messages, main_chat_str, old_plan_str, sample_plans_str, action=action
        )
        append = action == Action.APPEND
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
                    "finished_at_utc": datetime.datetime.utcnow().isoformat(),
                    "execution_plan": plan_to_json(plan=new_plan),
                    "plan_str": plan_str,
                    "agent_id": agent_id,
                    "model_id": self.fast_llm.model,
                    "prompt": main_chat_str,
                    "action": action,
                },
            )
        except Exception:
            log_event(
                event_name="agent_plan_generated",
                event_data={
                    "started_at_utc": execution_plan_start,
                    "finished_at_utc": datetime.datetime.utcnow().isoformat(),
                    "error_message": traceback.format_exc(),
                    "plan_str": plan_str,
                    "agent_id": agent_id,
                    "model_id": self.fast_llm.model,
                    "prompt": main_chat_str,
                    "action": action,
                },
            )
            logger.warning(f"Failed to validate replan with original LLM output string: {plan_str}")
            return None

        return new_plan

    @async_perf_logger
    async def rewrite_plan_after_error(
        self,
        error_info: ErrorInfo,
        chat_context: ChatContext,
        last_plan: ExecutionPlan,
        action: Action,
        use_sample_plans: bool = True,
    ) -> Optional[ExecutionPlan]:

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
                error_info, chat_context, last_plan, sample_plans_str, action=action
            )
            for _ in range(INITIAL_PLAN_TRIES)
        ]
        results = await gather_with_stop(tasks, stop_count=MIN_SUCCESSFUL_FOR_STOP)
        if results:
            logger.info(f"{len(results)} of {INITIAL_PLAN_TRIES} replan runs succeeded")
            results = list(set(results))  # get rid of complete duplicates
            # GPT 4O seems to have done it now need to just pick
            if len(results) > 1:
                best_plan = await self._pick_best_plan(chat_context, results)
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
        action: Action,
    ) -> Optional[ExecutionPlan]:
        execution_plan_start = datetime.datetime.utcnow().isoformat()
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
                    "finished_at_utc": datetime.datetime.utcnow().isoformat(),
                    "execution_plan": plan_to_json(plan=new_plan),
                    "plan_str": new_plan_str,
                    "agent_id": agent_id,
                    "model_id": self.smart_llm.model,
                    "prompt": error_str,
                    "action": action,
                },
            )
        except Exception:
            log_event(
                event_name="agent_plan_generated",
                event_data={
                    "started_at_utc": execution_plan_start,
                    "finished_at_utc": datetime.datetime.utcnow().isoformat(),
                    "error_message": traceback.format_exc(),
                    "plan_str": new_plan_str,
                    "agent_id": agent_id,
                    "model_id": self.smart_llm.model,
                    "prompt": error_str,
                    "action": action,
                },
            )
            logger.warning(
                f"Failed to validate replan with original LLM output string: {new_plan_str}"
            )
            return None

        return new_plan

    async def _query_GPT_for_initial_plan(
        self, user_input: str, llm: GPT, sample_plans: str = ""
    ) -> str:
        sys_prompt = PLANNER_SYS_PROMPT.format(
            rules=PLAN_RULES,
            guidelines=PLAN_GUIDELINES,
            example=PLAN_EXAMPLE,
            tools=self.tool_string,
        )

        main_prompt = PLANNER_MAIN_PROMPT.format(message=user_input, sample_plans=sample_plans)
        return await llm.do_chat_w_sys_prompt(main_prompt, sys_prompt, no_cache=True)

    async def _query_GPT_for_new_plan_after_input(
        self,
        new_user_input: str,
        existing_context: str,
        old_plan: str,
        sample_plans_str: str,
        action: Action,
    ) -> str:
        if action == Action.APPEND:
            sys_prompt_template = USER_INPUT_APPEND_SYS_PROMPT
            main_prompt_template = USER_INPUT_APPEND_MAIN_PROMPT
        else:
            sys_prompt_template = USER_INPUT_REPLAN_SYS_PROMPT
            main_prompt_template = USER_INPUT_REPLAN_MAIN_PROMPT
        sys_prompt = sys_prompt_template.format(
            rules=PLAN_RULES,
            guidelines=PLAN_GUIDELINES,
            example=PLAN_EXAMPLE,
            tools=self.tool_string,
        )

        main_prompt = main_prompt_template.format(
            new_message=new_user_input,
            chat_context=existing_context,
            old_plan=old_plan,
            sample_plans=sample_plans_str,
        )

        return await self.fast_llm.do_chat_w_sys_prompt(main_prompt, sys_prompt, no_cache=True)

    async def _query_GPT_for_new_plan_after_error(
        self, error: str, failed_step: str, chat_context: str, old_plan: str, sample_plans_str: str
    ) -> str:
        sys_prompt = ERROR_REPLAN_SYS_PROMPT.format(
            rules=PLAN_RULES,
            guidelines=PLAN_GUIDELINES,
            example=PLAN_EXAMPLE,
            tools=self.tool_string,
        )

        main_prompt = ERROR_REPLAN_MAIN_PROMPT.format(
            error=error,
            failed_step=failed_step,
            chat_context=chat_context,
            old_plan=old_plan,
            sample_plans=sample_plans_str,
        )

        return await self.smart_llm.do_chat_w_sys_prompt(main_prompt, sys_prompt, no_cache=True)

    async def _get_sample_plan_str(self, input: str) -> str:

        sample_plans = await get_similar_sample_plans(input, self.context)

        if sample_plans:
            sample_plans_str = "\n\n".join(
                [sample_plan.get_formatted_plan() for sample_plan in sample_plans]
            )
            logger.info(f"Found relevant sample plan(s): {sample_plans_str}")
            sample_plans_str = PLAN_SAMPLE_TEMPLATE.format(sample_plans=sample_plans_str)
        else:
            logger.info("No relevant sample plan(s) found")
            sample_plans_str = ""

        return sample_plans_str

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

    def _try_parse_variable(
        self, var_name: str, variable_lookup: Dict[str, Type[IOType]]
    ) -> Optional[Type[IOType]]:
        return variable_lookup.get(var_name)

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
            var_type = self._try_parse_variable(
                var_name=val_or_name, variable_lookup=variable_lookup
            )
            if var_type is None:
                return None
            if not check_type_is_valid(var_type, expected_type):
                raise ExecutionPlanParsingError(
                    f"Variable {val_or_name} is invalid type, got {var_type}, expecting {expected_type}."
                )
            output = Variable(var_name=val_or_name)

        return output

    def _try_parse_list_literal(
        self,
        val: str,
        expected_type: Optional[Type],
        variable_lookup: Dict[str, Type[IOType]],
    ) -> Optional[List[Union[IOType, Variable]]]:
        if (not val.startswith("[")) or (not val.endswith("]")):
            return None
        contents_str = val[1:-1]
        list_type_tup: Tuple[Type] = get_args(expected_type)
        if list_type_tup[-1] is type(None):  # special Optional case, need to go an extra level in
            list_type_tup = get_args(list_type_tup[0])
        list_type: Type
        if not list_type_tup:
            list_type = Any
        else:
            list_type = list_type_tup[0]
        elements = contents_str.split(",")
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
        """
        Given a tool, a set of arguments, and a variable lookup table, validate
        all the arguments for the tool given by GPT. If validation is
        successful, return a PartialToolArgs, which represents both literals and
        variables that must be bound to values at execution time.
        """
        expected_args = tool.input_type.model_fields
        parsed_args: PartialToolArgs = {}

        for arg, val in args.items():
            if not val:
                raise ExecutionPlanParsingError(
                    f"Tool '{tool.name} has invalid empty argument for '{arg}'"
                )
            if arg not in expected_args:
                raise ExecutionPlanParsingError(f"Tool '{tool.name}' has invalid argument '{arg}'")

            arg_info = expected_args[arg]

            # For literals, we can parse out an actual value at "compile" time
            parsed_val = self._try_parse_primitive_literal(val, expected_type=arg_info.annotation)
            if parsed_val is None:
                parsed_val = self._try_parse_list_literal(
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

            parsed_variable_type = self._try_parse_variable(val, variable_lookup=variable_lookup)
            # First check for an undefined variable
            if not parsed_variable_type:
                logger.warning(
                    f"{tool.name}' has undefined variable argument '{arg} "
                    f"{val=}, {variable_lookup=}"
                )
                raise ExecutionPlanParsingError(
                    f"Tool '{tool.name}' has undefined variable argument '{arg}'."
                )
            # Next, check if the variable's type matches the expected type for the argument
            if not check_type_is_valid(actual=parsed_variable_type, expected=arg_info.annotation):
                raise ExecutionPlanParsingError(
                    (
                        f"Tool '{tool.name}' has a type mismatch for arg '{str(arg)}',"
                        f" expected '{arg_info.annotation}' but found '{val}: {parsed_variable_type}'"
                    )
                )
            parsed_args[arg] = Variable(var_name=val)

        return parsed_args

    def _validate_and_construct_plan(self, steps: List[ParsedStep]) -> ExecutionPlan:
        variable_lookup: Dict[str, Type[IOType]] = {}
        plan_nodes: List[ToolExecutionNode] = []
        has_output_tool = False
        for step in steps:
            if not self.tool_registry.is_tool_registered(step.function):
                raise ExecutionPlanParsingError(f"Invalid function '{step.function}'")

            tool = self.tool_registry.get_tool(step.function)
            if tool.is_output_tool:
                has_output_tool = True
            partial_args = self._validate_tool_arguments(
                tool, args=step.arguments, variable_lookup=variable_lookup
            )
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

        if not has_output_tool:
            raise ExecutionPlanParsingError("No call to `prepare_output` found!")

        return ExecutionPlan(nodes=plan_nodes)

    def _parse_plan_str(self, plan_str: str) -> List[ParsedStep]:
        plan_steps: List[ParsedStep] = []
        for line in plan_str.split("\n"):
            if line.startswith("#"):  # allow initial comment line
                continue
            if line.startswith("```"):  # GPT likes to add this to Python code
                continue
            if not line.strip():
                continue

            match = ASSIGNMENT_RE.match(line)

            if match is None:
                raise ExecutionPlanParsingError(f"Failed to parse plan line: {line}")

            output_var, function, arguments, description = match.groups()
            arg_dict = get_arg_dict(arguments)
            plan_steps.append(
                ParsedStep(
                    output_var=output_var,
                    function=function,
                    arguments=arg_dict,
                    description=description,
                )
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
    plan = await planner.create_initial_plan(chat_context)

    print(plan)

    # new_user_input = "I need you to include Amazon in the summary as well"
    new_user_input = "Focus the summary on Generative AI research"

    new_message = Message(message=new_user_input, is_user_message=True, message_time=get_now_utc())
    chat_context.messages.append(new_message)
    updated_plan = await planner.rewrite_plan_after_input(chat_context, plan)
    print(updated_plan)
    chat_context.messages.pop()


if __name__ == "__main__":
    asyncio.run(main())
