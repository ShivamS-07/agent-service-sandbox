import json
from datetime import date
from typing import Optional

from gpt_service_proto_v1.service_grpc import GPTServiceStub

from agent_service.chatbot.prompts import (
    AGENT_DESCRIPTION_PROMPT,
    ERROR_REPLAN_POSTPLAN_MAIN_PROMPT,
    ERROR_REPLAN_POSTPLAN_SYS_PROMPT,
    ERROR_REPLAN_PREPLAN_MAIN_PROMPT,
    ERROR_REPLAN_PREPLAN_SYS_PROMPT,
    FIRST_RESPONSE_NONE_MAIN_PROMPT,
    FIRST_RESPONSE_NONE_SYS_PROMPT,
    FIRST_RESPONSE_REFER_MAIN_PROMPT,
    FIRST_RESPONSE_REFER_SYS_PROMPT,
    INITIAL_MIDPLAN_MAIN_PROMPT,
    INITIAL_MIDPLAN_SYS_PROMPT,
    INITIAL_PLAN_FAILED_MAIN_PROMPT,
    INITIAL_PLAN_FAILED_SUGGESTIONS_MAIN_PROMPT,
    INITIAL_PLAN_FAILED_SUGGESTIONS_SYS_PROMPT,
    INITIAL_PLAN_FAILED_SYS_PROMPT,
    INITIAL_POSTPLAN_INCOMPLETE_MAIN_PROMPT,
    INITIAL_POSTPLAN_INCOMPLETE_SYS_PROMPT,
    INITIAL_POSTPLAN_MAIN_PROMPT,
    INITIAL_POSTPLAN_SYS_PROMPT,
    INITIAL_PREPLAN_MAIN_PROMPT,
    INITIAL_PREPLAN_SYS_PROMPT,
    INPUT_UPDATE_NO_ACTION_MAIN_PROMPT,
    INPUT_UPDATE_NO_ACTION_SYS_PROMPT,
    INPUT_UPDATE_REPLAN_POSTPLAN_MAIN_PROMPT,
    INPUT_UPDATE_REPLAN_POSTPLAN_SYS_PROMPT,
    INPUT_UPDATE_REPLAN_PREPLAN_MAIN_PROMPT,
    INPUT_UPDATE_REPLAN_PREPLAN_SYS_PROMPT,
    INPUT_UPDATE_RERUN_MAIN_PROMPT,
    INPUT_UPDATE_RERUN_SYS_PROMPT,
    NON_RETRIABLE_ERROR_MAIN_PROMPT,
    NON_RETRIABLE_ERROR_SYS_PROMPT,
    NOTIFICATION_UPDATE_MAIN_PROMPT,
    NOTIFICATION_UPDATE_SYS_PROMPT,
)
from agent_service.GPT.constants import GPT4_O
from agent_service.GPT.requests import GPT
from agent_service.planner.planner_types import (
    ErrorInfo,
    ExecutionPlan,
    ToolExecutionNode,
)
from agent_service.tool import ToolRegistry, default_tool_registry
from agent_service.types import ChatContext
from agent_service.utils.gpt_logging import chatbot_context
from agent_service.utils.logs import async_perf_logger


class Chatbot:
    def __init__(
        self,
        agent_id: str,
        model: str = GPT4_O,
        gpt_service_stub: Optional[GPTServiceStub] = None,
        tool_registry: Optional[ToolRegistry] = None,
    ) -> None:
        self.agent_id = agent_id
        context = chatbot_context(agent_id=agent_id)
        self.llm = GPT(context, model, gpt_service_stub=gpt_service_stub)
        self.tool_registry = tool_registry or default_tool_registry()

    @async_perf_logger
    async def generate_first_response_refer(self, chat_context: ChatContext) -> str:
        main_prompt = FIRST_RESPONSE_REFER_MAIN_PROMPT.format(
            chat_context=chat_context.get_gpt_input()
        )
        sys_prompt = FIRST_RESPONSE_REFER_SYS_PROMPT.format(
            agent_description=await self.prepare_agent_description()
        )
        result = await self.llm.do_chat_w_sys_prompt(main_prompt, sys_prompt, max_tokens=50)
        return result

    @async_perf_logger
    async def generate_first_response_none(self, chat_context: ChatContext) -> str:
        main_prompt = FIRST_RESPONSE_NONE_MAIN_PROMPT.format(
            chat_context=chat_context.get_gpt_input()
        )
        sys_prompt = FIRST_RESPONSE_NONE_SYS_PROMPT.format(
            agent_description=await self.prepare_agent_description()
        )
        result = await self.llm.do_chat_w_sys_prompt(main_prompt, sys_prompt, max_tokens=50)
        return result

    @async_perf_logger
    async def generate_first_response_notification(self, chat_context: ChatContext) -> str:
        result = "To update my notification settings, please visit the Settings tab."
        return result

    @async_perf_logger
    async def generate_initial_preplan_response(self, chat_context: ChatContext) -> str:
        main_prompt = INITIAL_PREPLAN_MAIN_PROMPT.format(chat_context=chat_context.get_gpt_input())
        sys_prompt = INITIAL_PREPLAN_SYS_PROMPT.format(
            agent_description=await self.prepare_agent_description()
        )
        result = await self.llm.do_chat_w_sys_prompt(main_prompt, sys_prompt, max_tokens=70)
        return result

    async def generate_initial_midplan_response(self, chat_context: ChatContext) -> str:
        main_prompt = INITIAL_MIDPLAN_MAIN_PROMPT.format(chat_context=chat_context.get_gpt_input())
        sys_prompt = INITIAL_MIDPLAN_SYS_PROMPT.format(
            agent_description=await self.prepare_agent_description()
        )
        result = await self.llm.do_chat_w_sys_prompt(main_prompt, sys_prompt, max_tokens=50)
        return result

    async def generate_initial_postplan_response(
        self, chat_context: ChatContext, execution_plan: ExecutionPlan
    ) -> str:
        main_prompt = INITIAL_POSTPLAN_MAIN_PROMPT.format(
            chat_context=chat_context.get_gpt_input(), plan=execution_plan.get_plan_steps_for_gpt()
        )
        sys_prompt = INITIAL_POSTPLAN_SYS_PROMPT.format(
            agent_description=await self.prepare_agent_description()
        )
        result = await self.llm.do_chat_w_sys_prompt(main_prompt, sys_prompt, max_tokens=100)
        return result

    async def generate_initial_postplan_incomplete_response(
        self, chat_context: ChatContext, execution_plan: ExecutionPlan, missing_str: str
    ) -> str:
        main_prompt = INITIAL_POSTPLAN_INCOMPLETE_MAIN_PROMPT.format(
            chat_context=chat_context.get_gpt_input(),
            plan=execution_plan.get_plan_steps_for_gpt(),
            missing=missing_str,
        )
        sys_prompt = INITIAL_POSTPLAN_INCOMPLETE_SYS_PROMPT.format(
            agent_description=await self.prepare_agent_description()
        )
        result = await self.llm.do_chat_w_sys_prompt(main_prompt, sys_prompt, max_tokens=130)
        return result

    async def generate_initial_plan_failed_response(self, chat_context: ChatContext) -> str:
        main_prompt = INITIAL_PLAN_FAILED_MAIN_PROMPT.format(
            chat_context=chat_context.get_gpt_input()
        )
        sys_prompt = INITIAL_PLAN_FAILED_SYS_PROMPT.format(
            agent_description=await self.prepare_agent_description()
        )
        result = await self.llm.do_chat_w_sys_prompt(main_prompt, sys_prompt, max_tokens=50)
        return result

    async def generate_initial_plan_failed_response_suggestions(
        self, chat_context: ChatContext
    ) -> str:
        main_prompt = INITIAL_PLAN_FAILED_SUGGESTIONS_MAIN_PROMPT.format(
            chat_context=chat_context.get_gpt_input()
        )
        sys_prompt = INITIAL_PLAN_FAILED_SUGGESTIONS_SYS_PROMPT.format(
            agent_description=await self.prepare_agent_description(),
            tools=self.tool_registry.get_tool_str(),
        )
        result = await self.llm.do_chat_w_sys_prompt(main_prompt, sys_prompt, max_tokens=500)
        return result

    async def generate_input_update_no_action_response(self, chat_context: ChatContext) -> str:
        main_prompt = INPUT_UPDATE_NO_ACTION_MAIN_PROMPT.format(
            chat_context=chat_context.get_gpt_input()
        )
        sys_prompt = INPUT_UPDATE_NO_ACTION_SYS_PROMPT.format(
            agent_description=await self.prepare_agent_description()
        )
        result = await self.llm.do_chat_w_sys_prompt(main_prompt, sys_prompt, max_tokens=30)
        return result

    async def generate_input_update_rerun_response(
        self, chat_context: ChatContext, execution_plan: ExecutionPlan, functions: str
    ) -> str:
        main_prompt = INPUT_UPDATE_RERUN_MAIN_PROMPT.format(
            chat_context=chat_context.get_gpt_input(), plan=execution_plan, functions=functions
        )
        sys_prompt = INPUT_UPDATE_RERUN_SYS_PROMPT.format(
            agent_description=await self.prepare_agent_description()
        )
        result = await self.llm.do_chat_w_sys_prompt(main_prompt, sys_prompt, max_tokens=60)
        return result

    async def generate_input_update_replan_preplan_response(self, chat_context: ChatContext) -> str:
        main_prompt = INPUT_UPDATE_REPLAN_PREPLAN_MAIN_PROMPT.format(
            chat_context=chat_context.get_gpt_input()
        )
        sys_prompt = INPUT_UPDATE_REPLAN_PREPLAN_SYS_PROMPT.format(
            agent_description=await self.prepare_agent_description()
        )
        result = await self.llm.do_chat_w_sys_prompt(main_prompt, sys_prompt, max_tokens=80)
        return result

    async def generate_input_update_replan_postplan_response(
        self, chat_context: ChatContext, old_plan: ExecutionPlan, new_plan: ExecutionPlan
    ) -> str:
        main_prompt = INPUT_UPDATE_REPLAN_POSTPLAN_MAIN_PROMPT.format(
            chat_context=chat_context.get_gpt_input(),
            old_plan=old_plan.get_formatted_plan(),
            new_plan=new_plan.get_formatted_plan(),
        )
        sys_prompt = INPUT_UPDATE_REPLAN_POSTPLAN_SYS_PROMPT.format(
            agent_description=await self.prepare_agent_description()
        )
        result = await self.llm.do_chat_w_sys_prompt(main_prompt, sys_prompt, max_tokens=80)
        return result

    async def generate_error_replan_preplan_response(
        self, chat_context: ChatContext, last_plan: ExecutionPlan, error_info: ErrorInfo
    ) -> str:
        main_prompt = ERROR_REPLAN_PREPLAN_MAIN_PROMPT.format(
            chat_context=chat_context.get_gpt_input(),
            old_plan=last_plan.get_formatted_plan(),
            step=error_info.step.get_plan_step_str(),
            error=error_info.error,
            change=error_info.change,
        )
        sys_prompt = ERROR_REPLAN_PREPLAN_SYS_PROMPT.format(
            agent_description=await self.prepare_agent_description()
        )
        result = await self.llm.do_chat_w_sys_prompt(main_prompt, sys_prompt, max_tokens=120)
        return result

    async def generate_error_replan_postplan_response(
        self, chat_context: ChatContext, old_plan: ExecutionPlan, new_plan: ExecutionPlan
    ) -> str:
        main_prompt = ERROR_REPLAN_POSTPLAN_MAIN_PROMPT.format(
            chat_context=chat_context.get_gpt_input(),
            old_plan=old_plan.get_formatted_plan(),
            new_plan=new_plan.get_formatted_plan(),
        )
        sys_prompt = ERROR_REPLAN_POSTPLAN_SYS_PROMPT.format(
            agent_description=await self.prepare_agent_description()
        )
        result = await self.llm.do_chat_w_sys_prompt(main_prompt, sys_prompt, max_tokens=80)
        return result

    async def generate_notification_response(self, chat_context: ChatContext) -> str:
        main_prompt = NOTIFICATION_UPDATE_MAIN_PROMPT.format(
            chat_context=chat_context.get_gpt_input(),
        )
        sys_prompt = NOTIFICATION_UPDATE_SYS_PROMPT.format(
            agent_description=await self.prepare_agent_description()
        )
        result = await self.llm.do_chat_w_sys_prompt(main_prompt, sys_prompt, max_tokens=60)
        return result

    async def generate_non_retriable_error_response(
        self,
        chat_context: ChatContext,
        plan: ExecutionPlan,
        step: ToolExecutionNode,
        error: str,
    ) -> str:
        sys_prompt = NON_RETRIABLE_ERROR_SYS_PROMPT.format(
            chat_context=chat_context.get_gpt_input(),
            old_plan=plan.get_formatted_plan(),
            step=step.get_plan_step_str(),
            error=error,
        )
        main_prompt = NON_RETRIABLE_ERROR_MAIN_PROMPT.format(
            agent_description=await self.prepare_agent_description()
        )
        result = await self.llm.do_chat_w_sys_prompt(main_prompt, sys_prompt, max_tokens=120)
        return result

    async def prepare_agent_description(self) -> str:
        agent_description = AGENT_DESCRIPTION_PROMPT.format(
            today=str(date.today().strftime("%Y-%m-%d"))
        ).filled_prompt
        return agent_description

    def generate_execution_complete_response(
        self, existing_output_titles: dict, new_output_titles: dict, plan_run_id: str
    ) -> str:
        chat_anchor_links = [
            {
                "type": "output_widget",
                "name": title,
                "output_id": output_id,
                "plan_run_id": plan_run_id,
            }
            for title, output_id in new_output_titles.items()
            if title not in existing_output_titles
        ]
        chat_anchor_link_str = ", ".join(
            "```" + json.dumps(chat_anchor_link) + "```" for chat_anchor_link in chat_anchor_links
        )
        FOLLOW_UP_QUESTION = "Need anything else?"
        message = f"The report for {chat_anchor_link_str} is ready. {FOLLOW_UP_QUESTION}"
        return message
