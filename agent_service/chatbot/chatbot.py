from typing import Optional

from gpt_service_proto_v1.service_grpc import GPTServiceStub

from agent_service.chatbot.prompts import (
    AGENT_DESCRIPTION,
    COMPLETE_EXECUTION_MAIN_PROMPT,
    COMPLETE_EXECUTION_SYS_PROMPT,
    INITIAL_MIDPLAN_MAIN_PROMPT,
    INITIAL_MIDPLAN_SYS_PROMPT,
    INITIAL_PLAN_FAILED_MAIN_PROMPT,
    INITIAL_PLAN_FAILED_SYS_PROMPT,
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
)
from agent_service.GPT.constants import DEFAULT_SMART_MODEL
from agent_service.GPT.requests import GPT
from agent_service.io_type_utils import IOType
from agent_service.planner.planner_types import ExecutionPlan
from agent_service.types import ChatContext
from agent_service.utils.gpt_logging import GptJobIdType, GptJobType, create_gpt_context


class Chatbot:
    def __init__(
        self,
        agent_id: str,
        model: str = DEFAULT_SMART_MODEL,
        gpt_service_stub: Optional[GPTServiceStub] = None,
    ) -> None:
        self.agent_id = agent_id
        context = create_gpt_context(GptJobType.AGENT_CHATBOT, agent_id, GptJobIdType.AGENT_ID)
        self.llm = GPT(context, model, gpt_service_stub=gpt_service_stub)

    async def generate_initial_preplan_response(self, chat_context: ChatContext) -> str:
        main_prompt = INITIAL_PREPLAN_MAIN_PROMPT.format(chat_context=chat_context.get_gpt_input())
        sys_prompt = INITIAL_PREPLAN_SYS_PROMPT.format(agent_description=AGENT_DESCRIPTION)
        result = await self.llm.do_chat_w_sys_prompt(main_prompt, sys_prompt, max_tokens=50)
        return result

    async def generate_initial_midplan_response(self, chat_context: ChatContext) -> str:
        main_prompt = INITIAL_MIDPLAN_MAIN_PROMPT.format(chat_context=chat_context.get_gpt_input())
        sys_prompt = INITIAL_MIDPLAN_SYS_PROMPT.format(agent_description=AGENT_DESCRIPTION)
        result = await self.llm.do_chat_w_sys_prompt(main_prompt, sys_prompt, max_tokens=50)
        return result

    async def generate_initial_postplan_response(
        self, chat_context: ChatContext, execution_plan: ExecutionPlan
    ) -> str:
        main_prompt = INITIAL_POSTPLAN_MAIN_PROMPT.format(
            chat_context=chat_context.get_gpt_input(), plan=execution_plan.get_plan_steps_for_gpt()
        )
        sys_prompt = INITIAL_POSTPLAN_SYS_PROMPT.format(agent_description=AGENT_DESCRIPTION)
        result = await self.llm.do_chat_w_sys_prompt(main_prompt, sys_prompt, max_tokens=100)
        return result

    async def generate_initial_plan_failed_response(self, chat_context: ChatContext) -> str:
        main_prompt = INITIAL_PLAN_FAILED_MAIN_PROMPT.format(
            chat_context=chat_context.get_gpt_input()
        )
        sys_prompt = INITIAL_PLAN_FAILED_SYS_PROMPT.format(agent_description=AGENT_DESCRIPTION)
        result = await self.llm.do_chat_w_sys_prompt(main_prompt, sys_prompt, max_tokens=50)
        return result

    async def generate_execution_complete_response(
        self, chat_context: ChatContext, execution_plan: ExecutionPlan, output: IOType
    ) -> str:
        main_prompt = COMPLETE_EXECUTION_MAIN_PROMPT.format(
            chat_context=chat_context.get_gpt_input(),
            plan=execution_plan.get_plan_steps_for_gpt(),
            output=output,
        )
        sys_prompt = COMPLETE_EXECUTION_SYS_PROMPT.format(agent_description=AGENT_DESCRIPTION)
        result = await self.llm.do_chat_w_sys_prompt(main_prompt, sys_prompt, max_tokens=100)
        return result

    async def generate_input_update_no_action_response(self, chat_context: ChatContext) -> str:
        main_prompt = INPUT_UPDATE_NO_ACTION_MAIN_PROMPT.format(
            chat_context=chat_context.get_gpt_input()
        )
        sys_prompt = INPUT_UPDATE_NO_ACTION_SYS_PROMPT.format(agent_description=AGENT_DESCRIPTION)
        result = await self.llm.do_chat_w_sys_prompt(main_prompt, sys_prompt, max_tokens=30)
        return result

    async def generate_input_update_rerun_response(
        self, chat_context: ChatContext, execution_plan: ExecutionPlan, functions: str
    ) -> str:
        main_prompt = INPUT_UPDATE_RERUN_MAIN_PROMPT.format(
            chat_context=chat_context.get_gpt_input(), plan=execution_plan, functions=functions
        )
        sys_prompt = INPUT_UPDATE_RERUN_SYS_PROMPT.format(agent_description=AGENT_DESCRIPTION)
        result = await self.llm.do_chat_w_sys_prompt(main_prompt, sys_prompt, max_tokens=60)
        return result

    async def generate_input_update_replan_preplan_response(self, chat_context: ChatContext) -> str:
        main_prompt = INPUT_UPDATE_REPLAN_PREPLAN_MAIN_PROMPT.format(
            chat_context=chat_context.get_gpt_input()
        )
        sys_prompt = INPUT_UPDATE_REPLAN_PREPLAN_SYS_PROMPT.format(
            agent_description=AGENT_DESCRIPTION
        )
        result = await self.llm.do_chat_w_sys_prompt(main_prompt, sys_prompt, max_tokens=30)
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
            agent_description=AGENT_DESCRIPTION
        )
        result = await self.llm.do_chat_w_sys_prompt(main_prompt, sys_prompt, max_tokens=80)
        return result
