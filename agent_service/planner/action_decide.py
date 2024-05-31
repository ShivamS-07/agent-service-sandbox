import asyncio
from typing import List, Tuple, Type

from agent_service.GPT.constants import GPT4_O, GPT4_TURBO
from agent_service.GPT.requests import GPT
from agent_service.planner.constants import Action
from agent_service.planner.planner import Planner
from agent_service.planner.planner_types import ExecutionPlan, ToolExecutionNode
from agent_service.planner.prompts import (
    ACTION_DECIDER_MAIN_PROMPT,
    ACTION_DECIDER_SYS_PROMPT,
    ERROR_ACTION_DECIDER_MAIN_PROMPT,
    ERROR_ACTION_DECIDER_SYS_PROMPT,
)
from agent_service.tool import ToolRegistry
from agent_service.types import ChatContext, Message
from agent_service.utils.date_utils import get_now_utc
from agent_service.utils.gpt_logging import GptJobIdType, GptJobType, create_gpt_context


class InputActionDecider:
    def __init__(
        self,
        agent_id: str,
        model: str = GPT4_O,
        tool_registry: Type[ToolRegistry] = ToolRegistry,
    ) -> None:
        self.agent_id = agent_id
        context = create_gpt_context(GptJobType.AGENT_PLANNER, agent_id, GptJobIdType.AGENT_ID)
        self.llm = GPT(context, model)
        self.tool_registry = tool_registry

    async def decide_action(self, chat_context: ChatContext, current_plan: ExecutionPlan) -> Action:
        reads_chat_list = []
        for step in current_plan.nodes:
            if ToolRegistry.does_tool_read_chat(step.tool_name):
                reads_chat_list.append(step.tool_name)

        latest_message = chat_context.messages.pop()
        main_prompt = ACTION_DECIDER_MAIN_PROMPT.format(
            plan=current_plan.get_formatted_plan(),
            reads_chat_list=reads_chat_list,
            chat_context=chat_context.get_gpt_input(),
            message=latest_message.get_gpt_input(),
        )
        result = await self.llm.do_chat_w_sys_prompt(
            main_prompt, ACTION_DECIDER_SYS_PROMPT.format()
        )  # , max_tokens=3)
        chat_context.messages.append(latest_message)
        return Action[result.split()[-1].strip().upper()]


class ErrorActionDecider:
    def __init__(
        self,
        agent_id: str,
        model: str = GPT4_TURBO,  # Turbo seems to do a fair bit better on this
        tool_registry: Type[ToolRegistry] = ToolRegistry,
    ) -> None:
        self.agent_id = agent_id
        context = create_gpt_context(GptJobType.AGENT_PLANNER, agent_id, GptJobIdType.AGENT_ID)
        self.llm = GPT(context, model)
        self.tool_registry = tool_registry

    async def decide_action(
        self,
        error: Exception,
        failed_step: ToolExecutionNode,
        plans: List[ExecutionPlan],
        chat_context: ChatContext,
    ) -> Tuple[Action, str]:
        latest_plan = plans[-1]

        sys_prompt = ERROR_ACTION_DECIDER_SYS_PROMPT.format(tools=self.tool_registry.get_tool_str())

        main_prompt = ERROR_ACTION_DECIDER_MAIN_PROMPT.format(
            plan=latest_plan.get_formatted_plan(),
            failed_step=failed_step.get_plan_step_str(),
            error=error,
            chat_context=chat_context.get_gpt_input(),
            old_plans="\n***\n".join([old_plan.get_formatted_plan() for old_plan in plans[:-1]]),
        )

        result = await self.llm.do_chat_w_sys_prompt(main_prompt, sys_prompt)

        lines = result.split("\n")
        return Action[lines[-1].strip().upper()], lines[0]


async def main() -> None:
    input_text = "Can you give me a single summary of news published in the last week about machine learning at Meta, Apple, and Microsoft?"  # noqa: E501
    user_message = Message(message=input_text, is_user_message=True, message_time=get_now_utc())
    AI_response = "Okay, I'm doing that summary for you."
    AI_message = Message(message=AI_response, is_user_message=True, message_time=get_now_utc())
    chat_context = ChatContext(messages=[user_message, AI_message])
    planner = Planner("123")
    plan = await planner.create_initial_plan(chat_context)

    new_user_inputs = [
        "Thanks!",
        "Make sure the summary is no more than a paragraph",
        "I need you to include Amazon in the summary as well",
        "That's good, but I also need their current stock prices",
    ]

    action_decider = InputActionDecider("123")
    for new_input in new_user_inputs:
        new_message = Message(message=new_input, is_user_message=True, message_time=get_now_utc())
        print(new_message)
        chat_context.messages.append(new_message)
        result = await action_decider.decide_action(chat_context, plan)  # type: ignore
        print(result)
        chat_context.messages.pop()


if __name__ == "__main__":
    asyncio.run(main())
