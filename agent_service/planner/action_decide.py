import asyncio
from typing import List, Optional, Tuple
from uuid import uuid4

from agent_service.GPT.constants import GPT4_O, GPT4_TURBO, NO_PROMPT
from agent_service.GPT.requests import GPT
from agent_service.planner.constants import FirstAction, FollowupAction
from agent_service.planner.planner import Planner
from agent_service.planner.planner_types import ExecutionPlan, ToolExecutionNode
from agent_service.planner.prompts import (
    ERROR_ACTION_DECIDER_MAIN_PROMPT,
    ERROR_ACTION_DECIDER_SYS_PROMPT,
    ERROR_REPLAN_GUIDELINES,
    FIRST_ACTION_DECIDER_MAIN_PROMPT,
    FIRST_ACTION_DECIDER_SYS_PROMPT,
    FOLLOWUP_ACTION_DECIDER_MAIN_PROMPT,
    FOLLOWUP_ACTION_DECIDER_SYS_PROMPT,
    NOTIFICATION_CREATE_MAIN_PROMPT,
    NOTIFICATION_DEFAULT_MAIN_PROMPT,
    NOTIFICATION_EXAMPLE,
    NOTIFICATION_UPDATE_MAIN_PROMPT,
)
from agent_service.tool import ToolRegistry, default_tool_registry
from agent_service.types import ChatContext, Message
from agent_service.utils.async_db import AsyncDB
from agent_service.utils.date_utils import get_now_utc
from agent_service.utils.gpt_logging import GptJobIdType, GptJobType, create_gpt_context
from agent_service.utils.logs import async_perf_logger
from agent_service.utils.postgres import SyncBoostedPG


class FirstActionDecider:
    def __init__(
        self,
        agent_id: str,
        skip_db_commit: bool = False,
        model: str = GPT4_O,
        tool_registry: Optional[ToolRegistry] = None,
    ) -> None:
        self.agent_id = agent_id
        context = create_gpt_context(GptJobType.AGENT_PLANNER, agent_id, GptJobIdType.AGENT_ID)
        self.llm = GPT(context, model)
        self.tool_registry = tool_registry or default_tool_registry()
        self.db = AsyncDB(pg=SyncBoostedPG(skip_commit=skip_db_commit))

    @async_perf_logger
    async def decide_action(self, chat_context: ChatContext) -> FirstAction:
        latest_message = chat_context.messages.pop()
        main_prompt = FIRST_ACTION_DECIDER_MAIN_PROMPT.format(
            chat_context=chat_context.get_gpt_input(), message=latest_message.get_gpt_input()
        )
        chat_context.messages.append(latest_message)
        result = await self.llm.do_chat_w_sys_prompt(
            main_prompt, FIRST_ACTION_DECIDER_SYS_PROMPT.format()
        )
        action = FirstAction[result.split()[-1].strip("`").strip().upper()]
        return action


class FollowupActionDecider:
    def __init__(
        self,
        agent_id: str,
        skip_db_commit: bool = False,
        model: str = GPT4_O,
        tool_registry: Optional[ToolRegistry] = None,
    ) -> None:
        self.agent_id = agent_id
        context = create_gpt_context(GptJobType.AGENT_PLANNER, agent_id, GptJobIdType.AGENT_ID)
        self.llm = GPT(context, model)
        self.tool_registry = tool_registry or default_tool_registry()
        self.db = AsyncDB(pg=SyncBoostedPG(skip_commit=skip_db_commit))

    async def decide_action(
        self, chat_context: ChatContext, current_plan: ExecutionPlan
    ) -> FollowupAction:
        reads_chat_list = []
        instruction_list = []
        for step in current_plan.nodes:
            tool = self.tool_registry.get_tool(step.tool_name)
            if tool.reads_chat:
                reads_chat_list.append(step.tool_name)
            if tool.update_instructions:
                instruction_list.append(f"{step.tool_name}: {tool.update_instructions}")

        chat_context.sort_messages()
        latest_message = chat_context.messages.pop()
        main_prompt = FOLLOWUP_ACTION_DECIDER_MAIN_PROMPT.format(
            plan=current_plan.get_formatted_plan(),
            reads_chat_list=reads_chat_list,
            decision_instructions="\n".join(instruction_list),
            chat_context=chat_context.get_gpt_input(),
            message=latest_message.get_gpt_input(),
        )
        result = await self.llm.do_chat_w_sys_prompt(
            main_prompt, FOLLOWUP_ACTION_DECIDER_SYS_PROMPT.format()
        )
        chat_context.messages.append(latest_message)
        action = FollowupAction[result.split()[-1].strip("`").strip().upper()]
        if len(reads_chat_list) == 0 and action == FollowupAction.RERUN:  # GPT shouldn't do this
            action = FollowupAction.REPLAN
        automation_enabled = await self.db.get_agent_automation_enabled(agent_id=self.agent_id)
        if automation_enabled and action in [FollowupAction.RERUN, FollowupAction.REPLAN]:
            return FollowupAction.APPEND
        return action

    async def create_custom_notifications(self, chat_context: ChatContext) -> str:
        main_prompt = NOTIFICATION_CREATE_MAIN_PROMPT.format(
            chat_context=chat_context.get_gpt_input(), example=NOTIFICATION_EXAMPLE
        )
        result = (await self.llm.do_chat_w_sys_prompt(main_prompt, NO_PROMPT)).replace("\n\n", "\n")
        return result

    async def update_custom_notifications(
        self, chat_context: ChatContext, current_notifications: str
    ) -> str:
        main_prompt = NOTIFICATION_UPDATE_MAIN_PROMPT.format(
            chat_context=chat_context.get_gpt_input(),
            current_notifications=current_notifications,
            example=NOTIFICATION_EXAMPLE,
        )

        result = (await self.llm.do_chat_w_sys_prompt(main_prompt, NO_PROMPT)).replace("\n\n", "\n")
        return result

    async def generate_default_custom_notifications(self, chat_context: ChatContext) -> str:
        main_prompt = NOTIFICATION_DEFAULT_MAIN_PROMPT.format(
            chat_context=chat_context.get_gpt_input()
        )
        result = (await self.llm.do_chat_w_sys_prompt(main_prompt, NO_PROMPT)).replace("\n\n", "\n")
        return result


class ErrorActionDecider:
    def __init__(
        self,
        agent_id: str,
        model: str = GPT4_TURBO,  # Turbo seems to do a fair bit better on this
        tool_registry: Optional[ToolRegistry] = None,
    ) -> None:
        self.agent_id = agent_id
        context = create_gpt_context(GptJobType.AGENT_PLANNER, agent_id, GptJobIdType.AGENT_ID)
        self.llm = GPT(context, model)
        self.tool_registry = tool_registry or default_tool_registry()

    async def decide_action(
        self,
        error: Exception,
        failed_step: ToolExecutionNode,
        plans: List[ExecutionPlan],
        chat_context: ChatContext,
    ) -> Tuple[FollowupAction, str]:
        latest_plan = plans[-1]

        sys_prompt = ERROR_ACTION_DECIDER_SYS_PROMPT.format(
            tools=self.tool_registry.get_tool_str(), replan_guidelines=ERROR_REPLAN_GUIDELINES
        )

        main_prompt = ERROR_ACTION_DECIDER_MAIN_PROMPT.format(
            plan=latest_plan.get_formatted_plan(),
            failed_step=failed_step.get_plan_step_str(),
            error=error,
            chat_context=chat_context.get_gpt_input(),
            old_plans="\n***\n".join([old_plan.get_formatted_plan() for old_plan in plans[:-1]]),
        )

        result = await self.llm.do_chat_w_sys_prompt(main_prompt, sys_prompt)

        lines = result.split("\n")
        return FollowupAction[lines[-1].strip().upper()], lines[0]


async def main() -> None:
    input_text = "Can you give me a single summary of news published in the last week about machine learning at Meta, Apple, and Microsoft?"  # noqa: E501
    user_message = Message(message=input_text, is_user_message=True, message_time=get_now_utc())
    AI_response = "Okay, I'm doing that summary for you."
    AI_message = Message(message=AI_response, is_user_message=True, message_time=get_now_utc())
    chat_context = ChatContext(messages=[user_message, AI_message])
    planner = Planner()
    plan = await planner.create_initial_plan(chat_context, plan_id=str(uuid4()))

    new_user_inputs = [
        "hey can you alert me when sell-side 2025 earnings estimates for AIG change?"
        "create another widget for LULU",
        "No wait! just do it for TSLA",
        "Thanks!",
        "Make sure the summary is no more than a paragraph",
        "I need you to include Amazon in the summary as well",
        "That's good, but I also need their current stock prices",
        "Move the text down below the graph please",
        "Let me know when new data is avaialble every week",
    ]

    followup_action_decider = FollowupActionDecider("71e3c9dd-2dc5-42c2-bc99-de2f045628d2")
    first_action_decider = FirstActionDecider("aa0f8b5e-ef77-4c67-acdb-e14d3689e7e2")
    for new_input in new_user_inputs:
        new_message = Message(message=new_input, is_user_message=True, message_time=get_now_utc())
        print(new_message)
        chat_context.messages.append(new_message)
        result = await followup_action_decider.decide_action(chat_context, plan)  # type: ignore
        print("Followup action:", result)
        result = await first_action_decider.decide_action(chat_context)  # type: ignore
        print("First action:", result)
        print("=" * 100)
        chat_context.messages.pop()


if __name__ == "__main__":
    asyncio.run(main())
