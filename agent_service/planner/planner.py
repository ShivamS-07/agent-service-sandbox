from typing import Dict, List, Type

from agent_service.GPT.constants import DEFAULT_SMART_MODEL
from agent_service.GPT.requests import GPT
from agent_service.planner.constants import ARGUMENT_RE, ASSIGNMENT_RE
from agent_service.planner.planner_types import ParsedStep
from agent_service.planner.prompts import (
    PLAN_EXAMPLE,
    PLAN_GUIDELINES,
    PLAN_RULES,
    PLANNER_MAIN_PROMPT,
    PLANNER_SYS_PROMPT,
)
from agent_service.tools.tool import ToolRegistry
from agent_service.types import ChatContext, ExecutionPlan
from agent_service.utils.gpt_logging import GptJobIdType, GptJobType, create_gpt_context


def get_arg_dict(arg_str: str) -> Dict[str, str]:
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
        model: str = DEFAULT_SMART_MODEL,
        tool_registry: Type[ToolRegistry] = ToolRegistry,
    ) -> None:
        self.agent_id = agent_id
        context = create_gpt_context(GptJobType.AGENT_PLANNER, agent_id, GptJobIdType.AGENT_ID)
        self.llm = GPT(context, model)
        self.tool_string = tool_registry.get_tool_str()

    async def create_initial_plan(self, chat_context: ChatContext) -> ExecutionPlan:

        # TODO: put this in a loop where if the parse fails, we query GPT with
        # two additional messages: the GPT response and the parse exception

        plan_str = await self._query_GPT_for_initial_plan(chat_context.get_gpt_input())

        plan = self._parse_plan_str(plan_str)

        return plan

    async def _query_GPT_for_initial_plan(self, user_input: str) -> str:
        sys_prompt = PLANNER_SYS_PROMPT.format(
            rules=PLAN_RULES,
            guidelines=PLAN_GUIDELINES,
            example=PLAN_EXAMPLE,
            tools=self.tool_string,
        )
        main_prompt = PLANNER_MAIN_PROMPT.format(message=user_input)
        return await self.llm.do_chat_w_sys_prompt(main_prompt, sys_prompt)

    def _parse_plan_str(self, plan_str: str) -> ExecutionPlan:
        plan_steps: List[ParsedStep] = []
        for line in plan_str.split("\n"):
            if line.startswith("```"):  # GPT likes to add this to Python code
                continue

            match = ASSIGNMENT_RE.match(line)

            if match is None:
                raise Exception(f"Failed to parse plan line: {line}")

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

        print(plan_steps)

        # TODO: fill in the rest
        return ExecutionPlan()
