import logging
from typing import Optional, cast

from llm_client.datamodels import DoChatArgs, LLMFunction, LLMResponse
from llm_client.llm_client import LLMClient

from agent_service.GPT.constants import GPT4_O
from agent_service.q_and_a.examine_plan import EXAMINE_PLAN_FUNC
from agent_service.q_and_a.examine_task import EXAMINE_TASK_FUNC
from agent_service.q_and_a.general_knowledge import ASK_GENERAL_QUESTION_FUNC
from agent_service.q_and_a.utils import QAContext
from agent_service.types import ChatContext
from agent_service.utils.gpt_logging import q_and_a_context
from agent_service.utils.logs import async_perf_logger
from agent_service.utils.prompt_utils import Prompt

Q_AND_A_PROMPT = Prompt(
    name="Q_AND_A_PROMPT",
    template="""
You are a financial analyst who writes python-based workflows for your client,
and communicate with them via chat. Your client just gave you the following
query:
{query}
The full context of the chat between you and your client is:
{chat_context}

Based on the query and the full context, either gather more data with one of the
provided tools or simply give a quick response if a tool isn't needed.
""",
)

logger = logging.getLogger(__name__)


class QAAgent:
    def __init__(
        self,
        agent_id: str,
        chat_context: ChatContext,
        user_id: Optional[str],
    ):
        self.LLM_FUNCS = [EXAMINE_TASK_FUNC, ASK_GENERAL_QUESTION_FUNC, EXAMINE_PLAN_FUNC]
        self.LLM_FUNCS_MAP: dict[str, LLMFunction] = {func.name: func for func in self.LLM_FUNCS}
        self.llm_client = LLMClient()
        self.chat_context = chat_context
        self.gpt_context = q_and_a_context(agent_id=agent_id, user_id=user_id)
        self.user_id = user_id
        self.agent_id = agent_id

    @async_perf_logger
    async def query(self, query: str, about_plan_run_id: str) -> str:
        main_prompt_str = Q_AND_A_PROMPT.format(
            query=query, chat_context=self.chat_context.get_gpt_input()
        ).filled_prompt
        result = await self.llm_client.do_chat(
            DoChatArgs(
                model_id=GPT4_O,
                main_prompt=main_prompt_str,
                sys_prompt="",
                context=self.gpt_context,
                tools=self.LLM_FUNCS,
            )
        )
        if isinstance(result, LLMResponse):
            if not result.tool_call:
                return cast(str, result.result)

            context = QAContext(
                agent_id=self.agent_id,
                plan_run_id=about_plan_run_id,
                user_id=self.user_id,
                chat_context=self.chat_context,
            )
            func_to_call = self.LLM_FUNCS_MAP[result.tool_call.name]
            args = result.tool_call.args
            logger.info(f"Calling '{func_to_call.name}' with {args=}")
            # TODO make the type system work for us more here
            return await func_to_call.func(args, context)

        # TODO
        if isinstance(result, str):
            return result
        return str(result)
