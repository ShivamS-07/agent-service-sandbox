import logging
from typing import Optional, cast

from llm_client.datamodels import DoChatArgs, LLMFunction, LLMResponse
from llm_client.llm_client import LLMClient

from agent_service.GPT.constants import GPT4_O
from agent_service.io_type_utils import load_io_type
from agent_service.q_and_a.examine_plan import EXAMINE_PLAN_FUNC
from agent_service.q_and_a.examine_task import EXAMINE_TASK_FUNC
from agent_service.q_and_a.general_knowledge import ASK_GENERAL_QUESTION_FUNC
from agent_service.q_and_a.utils import QAContext
from agent_service.types import ChatContext
from agent_service.utils.async_db import get_async_db
from agent_service.utils.gpt_logging import q_and_a_context
from agent_service.utils.logs import async_perf_logger
from agent_service.utils.output_utils.utils import io_type_to_gpt_input
from agent_service.utils.prompt_utils import Prompt

Q_AND_A_SYS_PROMPT = Prompt(
    name="Q_AND_A_SYS_PROMPT",
    template=(
        "You are a financial analyst who writes python-based workflows for your client, "
        "and communicate with them via chat. Your goal here is to provide a clear and accurate answer to a query. "
        "Here is some guidance to help you craft your response:\n"
        "\n- Your answer must be concise, relevant, and directly address the user's query. "
        "It should be less than 150 words unless the user asks for more details.\n"
        "\n- You will be provided with a list of tools that can use to gather more information. "
    ),
)
Q_AND_A_MAIN_PROMPT = Prompt(
    name="Q_AND_A_PROMPT",
    template=(
        "You are a financial analyst who writes python-based workflows for your client, "
        "and communicate with them via chat. Your client just gave you the following query: "
        "{query}"
        "The full context of the chat between you and your client is: "
        "\n{chat_context}\n"
        "The workflow plan that your are running is: "
        "\n{plan}\n"
        "The workflow logs for the executed plan is:\n"
        "\n{logs}\n"
        "The final report of the executed workflow is:\n"
        "\n{report}\n"
        "Based on the query and the full context, either gather more data with one of the "
        "provided tools or simply give a quick response if a tool isn't needed. "
    ),
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
        async_db = get_async_db()

        # get the plan
        _, plan = await async_db.get_execution_plan_for_run(about_plan_run_id)

        # get the logs
        work_logs = await async_db.get_agent_worklogs(
            agent_id=self.agent_id, plan_run_ids=[about_plan_run_id]
        )
        logs = [f'- {log["log_message"]}' for log in reversed(work_logs)]

        # get the reports outputs
        outputs_raw = await async_db.get_agent_outputs_data_from_db(
            agent_id=self.agent_id, include_output=True, plan_run_id=about_plan_run_id
        )
        io_outputs = [
            await io_type_to_gpt_input(load_io_type(row["output"]), use_abbreviated_output=False)
            if row["output"]
            else row["output"]
            for row in outputs_raw
        ]
        # trauncate each output to 10000 characters
        io_outputs = [
            f"\n# Output {i+1}: " + " ".join(output.split()[:10000])
            for i, output in enumerate(io_outputs)
        ]
        report = "\n# Output".join(io_outputs)

        main_prompt_str = Q_AND_A_MAIN_PROMPT.format(
            query=query,
            chat_context=self.chat_context.get_gpt_input(),
            plan=plan.get_formatted_plan(numbered=True, include_task_ids=True),
            logs="\n".join(logs),
            report=report,
        ).filled_prompt

        # # save main prompt as txt file
        # with open("main_prompt.txt", "w") as f:
        #     f.write(main_prompt_str)

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


if __name__ == "__main__":
    import asyncio

    async def main() -> None:
        qa_agent = QAAgent(
            agent_id="3d766275-b00c-4e56-80d8-3c3746547c23",
            chat_context=ChatContext(),
            user_id="f495ac3c-1bc2-4780-8e6d-d9f2c58164cd",
        )

        query = "what is Hang Seng Index?"
        about_plan_run_id = "3719cf35-5cad-4a6c-9527-04e7be7991ff"
        print(await qa_agent.query(query, about_plan_run_id))

    asyncio.run(main())
