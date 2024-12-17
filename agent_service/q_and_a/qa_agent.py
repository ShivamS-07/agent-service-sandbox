import asyncio
import logging
from typing import Optional

from llm_client.datamodels import DoChatArgs, LLMFunction
from llm_client.llm_client import LLMClient

from agent_service.external.gemini_client import GeminiClient
from agent_service.GPT.constants import GPT4_O
from agent_service.io_type_utils import IOType, load_io_type
from agent_service.q_and_a.examine_plan import EXAMINE_PLAN_FUNC
from agent_service.q_and_a.examine_task import (
    EXAMINE_TASK_FUNC,
    ExamineTaskArgs,
    get_all_task_related_info,
)
from agent_service.q_and_a.general_knowledge import (
    ASK_GENERAL_QUESTION_FUNC,
)
from agent_service.q_and_a.utils import QAContext
from agent_service.types import ChatContext
from agent_service.utils.async_db import AsyncDB, get_async_db
from agent_service.utils.async_utils import gather_with_concurrency
from agent_service.utils.gpt_logging import q_and_a_context
from agent_service.utils.logs import async_perf_logger
from agent_service.utils.output_utils.output_construction import get_output_from_io_type
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
        "\n- Your answer must only directly answer the user's question, "
        "try not to give too much supplemental info they weren't asking for. Concise is good."
        "\n- Note that some inputs and outputs for the tasks might be truncated to fit in the prompt. "
    ),
)
Q_AND_A_MAIN_PROMPT = Prompt(
    name="Q_AND_A_PROMPT",
    template=(
        "You are a financial analyst who writes python-based workflows for your client, "
        "and communicate with them via chat. Your client just gave you the following query: "
        "{query}"
        "The full context of the chat between you and your client is: "
        "\n{chat_context}\n------\n"
        "The workflow plan that your are running is: "
        "\n{plan}\n------\n"
        "The info related to each task in the plan are:\n"
        "\n{tasks_info}\n------\n"
        "The workflow logs for the executed plan is:\n"
        "\n{logs}\n------\n"
        "The final report of the executed workflow is here. Note that there may be a lot of "
        "citations that aren't relevant to the user's question, you should ignore those. Report:\n"
        "\n{report}\n------\n"
        "You also just ran a supplemental google serach with this query, and got the following "
        "result. Note that this info may or may not be relevant, just ignore it "
        "if you can answer the question with the above info. Google result:\n"
        "\n{grounding_result}\n"
        "\nBased on the query and the full context, provide the user with an answer. "
        "It is very important that you ANSWER THIS QUESTION IN THE CONTEXT OF THE WORKFLOW. "
        "The output sections often have titles which will give you an idea of the overall theme "
        "of the workflow, you should use this to decide what is relevant. "
        "DO NOT DISCUSS THINGS THAT ARE NOT RELEVANT TO THE REPORT IN QUESTION. "
        "If you discuss irrelevant things, you will be fired."
        "Don't give ANY preamble, simply and directly answer the user's question. "
        "Do not give any introduction and no need to say e.g. 'based on...'."
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

    async def _prep_report_string(self, outputs: list[IOType], db: AsyncDB) -> str:
        resolved_output_gather = gather_with_concurrency(
            [get_output_from_io_type(val=out, pg=db.pg) for out in outputs], n=10
        )
        gpt_input_gather = gather_with_concurrency(
            [
                io_type_to_gpt_input(io_type=out, use_abbreviated_output=False, truncate_to=10000)
                for out in outputs
            ],
            n=10,
        )
        resolved_outputs, gpt_inputs = await asyncio.gather(
            resolved_output_gather, gpt_input_gather
        )
        report_sections = []
        for gpt_input, resolved_output in zip(gpt_inputs, resolved_outputs):
            citations = "\n".join(
                [
                    f"  - {cit.name} (Citation Type: {cit.citation_type}): {cit.summary}"
                    for cit in resolved_output.citations
                ]
            )

            if resolved_output.title:
                section = f"SECTION TITLE: {resolved_output.title}\n\n{gpt_input}\n\nCITATIONS:\n{citations}"  # type: ignore
            else:
                section = f"{gpt_input}\n\nCITATIONS:\n{citations}"
            report_sections.append(section)

        return "------\n\n\n\n".join(report_sections)

    @async_perf_logger
    async def query(self, query: str, about_plan_run_id: Optional[str] = None) -> str:
        async_db = get_async_db()

        if not about_plan_run_id:
            latest_run, _ = await async_db.get_agent_plan_runs(agent_id=self.agent_id, limit_num=1)
            about_plan_run_id = latest_run[0][0]

        gemini_client = GeminiClient(context=self.gpt_context)
        grounding_query = f"""
        Please look for info on this question: {query}
        (Never return info on stock prices.)
        The chat context of the question is:
        {self.chat_context.get_gpt_input()}
        """

        tasks = [
            async_db.get_execution_plan_for_run(about_plan_run_id),
            async_db.get_agent_worklogs(agent_id=self.agent_id, plan_run_ids=[about_plan_run_id]),
            async_db.get_agent_outputs_data_from_db(
                agent_id=self.agent_id, include_output=True, plan_run_id=about_plan_run_id
            ),
            gemini_client.query_google_grounding(query=grounding_query, db=async_db.pg),
        ]
        # get the plan
        (_, plan), work_logs, outputs_raw, grounding_result = await asyncio.gather(*tasks)  # type: ignore

        logs = [f'- {log["log_message"]}' for log in reversed(work_logs) if log]  # type: ignore
        # get the outputs and prepare the report
        io_outputs = [
            load_io_type(row["output"]) if row["output"] else row["output"]
            for row in outputs_raw  # type: ignore
        ]
        report = await self._prep_report_string(io_outputs, db=async_db)  # type: ignore
        # get all the tasks related info for all the tasks in the plan
        tasks = [
            get_all_task_related_info(
                ExamineTaskArgs(
                    task_id=task.tool_task_id,  # type: ignore
                    tool_name=task.tool_name,  # type: ignore
                    query=query,
                ),
                context=QAContext(
                    agent_id=self.agent_id,
                    plan_run_id=about_plan_run_id,
                    user_id=self.user_id,
                    chat_context=self.chat_context,
                ),
            )
            for task in plan.nodes  # type: ignore
        ]
        tasks_info = await asyncio.gather(*tasks)

        main_prompt_str = Q_AND_A_MAIN_PROMPT.format(
            query=query,
            chat_context=self.chat_context.get_gpt_input(),
            plan=plan.get_formatted_plan(numbered=True, include_task_ids=True),  # type: ignore
            tasks_info="\n".join(tasks_info),
            logs="\n".join(logs),
            report=report,
            grounding_result=grounding_result,
        ).filled_prompt
        # save main prompt as txt
        # with open("main_prompt.txt", "w") as f:
        #     f.write(main_prompt_str)

        result = await self.llm_client.do_chat(
            DoChatArgs(
                model_id=GPT4_O,
                main_prompt=main_prompt_str,
                sys_prompt="",
                context=self.gpt_context,
                # tools=self.LLM_FUNCS,
            )
        )
        # if isinstance(result, LLMResponse):
        #     if not result.tool_call:
        #         return cast(str, result.result)

        #     func_to_call = self.LLM_FUNCS_MAP[result.tool_call.name]
        #     args = result.tool_call.args
        #     logger.info(f"Calling '{func_to_call.name}' with {args=}")
        #     # TODO make the type system work for us more here
        #     return await func_to_call.func(args, context)

        # TODO
        if isinstance(result, str):
            return result
        return str(result)


if __name__ == "__main__":

    async def main() -> None:
        qa_agent = QAAgent(
            agent_id="61037b16-10da-4520-97b4-4039914db895",
            chat_context=ChatContext(),
            user_id="3b997275-dcfe-4c19-8bb2-3e1366c4d5f3",
        )

        query = "why XOM is selected in the filtering step?"
        about_plan_run_id = "52b2c0fe-dec0-45b2-ad95-0ba84367fd19"
        print(await qa_agent.query(query, about_plan_run_id))

    asyncio.run(main())
