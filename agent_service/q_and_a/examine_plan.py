import logging

from llm_client.datamodels import LLMFunction
from llm_client.functions.llm_func import LLMFuncArgs

from agent_service.GPT.constants import GPT4_O_MINI, Prompt
from agent_service.GPT.requests import GPT
from agent_service.io_types.text import Text
from agent_service.q_and_a.utils import QAContext
from agent_service.tool import Tool, default_tool_registry
from agent_service.tools import *  # noqa
from agent_service.utils.async_db import get_async_db
from agent_service.utils.logs import async_perf_logger

logger = logging.getLogger(__name__)


PLAN_EXAMINE_SYS_PROMPT = Prompt(
    name="PLAN_EXAMINE_SYS_PROMPT",
    template=(
        "You are tasked with answering a question related to a plan designed to address a client's request. "
        "The plan consists of a sequence of tasks executed using various tools, "
        "each with specific functions and arguments. "
        "You will be provided with detailed information about the plan, including the tools used and their purposes, "
        "as well as the latest changes happened in the most recent output of the plan which is called 'What's New'. "
        "Your goal is to deliver a concise and accurate response to the question "
        "based solely on the provided information. "
        "Avoid mentioning exact function or tool names in your response. "
        "Ensure your reply is clear, relevant, and less than 100 words."
    ),
)
PLAN_EXAMINE_MAIN_PROMPT = Prompt(
    name="PLAN_EXAMINE_MAIN_PROMPT",
    template=(
        "You are provided with the details of a plan used to address a client's request, "
        "as well as the most recent changes happened in the output of the latest run. "
        "The following information is available to help you answer a question:"
        "\n### Plan Details ###\n"
        "{plan}"
        "\n### Tool Descriptions ###\n"
        "{tool_descs}"
        "\n### What's New ###\n"
        "{plan_whats_new}"
        "\nBased on the provided information, answer the following question:\n"
        "{question}"
        "\nWrite a clear, well-structured response that directly addresses the question."
    ),
)


class ExaminePlanArgs(LLMFuncArgs):
    question: str


@async_perf_logger
async def examine_plan(args: ExaminePlanArgs, context: QAContext) -> str:
    async_db = get_async_db()
    _, plan = await async_db.get_execution_plan_for_run(context.plan_run_id)
    plan_run_metadata = await async_db.get_plan_run_metadata(context.plan_run_id)

    plan_str = plan.get_formatted_plan(numbered=True)
    run_summary_long = (
        plan_run_metadata.run_summary_long.val
        if isinstance(plan_run_metadata.run_summary_long, Text)
        else plan_run_metadata.run_summary_long
    )
    run_summary_short = (
        plan_run_metadata.run_summary_short if plan_run_metadata.run_summary_short else ""
    )
    plan_whats_new = f"**Summary:** {run_summary_short}\n\n**Details:** {run_summary_long}"

    # get tool descriptions for each tool in the plan
    tool_descs = []
    for node in plan.nodes:
        tool_name = node.tool_name
        tool: Tool = default_tool_registry().get_tool(tool_name)
        tool_descs.append(f"Description for {tool_name}: {tool.description}")

    llm = GPT(model=GPT4_O_MINI, context=context.gpt_context)
    main_prompt = PLAN_EXAMINE_MAIN_PROMPT.format(
        plan=plan_str,
        tool_descs="\n\n".join(tool_descs),
        plan_whats_new=plan_whats_new,
        question=args.question,
    )

    res = await llm.do_chat_w_sys_prompt(
        main_prompt=main_prompt,
        sys_prompt=PLAN_EXAMINE_SYS_PROMPT.format(),
    )
    return res


EXAMINE_PLAN_FUNC = LLMFunction(
    name="examine_entire_workflow",
    args=ExaminePlanArgs,
    func=examine_plan,
    description="""
Examine a workflow as a whole, e.g. to explain multiple pieces of the workflow,
where an element might have appeared or been removed, etc. For example, if the
user asks about the overall stratgy to solve a problem or to explain why a stock
is missing from the output, use this tool. In general, don't use this tool if
the user asks a question by only looking at a specific task in the
workflow. Priority should go to looking at the specific task that is relevant.
    """,
)
