import asyncio
import logging

from llm_client.datamodels import LLMFunction
from llm_client.functions.llm_func import LLMFuncArgs

from agent_service.GPT.constants import GPT4_O_MINI, Prompt
from agent_service.GPT.requests import GPT
from agent_service.q_and_a.utils import QAContext
from agent_service.tool import default_tool_registry
from agent_service.tools import *  # noqa
from agent_service.utils.async_db import get_async_db
from agent_service.utils.logs import async_perf_logger

logger = logging.getLogger(__name__)


TASK_EXAMINE_SYSTEM_PROMPT = Prompt(
    name="TASK_EXAMINE_SYSTEM_PROMPT",
    template=(
        "Your goal is to provide a clear and accurate answer to a question about performing a specific task. "
        "The task involves a tool function that has been executed as part of a plan to address a user's request. "
        "You will be given the following information to assist in crafting your response:\n"
        "- Task input arguments\n"
        "- Output of the tool function\n"
        "- Debug information (if available)\n"
        "- Tool description\n\n"
        "- Logs (if available)"
        "Guidelines for providing a good answer:\n"
        "\n1. Use the debug information, if available, to explain the inner workings of the tool function.\n"
        "\n2. Refer to the tool description for context about the tool function and its purpose.\n"
        "\n3. Analyze the task input arguments to explain the inputs provided to the tool.\n"
        "\n4. Use the tool function's output to describe the results produced by the task.\n\n"
        "\n5. Ensure your response is concise, relevant, and addresses the user's question effectively. "
        "\n6. Never mention the word 'tool', 'function', etc. the user is not technical."
        "The answer should be in one paragraph and should not exceed 200 words unless user asks for more details."
    ),
)

TASK_EXAMINE_MAIN_PROMPT = Prompt(
    name="TASK_EXAMINE_MAIN_PROMPT",
    template=(
        "You have been asked a question about the details of a task performed using a tool function. "
        "This task was executed as part of a plan to address a user's request.\n\n"
        "Below is the relevant information to help you prepare your response:\n"
        "### Task Input Arguments ###\n"
        "{task_debug_info}\n\n"
        "### Tool Description ###\n"
        "{tool_desc}\n\n"
        "### Logs ###\n"
        "{logs}\n\n"
        "Now, answer the following question based on the provided information:\n"
        "{question}\n\n"
        "Write a clear, well-structured response that directly addresses the question."
    ),
)


class ExamineTaskArgs(LLMFuncArgs):
    task_id: str
    tool_name: str
    question: str


@async_perf_logger
async def examine_task(args: ExamineTaskArgs, context: QAContext) -> str:
    task_id = args.task_id
    tool_name = args.tool_name
    plan_run_id = context.plan_run_id
    # extract task details from db
    async_db = get_async_db()
    task_run_info, work_logs = await asyncio.gather(
        async_db.get_task_run_info(
            plan_run_id=plan_run_id,
            task_id=task_id,
            tool_name=tool_name,
        ),
        async_db.get_agent_worklogs(agent_id=context.agent_id, plan_run_ids=[context.plan_run_id]),
    )
    if task_run_info:
        _, _, task_debug_info, _ = task_run_info

    else:
        logger.info(f"Task run info not found for plan_run_id={plan_run_id}, task_id={task_id}")
        raise Exception("Task run info not found")

    # get task description from db
    tool = default_tool_registry().get_tool(tool_name)

    logs = [
        f'- {log["log_message"]}'
        for log in reversed(work_logs)
        if log["task_id"] == args.task_id and not log.get("is_task_output")
    ]
    llm = GPT(model=GPT4_O_MINI, context=context.gpt_context)
    main_prompt = TASK_EXAMINE_MAIN_PROMPT.format(
        task_debug_info=task_debug_info,
        tool_desc=tool.description,
        question=args.question,
        logs="\n".join(logs),
    )

    res = await llm.do_chat_w_sys_prompt(
        main_prompt=main_prompt,
        sys_prompt=TASK_EXAMINE_SYSTEM_PROMPT.format(),
    )
    return res


EXAMINE_TASK_FUNC = LLMFunction(
    name="examine_workflow_task",
    args=ExamineTaskArgs,
    func=examine_task,
    description="""
Examine a specific function in a the workflow, for example to see what it does
or give details about how it works and what it returned. For example, if the
user asks about a specific stock or other lookup task or how a specific
filtering task was done, use this tool. task_id should be the task ID of the
task that you are looking at.
    """,
)
