import logging

from agent_service.GPT.constants import GPT4_O_MINI, Prompt
from agent_service.GPT.requests import GPT
from agent_service.tool import default_tool_registry
from agent_service.tools import *  # noqa
from agent_service.utils.async_db import AsyncDB
from agent_service.utils.logs import async_perf_logger
from agent_service.utils.postgres import SyncBoostedPG

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
        "Guidelines for providing a good answer:\n"
        "\n1. Use the debug information, if available, to explain the inner workings of the tool function.\n"
        "\n2. Refer to the tool description for context about the tool function and its purpose.\n"
        "\n3. Analyze the task input arguments to explain the inputs provided to the tool.\n"
        "\n4. Use the tool function's output to describe the results produced by the task.\n\n"
        "\n5. Ensure your response is concise, relevant, and addresses the user's question effectively. "
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
        "Now, answer the following question based on the provided information:\n"
        "{question}\n\n"
        "Write a clear, well-structured response that directly addresses the question."
    ),
)


@async_perf_logger
async def examine_task(plan_run_id: str, task_id: str, tool_name: str, question: str) -> str:
    # extract task details from db
    async_db = AsyncDB(pg=SyncBoostedPG(skip_commit=True))
    task_run_info = await async_db.get_task_run_info(
        plan_run_id=plan_run_id,
        task_id=task_id,
        tool_name=tool_name,
    )
    if task_run_info:
        _, _, task_debug_info, _ = task_run_info

    else:
        logger.info(f"Task run info not found for plan_run_id={plan_run_id}, task_id={task_id}")
        raise Exception("Task run info not found")

    # get task description from db
    tool = default_tool_registry().get_tool(tool_name)

    llm = GPT(model=GPT4_O_MINI)
    main_prompt = TASK_EXAMINE_MAIN_PROMPT.format(
        task_debug_info=task_debug_info,
        tool_desc=tool.description,
        question=question,
    )

    res = await llm.do_chat_w_sys_prompt(
        main_prompt=main_prompt,
        sys_prompt=TASK_EXAMINE_SYSTEM_PROMPT.format(),
    )
    return res


if __name__ == "__main__":
    import asyncio

    res = asyncio.run(
        examine_task(
            plan_run_id="1ea31ffb-20f7-498d-a55f-a51113f595eb",
            task_id="b14da138-200a-45ee-a6c9-dac6826be50b",
            tool_name="add_lists",
            question="How did the list been added?",
        )
    )
    print(res)
