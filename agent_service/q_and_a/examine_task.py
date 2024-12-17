import asyncio
import logging

from llm_client.datamodels import LLMFunction
from llm_client.functions.llm_func import LLMFuncArgs

from agent_service.GPT.constants import GPT4_O, Prompt
from agent_service.GPT.requests import GPT
from agent_service.io_type_utils import IOType, load_io_type
from agent_service.io_types.table import Table
from agent_service.io_types.text import Text
from agent_service.q_and_a.utils import QAContext
from agent_service.tool import ToolArgs, default_tool_registry
from agent_service.tools import *  # noqa
from agent_service.types import ChatContext
from agent_service.utils.async_db import get_async_db
from agent_service.utils.logs import async_perf_logger
from agent_service.utils.output_utils.utils import io_type_to_gpt_input

logger = logging.getLogger(__name__)

# TODO: handle tables in inputs and outputs

TASK_EXAMINE_SYSTEM_PROMPT = Prompt(
    name="TASK_EXAMINE_SYSTEM_PROMPT",
    template=(
        "Your goal is to provide a clear and accurate answer to a query about performing a specific task. "
        "The task involves a tool function that has been executed as part of a plan to address a user's request. "
        "You will be given the following information to assist in crafting your response:\n"
        "- Task input arguments\n"
        "- Output of the tool function\n"
        "- Debug information (if available)\n"
        "- Tool description\n"
        "- Inputs and outputs for the task might also be provided if the query is about them.\n"
        "Guidelines for providing a good answer:\n"
        "\n1. Use the debug information, if available, to explain the inner workings of the tool function.\n"
        "\n2. Refer to the tool description for context about the tool function and its purpose.\n"
        "\n3. Analyze the task input arguments to explain the inputs provided to the tool.\n"
        "\n4. Use the tool function's output to describe the results produced by the task.\n\n"
        "\n5. Ensure your response is concise, relevant, and addresses the user's query effectively. "
        "\n6. Never mention the word 'tool', 'function', etc. the user is not technical."
        "The answer should be in one paragraph and should not exceed 200 words unless user asks for more details. "
    ),
)

TASK_EXAMINE_MAIN_PROMPT = Prompt(
    name="TASK_EXAMINE_MAIN_PROMPT",
    template=(
        "You have been asked a query about the details of a task performed using a tool function. "
        "This task was executed as part of a plan to address a user's request.\n\n"
        "Below is the relevant information to help you prepare your response:\n"
        "{task_related_info}\n\n"
        "Now, answer the following query based on the provided information:\n"
        "{query}\n\n"
        "Write a clear, well-structured response that directly addresses the query."
    ),
)

TASKS_RELATED_INFO = Prompt(
    name="TASKS_RELATED_INFO",
    template=(
        "### Task Name ###\n"
        "{tool_name}\n"
        "## Task Debug Info  ###\n"
        "{task_debug_info}\n"
        "## Tool Description ###\n"
        "{tool_desc}\n"
        "## Task Inputs ###\n"
        "{inputs}\n"
        "## Task Outputs ###\n"
        "{outputs}\n"
    ),
)


class ExamineTaskArgs(LLMFuncArgs):
    task_id: str
    tool_name: str
    query: str


@async_perf_logger
async def get_all_task_related_info(args: ExamineTaskArgs, context: QAContext) -> str:
    task_id = args.task_id
    tool_name = args.tool_name
    plan_run_id = context.plan_run_id
    # extract task details from db
    async_db = get_async_db()
    task_run_info = await async_db.get_task_run_info(
        plan_run_id=plan_run_id,
        task_id=task_id,
        tool_name=tool_name,
    )

    if task_run_info:
        task_args_raw, task_outputs_raw, task_debug_info, _ = task_run_info
    else:
        logger.info(f"Task run info not found for plan_run_id={plan_run_id}, task_id={task_id}")
        raise Exception("Task run info not found")

    # get task description from db
    tool = default_tool_registry().get_tool(tool_name)
    description = tool.description

    # prepare outputs
    outputs_io = load_io_type(task_outputs_raw)
    outputs = await prepare_for_gpt(outputs_io)

    # prepare inputs
    tool_args = tool.input_type.model_validate_json(task_args_raw)
    inputs = await prepare_for_gpt(tool_args)

    # prepare task related info
    task_related_info = TASKS_RELATED_INFO.format(
        tool_name=tool_name,
        task_debug_info=task_debug_info,
        tool_desc=description,
        inputs=inputs,
        outputs=outputs,
    ).filled_prompt

    return task_related_info


@async_perf_logger
async def examine_task(args: ExamineTaskArgs, context: QAContext) -> str:
    task_related_info = await get_all_task_related_info(args, context)
    llm = GPT(model=GPT4_O, context=context.gpt_context)
    main_prompt = TASK_EXAMINE_MAIN_PROMPT.format(
        query=args.query,
        task_related_info=task_related_info,
    )
    # save main prompt as txt file
    # with open("main_prompt.txt", "w") as f:
    #     f.write(main_prompt.filled_prompt)

    res = await llm.do_chat_w_sys_prompt(
        main_prompt=main_prompt,
        sys_prompt=TASK_EXAMINE_SYSTEM_PROMPT.format(),
    )
    return res


EXAMINE_TASK_FUNC = LLMFunction(
    name="examine_workflow_task",
    args=ExamineTaskArgs,
    func=examine_task,
    description=(
        "Examine a specific function in a the workflow, for example to see what it does "
        "or give details about how it works and what it returned or its inputs. "
        "Only call this when the query is related to a specific function in the workflow. "
        "`task_id` should be the task ID of the task that you are looking at. "
    ),
)


async def prepare_for_gpt(io_types: IOType) -> str:
    # for input cases
    if isinstance(io_types, ToolArgs):
        res = {}
        for arg_name, arg_value in io_types.__dict__.items():
            if isinstance(arg_value, list):
                if isinstance(arg_value[0], Text) or isinstance(arg_value[0], Table):
                    # only take the first 100 itmes if it is a list of text or table
                    arg_value = arg_value[:100]

            res[arg_name] = await io_type_to_gpt_input(
                arg_value, use_abbreviated_output=True, concurrency_n=50, truncate_to=10000
            )

        # format the res as string
        res = "\n\n".join([f"# `{k}`:\n{v}" for k, v in res.items()])
    # for list of outputs case
    elif isinstance(io_types, list):
        if isinstance(io_types[0], Text) or isinstance(io_types[0], Table):
            # only take the first 100 items if it is a list of text or table
            io_types = io_types[:100]
        res = await io_type_to_gpt_input(
            io_types, use_abbreviated_output=True, concurrency_n=50, truncate_to=10000
        )
    # for single output case
    else:
        res = await io_type_to_gpt_input(
            io_types, use_abbreviated_output=True, concurrency_n=50, truncate_to=10000
        )
    return str(res)


if __name__ == "__main__":
    import asyncio

    async def main() -> None:
        context = QAContext(
            chat_context=ChatContext(),
            agent_id="f495ac3c-1bc2-4780-8e6d-d9f2c58164cd",
            plan_run_id="df93dbbb-5048-41ea-a2f1-20684d7edcdf",
            user_id="3b997275-dcfe-4c19-8bb2-3e1366c4d5f3",
        )
        args = ExamineTaskArgs(
            task_id="a50d0113-fa38-4bfa-a5a9-b9f7bdd746e4",
            tool_name="write_commentary",
            query="what does the client_type arguement is set?",
        )
        res = await examine_task(args, context)
        print(res)

    asyncio.run(main())
