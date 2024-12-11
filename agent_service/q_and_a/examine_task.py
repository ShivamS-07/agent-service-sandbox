import asyncio
import logging

from llm_client.datamodels import LLMFunction
from llm_client.functions.llm_func import LLMFuncArgs

from agent_service.GPT.constants import GPT4_O, Prompt
from agent_service.GPT.requests import GPT
from agent_service.io_type_utils import load_io_type
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
        "Your goal is to provide a clear and accurate answer to a question about performing a specific task. "
        "The task involves a tool function that has been executed as part of a plan to address a user's request. "
        "You will be given the following information to assist in crafting your response:\n"
        "- Task input arguments\n"
        "- Output of the tool function\n"
        "- Debug information (if available)\n"
        "- Tool description\n"
        "- Inputs and outputs for the task might also be provided if the question is about them.\n"
        "Guidelines for providing a good answer:\n"
        "\n1. Use the debug information, if available, to explain the inner workings of the tool function.\n"
        "\n2. Refer to the tool description for context about the tool function and its purpose.\n"
        "\n3. Analyze the task input arguments to explain the inputs provided to the tool.\n"
        "\n4. Use the tool function's output to describe the results produced by the task.\n\n"
        "\n5. Ensure your response is concise, relevant, and addresses the user's question effectively. "
        "\n6. Never mention the word 'tool', 'function', etc. the user is not technical."
        "The answer should be in one paragraph and should not exceed 200 words unless user asks for more details. "
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
        "{input_prompt}"
        "{output_prompt}"
        "Now, answer the following question based on the provided information:\n"
        "{question}\n\n"
        "Write a clear, well-structured response that directly addresses the question."
    ),
)


class ExamineTaskArgs(LLMFuncArgs):
    task_id: str
    tool_name: str
    question: str
    is_question_about_inputs: bool = False
    is_question_about_outputs: bool = False


@async_perf_logger
async def examine_task(args: ExamineTaskArgs, context: QAContext) -> str:
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

    # get tool args object
    inputs, outputs = None, None
    if args.is_question_about_inputs:
        tool_args = tool.input_type.model_validate_json(task_args_raw)
        inputs = await prepare_input_for_gpt(tool_args, args.question)
    if args.is_question_about_outputs:
        outputs = await prepare_output_for_gpt(task_outputs_raw, args.question)

    llm = GPT(model=GPT4_O, context=context.gpt_context)
    main_prompt = TASK_EXAMINE_MAIN_PROMPT.format(
        task_debug_info=task_debug_info,
        tool_desc=tool.description,
        question=args.question,
        input_prompt=f"\n\n### Task Inputs ###\n{inputs}\n\n" if inputs else "",
        output_prompt=f"\n\n### Task Outputs ###\n{outputs}\n\n" if outputs else "",
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
        "Only call this when the question is related to a specific function in the workflow. "
        "`task_id` should be the task ID of the task that you are looking at. "
        "`is_question_about_inputs` should be set to True if the question is about the inputs of the task. "
        "`is_question_about_outputs` should be set to True if the question is about the outputs of the task. "
    ),
)


async def prepare_input_for_gpt(tool_args: ToolArgs, question: str) -> str:
    """
    helper function to convert inputs to gpt format and truncate if needed
    """
    res = {}
    for arg_name, arg_value in tool_args.__dict__.items():
        if isinstance(arg_value, list):
            # only take the first 100 items
            arg_value = arg_value[:100]
        res[arg_name] = await io_type_to_gpt_input(
            arg_value, use_abbreviated_output=True, concurrency_n=50
        )
        # truncate if the total length exceeds 10k
        res[arg_name] = " ".join(res[arg_name].split()[:10000])

    # format the res as string
    res = "\n\n".join([f"# `{k}`:\n{v}" for k, v in res.items()])
    return res


async def prepare_output_for_gpt(task_outputs_raw: str, question: str) -> str:
    outputs = load_io_type(task_outputs_raw)
    if isinstance(outputs, list):
        # only take the first 100 items
        outputs = outputs[:100]
    res = await io_type_to_gpt_input(outputs, use_abbreviated_output=True, concurrency_n=50)
    # truncate if the total length exceeds 10k
    res = " ".join(res.split()[:10000])
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
            question="what does the client_type arguement is set?",
            is_question_about_inputs=True,
            is_question_about_outputs=True,
        )
        res = await examine_task(args, context)
        print(res)

    asyncio.run(main())
