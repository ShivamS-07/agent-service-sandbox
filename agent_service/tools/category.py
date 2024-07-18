import asyncio
import json
from typing import List, Optional

from typing_extensions import Self

from agent_service.GPT.constants import GPT4_O
from agent_service.GPT.requests import GPT
from agent_service.io_type_utils import ComplexIOBase, io_type
from agent_service.io_types.output import Output
from agent_service.tool import ToolArgs, ToolCategory, ToolRegistry, tool
from agent_service.tools.tool_log import tool_log
from agent_service.types import ChatContext, Message, PlanRunContext
from agent_service.utils.boosted_pg import BoostedPG
from agent_service.utils.date_utils import get_now_utc
from agent_service.utils.output_utils.output_construction import get_output_from_io_type
from agent_service.utils.prefect import get_prefect_logger
from agent_service.utils.prompt_utils import Prompt
from agent_service.utils.string_utils import repair_json_if_needed

GET_CATEGORIES_FOR_STOCK_SYS_PROMPT = Prompt(
    name="GET_CATEGORIES_FOR_STOCK_SYS_PROMPT",
    template="Here is the chat context which may include information to "
    "make the output more specific or accurate:\n"
    "{chat_context}\n"
    "You are financial analyst. Your client would like to do "
    "some comparative analysis on stocks and the current market. "
    "You will be provided with a prompt that the client would like "
    "to evaluate. Your job is to identify specific key success criteria "
    "that the client should use to evaluate the given prompt or to determine "
    "the validity of the prompt.\n"
    "IMPORTANT: Identify only the {limit} most impactful key success criteria.\n",
)
GET_CATEGORIES_FOR_STOCK_MAIN_PROMPT = Prompt(
    name="GET_CATEGORIES_FOR_STOCK_MAIN_PROMPT",
    template="""
    Output the list in the format [CRITERIA_1, CRITERIA_2, CRITERIA_3] where
    each CRITERIA is a pythonic dictionary containing:
        name: containing the criteria heading
        explanation: explanation of what this criteria means in the analysis
        justification: reason for why this criteria is important for the
            analysis and any specific metrics to focus on
        weight: a float number out of 10.0 of how important this criteria is
            for the analysis
    IMPORTANT: Please correlate and group criteria that could fall into a similar group together.
        For example, key financial metrics could be grouped together under the criteria "Financial Metrics".
        Make sure to list the important metrics in the justification field.
    IMPORTANT: Do not supply any additional explanation or justification other than what is
        provided in the CRITERIA dictionary list
    {prompt_str}
    """,
)

DEFAULT_CATEGORY_LIMIT = 3


@io_type
class Category(ComplexIOBase):
    name: str
    explanation: str
    justification: str
    weight: float

    def to_markdown_string(self) -> str:
        return (
            f"**Category: {self.name}**\n"
            f" - Explanation: {self.explanation}\n"
            f" - Justification: {self.justification}\n"
            f" - Weight: {self.weight}\n"
        )

    async def to_rich_output(self, pg: BoostedPG, title: str = "") -> Output:
        # convert the category to a rich text format
        from agent_service.io_types.text import Text

        text = Text(val=self.to_markdown_string())
        return await text.to_rich_output(pg=pg)

    async def to_gpt_input(self, use_abbreviated_output: bool = True) -> str:
        return (
            f"- Criteria Name: {self.name}. Explanation: {self.explanation}. "
            f"Justification: {self.justification}."
        )

    @classmethod
    def multi_to_gpt_input(cls, categories: List[Self]) -> str:
        output_list = []
        for idx, category in enumerate(categories):
            output_list.append(
                f"- {idx}: {category.name}\n"
                f"Explanation: {category.explanation}\n"
                f"Justification: {category.justification}\n"
                f"Weight: {category.weight}"
            )

        return "\n".join(output_list)


class Categories(ComplexIOBase):
    val: List[Category]

    async def to_rich_output(self, pg: BoostedPG, title: str = "") -> Output:
        return await get_output_from_io_type(val=self.val, pg=pg, title=title)


class CategoriesForStockInput(ToolArgs):
    prompt: str
    limit: Optional[int] = None


@tool(
    description=f"""
    This function returns a list of success criteria
    which should be used to perform comparative analysis on the given stock.
    By default, the function returns up to {DEFAULT_CATEGORY_LIMIT}
    criteria however, a optional limit parameter can be passed in to
    increase or decrease the number of criteria outputted.
    IMPORTANT: This tool should only be used to identify success criteria
    for a prompt. Do not use this tool in conjunction with other tools.
    """,
    category=ToolCategory.STOCK,
    tool_registry=ToolRegistry,
)
async def get_categories(args: CategoriesForStockInput, context: PlanRunContext) -> List[Category]:
    llm = GPT(context=None, model=GPT4_O)
    categories = await get_categories_for_stock_impl(
        llm=llm,
        context=context,
        prompt=args.prompt,
        limit=args.limit,
    )

    await tool_log(
        log=f"Found {len(categories)} categories for prompt {args.prompt}", context=context
    )

    return categories


async def get_categories_for_stock_impl(
    llm: GPT, context: PlanRunContext, prompt: str, limit: Optional[int]
) -> List[Category]:
    logger = get_prefect_logger(__name__)

    if not limit:
        limit = DEFAULT_CATEGORY_LIMIT

    if prompt is None or prompt == "":
        logger.info("Could not generate categories because missing prompt")
        return []

    prompt_str = f"The prompt the client wants to evaluate is: {prompt}"

    # initial prompt for categories
    initial_categories_gpt_resp = await llm.do_chat_w_sys_prompt(
        main_prompt=GET_CATEGORIES_FOR_STOCK_MAIN_PROMPT.format(
            prompt_str=prompt_str,
        ),
        sys_prompt=GET_CATEGORIES_FOR_STOCK_SYS_PROMPT.format(
            chat_context=context.chat,
            limit=limit,
        ),
    )
    categories = json.loads(repair_json_if_needed(initial_categories_gpt_resp))

    return [Category(**category) for category in categories]


async def main() -> None:
    input_text = "Hello :)"
    user_message = Message(message=input_text, is_user_message=True, message_time=get_now_utc())
    chat_context = ChatContext(messages=[user_message])
    plan_context = PlanRunContext(
        agent_id="123",
        plan_id="123",
        user_id="123",
        plan_run_id="123",
        chat=chat_context,
        run_tasks_without_prefect=True,
        skip_db_commit=True,
    )
    categories_input = CategoriesForStockInput(
        prompt="Is Nintendo the leader in the Video Gaming space?",
    )
    output = await get_categories(args=categories_input, context=plan_context)
    print(output)


if __name__ == "__main__":
    asyncio.run(main())
