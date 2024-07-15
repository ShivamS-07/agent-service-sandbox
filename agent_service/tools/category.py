import asyncio
import json
from typing import List

from agent_service.GPT.constants import GPT4_O
from agent_service.GPT.requests import GPT
from agent_service.io_type_utils import ComplexIOBase, io_type
from agent_service.io_types.output import Output
from agent_service.tool import ToolArgs, ToolCategory, ToolRegistry, tool
from agent_service.tools.tool_log import tool_log
from agent_service.types import ChatContext, Message, PlanRunContext
from agent_service.utils.boosted_pg import BoostedPG
from agent_service.utils.date_utils import get_now_utc
from agent_service.utils.prefect import get_prefect_logger
from agent_service.utils.prompt_utils import Prompt
from agent_service.utils.string_utils import repair_json_if_needed

GET_CATEGORIES_FOR_STOCK_SYS_PROMPT = Prompt(
    name="GET_CATEGORIES_FOR_STOCK_SYS_PROMPT",
    template="Here is the chat context:\n"
    "{chat_context}\n"
    "You are financial analyst. Your client would like to do "
    "some comparative analysis on stocks and the current market. "
    "You will be provided with a prompt that the client would like "
    "to evaluate. Your job is to identify specific key success criteria "
    "that the client should use to evaluate the given prompt or to determine "
    "the validity of the prompt.\n"
    "IMPORTANT: Identify only the 5 most impactful key success criteria.\n",
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


class CategoriesForStockInput(ToolArgs):
    prompt: str


@tool(
    description="""
    This function returns a list of Categories (a list of success criteria)
    which should be used to perform comparative analysis on the given stock.
    Optionally, the enhance the accuracy of the output of this tool,
    a list of peer StockIDs, a sector or industry the stock operates in, and
    the prompt the user would like to evaluate can all be passed in to help
    the tool identify better and more specific Categories.
    """,
    category=ToolCategory.STOCK,
    tool_registry=ToolRegistry,
    enabled=False,
)
async def get_categories(args: CategoriesForStockInput, context: PlanRunContext) -> List[Category]:
    llm = GPT(context=None, model=GPT4_O)
    categories = await get_categories_for_stock_impl(
        llm=llm,
        context=context,
        prompt=args.prompt,
    )

    await tool_log(
        log=f"Found {len(categories)} categories for prompt {args.prompt}", context=context
    )

    return categories


async def get_categories_for_stock_impl(
    llm: GPT, context: PlanRunContext, prompt: str
) -> List[Category]:
    logger = get_prefect_logger(__name__)

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
