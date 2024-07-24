import asyncio
import json
from typing import List, Optional

from typing_extensions import Self

from agent_service.GPT.constants import GPT4_O
from agent_service.GPT.requests import GPT
from agent_service.io_type_utils import ComplexIOBase, io_type
from agent_service.io_types.output import Output
from agent_service.io_types.stock import StockID
from agent_service.tool import ToolArgs, ToolCategory, ToolRegistry, tool
from agent_service.tools.tool_log import tool_log
from agent_service.types import ChatContext, Message, PlanRunContext
from agent_service.utils.boosted_pg import BoostedPG
from agent_service.utils.date_utils import get_now_utc
from agent_service.utils.output_utils.output_construction import get_output_from_io_type
from agent_service.utils.postgres import get_psql
from agent_service.utils.prefect import get_prefect_logger
from agent_service.utils.prompt_utils import Prompt
from agent_service.utils.string_utils import repair_json_if_needed

GET_CATEGORIES_FOR_STOCK_SYS_PROMPT = Prompt(
    name="GET_CATEGORIES_FOR_STOCK_SYS_PROMPT",
    template="""
    Here is the chat context which may include information to make \
    the output more specific or accurate:
    {chat_context}
    You may be provided the sentence of a hypothesis evaluating something \
    in a financial setting, return the 3-{limit} most important criteria to evaluate it.
    If you are not provided with this prompt, you should ignore this step.

    Think about the criteria in a financial setting where the user is trying to \
    evaluate future looking trends. When looking at a hypothesis about a company \
    your objective is to find criteria that make comparing that company to peers \
    and competitors easy.

    For example if the hypothesis is "Is NVDA a leader in AI chip development" \
    the criteria could be:
    Innovation in Architecture
    Scalability
    Energy Efficiency of Current and Next Generation Chips
    LLM Efficiency of Current and Next Generation Chips
    Software Ecosystem
    Partnerships and Collaborations
    Manufacturing Capabilities

    If the hypothesis is "Evaluate if AVGO is a leader in the smartphone market" \
    the criteria could be:
    Power Efficiency
    Connectivity Solutions
    Graphic Processing Proficiency
    Support for AI and Machine Learning
    Supply Chain Reliability
    Collaborations with Smartphone Manufacturers

    If the hypothesis is "Tell me if Moderna is a leader in oncology" \
    the criteria could be:
    R&D Capabilities
    Robust Pipeline
    Strategic Partnerships
    Regulatory Success
    Commercialization Strategy
    Reputation and Credibility"

    If the hypothesis is "How is Ford positioned in the automotive \
    semi-truck market?"
    Appeal of Brand in the Semi-truck Space
    Cost of current and future models
    Warranty
    Towing Capacity of future and current models
    Fuel efficiency of current and future models

    You also may be provided a list of names of topics that the user is interested in
    using for a hypothesis evaluating something in a financial setting.
    If this is the case, you should append this to the list of categories
    generated in the previous step.
    If this list is none, you can skip this step.

    Output the list in the format [CRITERIA_1, CRITERIA_2, CRITERIA_3] where \
    each CRITERIA is a pythonic dictionary containing:
    name: containing the criteria heading
    explanation: explanation describing what this criteria is
    justification: reason for why this criteria is important to evaluate the hypothesis, \
    be sure to include any specific key metrics to focus on during evaluation
    weight: a float number out of 10.0 of how important this criteria is, 1.0 meaning the \
    criteria is not important at all and 10 meaning the most important criteria, it is not \
    a ranking but should be based on the importance of the criteria and be comparable
    IMPORTANT: Do not supply any additional explanation or justification other than what \
    is provided in the CRITERIA dictionary list
    IMPORTANT: Ensure that the name of the company and company \
    specific products are not mentioned in the criteria explanation or justification
    IMPORTANT: Ensure that you are returning exactly one criteria for each user provided topic name
    """,
)
GET_CATEGORIES_FOR_STOCK_MAIN_PROMPT = Prompt(
    name="GET_CATEGORIES_FOR_STOCK_MAIN_PROMPT",
    template="""
    {company_description_str}
    Here is the hypothesis: \"{prompt}\"
    Here are the names of the requested category topics: \"{names}\"
    """,
)

DEFAULT_CATEGORY_LIMIT = 7


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

    def __hash__(self) -> int:
        return hash((self.name, self.explanation, self.justification, self.weight))

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


@io_type
class Categories(ComplexIOBase):
    val: List[Category]

    async def to_rich_output(self, pg: BoostedPG, title: str = "") -> Output:
        return await get_output_from_io_type(val=self.val, pg=pg, title=title)


class CategoriesForStockInput(ToolArgs):
    prompt: Optional[str] = None
    category_names: Optional[List[str]] = None
    stock: Optional[StockID] = None
    limit: Optional[int] = None


@tool(
    description=f"""
    This function returns a list of success criteria
    that would be useful in evaluating a prompt.
    This function also takes in a list of user requested category names
    and creates a list of success criteria.
    IMPORTANT: If the user wants to add categories to a list of criteria,
    this function MUST be used to first convert the list of category_names to a list of
    criteria before invoking add_lists.
    By default, the function returns up to {DEFAULT_CATEGORY_LIMIT}
    criteria for prompts however, a optional limit parameter can be passed in to
    increase or decrease the number of criteria outputted.
    The function must only return one criteria for each user provided topic name.
    If the user provides a prompt, the function generate a list of criteria.
    If the user provides a list of specific category_names to be included,
    the function will return a list of corresponding criteria for those category_names.
    In order to enhance the accuracy of the tool's output, you may
    also provide the StockID of the stock associated with the prompt.
    IMPORTANT: This tool should only be used to identify success criteria
    for a prompt or user specified list of topics they want criteria generated for.
    IMPORTANT: Either a prompt or a list of category_names or both must be provided.
    Do not use this tool in conjunction with other tools.
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
        category_names=args.category_names,
        stock=args.stock,
        limit=args.limit,
    )

    await tool_log(
        log=f"Found {len(categories)} categories for prompt {args.prompt}", context=context
    )

    return categories


async def get_categories_for_stock_impl(
    llm: GPT,
    context: PlanRunContext,
    prompt: Optional[str],
    category_names: Optional[List[str]],
    stock: Optional[StockID],
    limit: Optional[int],
) -> List[Category]:
    logger = get_prefect_logger(__name__)

    if (prompt is None or prompt == "") and (category_names is None or category_names == []):
        logger.info("Could not generate categories because missing prompt and category names")
        return []

    if not limit:
        limit = DEFAULT_CATEGORY_LIMIT

    company_description_str = ""
    if stock:
        db = get_psql()
        company_description, _ = db.get_short_company_description(stock.gbi_id)
        company_description_str += f"""
        Here is the company description of {stock.company_name} for reference:
        {company_description}
        Please be specific in the criteria with respect to the actual business of \
        {stock.company_name} as described in the company description as well as any market trends.
        """

    # initial prompt for categories
    initial_categories_gpt_resp = await llm.do_chat_w_sys_prompt(
        main_prompt=GET_CATEGORIES_FOR_STOCK_MAIN_PROMPT.format(
            prompt=prompt, names=category_names, company_description_str=company_description_str
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
        prompt="Does EQIX have younger (or older) data centers than their competitors",
        stock=StockID(
            gbi_id=5766, symbol="INTC", isin="US4581401001", company_name="Intel Corporation"
        ),
        limit=7,
    )
    categories: List[Category] = await get_categories(args=categories_input, context=plan_context)  # type: ignore
    for category in categories:
        print(category.to_markdown_string())


if __name__ == "__main__":
    asyncio.run(main())
