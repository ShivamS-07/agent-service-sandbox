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
from agent_service.utils.prompt_utils import Prompt
from agent_service.utils.string_utils import repair_json_if_needed

GET_CATEGORIES_FOR_STOCK_SYS_PROMPT = Prompt(
    name="GET_CATEGORIES_FOR_STOCK_SYS_PROMPT",
    template="""
    You are a financial analyst who is creating a list of success criteria to evaluate a hypothesis \
    in a financial setting. Return the top 3 to {limit} MOST important criteria.
    The criteria should be in financial setting where the user is trying to evaluate future looking \
    trends. When looking at a hypothesis about a specific company your objective is to find criteria \
    that make it easy to compare the company with its peers and competitors. You should try to avoid \
    overlap between criteria and make sure that each criteria is unique and distinct. \

    For example if the question is "Is NVDA a leader in AI chip development" \
    the criteria could potentially be:
    Innovation in Architecture
    Scalability
    Energy Efficiency of Current and Next Generation Chips
    LLM Efficiency of Current and Next Generation Chips
    Software Ecosystem
    Partnerships and Collaborations
    Manufacturing Capabilities

    If the question is "Evaluate if AVGO is a leader in the smartphone market" \
    the criteria could potentially be:
    Power Efficiency
    Connectivity Solutions
    Graphic Processing Proficiency
    Support for AI and Machine Learning
    Supply Chain Reliability
    Collaborations with Smartphone Manufacturers

    If the question is "Tell me if Moderna is a leader in oncology" \
    the criteria could potentially be:
    R&D Capabilities
    Robust Pipeline
    Strategic Partnerships
    Regulatory Success
    Commercialization Strategy
    Reputation and Credibility"

    If the question is "How is Ford positioned in the automotive \
    semi-truck market?"
    Appeal of Brand in the Semi-truck Space
    Cost of current and future models
    Warranty
    Towing Capacity of future and current models
    Fuel efficiency of current and future models
    """,
)
GET_CATEGORIES_FOR_STOCK_MAIN_PROMPT = Prompt(
    name="GET_CATEGORIES_FOR_STOCK_MAIN_PROMPT",
    template="""
    Your output should be in a JSON format of a list of objects where each object has the following \
    keys:
    - name: containing the criteria heading in the title format (capitalize the main words).
    - explanation: explanation describing what this criteria is
    - weight: a float number out of 10.0 of how important this criteria is, 1.0 meaning the \
        criteria is not important at all and 10 meaning the most important criteria, it is not \
        a ranking but should be based on the importance of the criteria and be comparable
    - justification: reason for why this criteria is important to evaluate the hypothesis, \
        be sure to include any specific key metrics to focus on during evaluation. Your justification \
        should be consistent with the explanation and the weight but you should never mention the weight \
        explictly.
    You must not supply any additional explanation or justification other than what is provided in \
    the CRITERIA dictionary list. Your explanation and justification must not contain any specific \
    company names or products.

    Here is the hypothesis you should evaluate: {prompt}
    Here are the chat context details you may use to make the output more specific and accurate: {chat_context}
    {must_include_criteria_names_str}
    {company_description_str}
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
    def multi_to_gpt_input(cls, categories: List[Self], include_weight: bool = True) -> str:
        output_list = []
        for idx, category in enumerate(categories):
            text = (
                f"- {idx}: {category.name}\n"
                f"Explanation: {category.explanation}\n"
                f"Justification: {category.justification}"
            )
            if include_weight:
                text += f"\nWeight: {category.weight}"

            output_list.append(text)

        return "\n".join(output_list)


@io_type
class Categories(ComplexIOBase):
    val: List[Category]

    async def to_rich_output(self, pg: BoostedPG, title: str = "") -> Output:
        return await get_output_from_io_type(val=self.val, pg=pg, title=title)


class CategoriesForStockInput(ToolArgs):
    prompt: str
    must_include_criteria_names: Optional[List[str]] = None
    target_stock: Optional[StockID] = None
    limit: Optional[int] = None


@tool(
    description=f"""
    This function returns a list of success criteria that would be useful in evaluating a prompt. \
    By default, the function returns up to {DEFAULT_CATEGORY_LIMIT} criteria. However, an optional \
    'limit' parameter can be passed in to adjust the number of outputted criteria from the client query. \
    In addition, the function can also take in a list of criteria names that must be included in the \
    output which you should also get from the client query.
    If a single specific company is mentioned in the user input, then an identifier for that company \
    MUST be passed as the target stock. This stock will be used to get more information about the company \
    thus enhancing the accuracy of the output.
    For example, if the client query is 'What makes Expedia a leader in the travel industry', \
    then you should set 'stock` as Expedia's stock identifier. If the query asks 'You should include \
    X, Y in the criteria and give the top 5 criteria', then 'must_include_criteria_names' should be \
    set as ['X', 'Y'] and 'limit' should be set as 5.
    """,
    category=ToolCategory.STOCK,
    tool_registry=ToolRegistry,
)
async def get_success_criteria(
    args: CategoriesForStockInput, context: PlanRunContext
) -> List[Category]:
    if not args.prompt:
        raise ValueError("Prompt must be provided")

    llm = GPT(context=None, model=GPT4_O)
    categories = await get_categories_for_stock_impl(
        llm=llm,
        context=context,
        prompt=args.prompt,
        must_include_criteria_names=args.must_include_criteria_names,
        target_stock=args.target_stock,
        limit=args.limit,
    )

    await tool_log(
        log=f"Found {len(categories)} success criteria for prompt {args.prompt}", context=context
    )

    return categories


async def get_categories_for_stock_impl(
    llm: GPT,
    context: PlanRunContext,
    prompt: str,
    must_include_criteria_names: Optional[List[str]],
    target_stock: Optional[StockID],
    limit: Optional[int],
) -> List[Category]:
    if not limit:
        limit = DEFAULT_CATEGORY_LIMIT

    company_description_str = ""
    if target_stock:
        db = get_psql()
        company_description, _ = db.get_short_company_description(target_stock.gbi_id)
        company_description_str = f"""
            The target company is {target_stock.symbol} ({target_stock.company_name})
            Here is its company description {company_description}
            Read the company description for reference and be specific in the criteria with respect \
            to the actual business in the company description as well as any market trends.
        """

    must_include_criteria_names_str = ""
    if must_include_criteria_names:
        joined_criteria_names = ", ".join(must_include_criteria_names)
        must_include_criteria_names_str = f"""
            Here are the names of the criteria you MUST have in the output: {joined_criteria_names} \
            Each name inside the list represents a single criteria and you should NOT expand it to \
            more criteria. Use the names as they are provided, and create explanations and justifications \
            for them. Create other criteria as needed based on the provided information and do not \
            overlap with each other.
            Be objective to the weights of these criteria and DO NOT always rank the clients' requested \
            criteria at the top unless they explicitly tell you these are very important.
        """

    # initial prompt for categories
    initial_categories_gpt_resp = await llm.do_chat_w_sys_prompt(
        sys_prompt=GET_CATEGORIES_FOR_STOCK_SYS_PROMPT.format(limit=limit),
        main_prompt=GET_CATEGORIES_FOR_STOCK_MAIN_PROMPT.format(
            prompt=prompt,
            chat_context=context.chat,
            must_include_criteria_names_str=must_include_criteria_names_str,
            company_description_str=company_description_str,
        ),
    )
    categories = json.loads(repair_json_if_needed(initial_categories_gpt_resp))

    output = [Category(**category) for category in categories]
    output.sort(key=lambda x: x.weight, reverse=True)
    return output


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
        target_stock=StockID(
            gbi_id=5766, symbol="INTC", isin="US4581401001", company_name="Intel Corporation"
        ),
        limit=7,
    )
    categories: List[Category] = await get_success_criteria(args=categories_input, context=plan_context)  # type: ignore
    for category in categories:
        print(category.to_markdown_string())


if __name__ == "__main__":
    asyncio.run(main())
