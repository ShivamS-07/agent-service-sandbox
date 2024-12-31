import asyncio
import json
from typing import List, Optional, Self

import pydantic_core

from agent_service.GPT.constants import GPT4_O
from agent_service.GPT.requests import GPT
from agent_service.io_type_utils import ComplexIOBase, io_type
from agent_service.io_types.output import Output
from agent_service.io_types.stock import StockID
from agent_service.planner.errors import EmptyOutputError
from agent_service.tool import ToolArgs, ToolCategory, default_tool_registry, tool
from agent_service.tools.tool_log import tool_log
from agent_service.types import ChatContext, Message, PlanRunContext
from agent_service.utils.boosted_pg import BoostedPG
from agent_service.utils.cache_utils import PostgresCacheBackend
from agent_service.utils.date_utils import get_now_utc
from agent_service.utils.output_utils.output_construction import get_output_from_io_type
from agent_service.utils.postgres import get_psql
from agent_service.utils.prompt_utils import Prompt
from agent_service.utils.string_utils import repair_json_if_needed

GET_CATEGORIES_FOR_STOCK_SYS_PROMPT = Prompt(
    name="GET_CATEGORIES_FOR_STOCK_SYS_PROMPT",
    template="""
    You are a financial analyst who is creating a list of criteria for a competitive analysis
    in a financial setting. Return the top 3 to {limit} MOST important criteria for that market.
    The criteria should be in financial setting where the user is trying to evaluate future looking
    trends.  Your objective is to find criteria that make it easy to a group of competitors.
    You should try to avoid overlap between criteria and make sure that each criteria is unique and distinct.

    For example if the market is "AI chip development" \
    the criteria could potentially be:
    Innovation in Architecture
    Scalability
    Energy Efficiency of Current and Next Generation Chips
    LLM Efficiency of Current and Next Generation Chips
    Software Ecosystem
    Partnerships and Collaborations
    Manufacturing Capabilities

    If the market is "smartphone market" \
    the criteria could potentially be:
    Power Efficiency
    Connectivity Solutions
    Graphic Processing Proficiency
    Support for AI and Machine Learning
    Supply Chain Reliability
    Collaborations with Smartphone Manufacturers

    If the market is "oncology" \
    the criteria could potentially be:
    R&D Capabilities
    Robust Pipeline
    Strategic Partnerships
    Regulatory Success
    Commercialization Strategy
    Reputation and Credibility"

    If the market is "automotive semi-truck market"
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
    - justification: reason for why this criteria is important to carry out the competitive analysis, \
        be sure to include any specific key metrics to focus on during evaluation. Your justification \
        should be consistent with the explanation and the weight but you should never mention the weight \
        explictly.
    You must not supply any additional explanation or justification other than what is provided in \
    the CRITERIA dictionary list. Your explanation and justification must not contain any specific \
    company names or products.

    Here is the market you are doing a competitive analysis for: {market}
    Here are the chat context details you may use to make the output more specific and accurate:
    {chat_context}

    {must_include_criteria_names_str}
    {company_description_str}
    """,
)

DEFAULT_CATEGORY_LIMIT = 7
CATEGORY_CACHE_TTL = 10 * 365 * 24 * 60 * 60  # 10 years
MAX_RETRIES = 5


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

        text: Text = Text(val=self.to_markdown_string())
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


# for backward compatibility


class CategoriesForStockInput(ToolArgs):
    prompt: str
    must_include_criteria_names: Optional[List[str]] = None
    target_stock: Optional[StockID] = None
    limit: Optional[int] = None


@tool(
    description="",
    category=ToolCategory.COMPETITIVE_ANALYSIS,
    enabled=False,
)
async def get_success_criteria(
    args: CategoriesForStockInput, context: PlanRunContext
) -> List[Category]:
    return await get_criteria_for_competitive_analysis(  # type: ignore
        args=CriteriaForCompetitiveAnalysis(
            market=args.prompt,
            must_include_criteria=args.must_include_criteria_names,
            target_stock=args.target_stock,
            limit=args.limit,
        ),
        context=context,
    )


class CriteriaForCompetitiveAnalysis(ToolArgs):
    market: str
    must_include_criteria: Optional[List[str]] = None
    target_stock: Optional[StockID] = None
    limit: Optional[int] = None


# Cache on plan ID
def category_cache_key_fn(tool_name: str, args: ToolArgs, context: PlanRunContext) -> str:
    return f"{tool_name}-{context.task_id}-{context.plan_id}"


@tool(
    description=f"""
    This function returns a list of criteria that is required for doing a competitive market analysis of
    companies that offer the similar products or services. You use this function if and only if you are
    also calling the other competitive_analysis tools.
    The `market` is a string which describes the shared market for companies in the analysis. It is
    typically a class of product or service, or possibly an entire industry or sector. It should be
    as specific as possible, e.g. if the client asks for the leader in millennial gaming, the market
    should be millenial gaming, not gaming.
    If the client has explicitly mentioned relevant criteria in their request, the list of criteria should
    be passed to this tool as `must_include_criteria` which is a list of strings. For example, if the request
    includes the phrase `...make sure to compare based on innovation.`, then the string `innovation` should
    be in the list.
    If a single specific company is mentioned in the user input, then an identifier for that company
    MUST be passed as the `target_stock`. This stock will be used to get more information about the company
    thus enhancing the quality of the output. For example, if the client asks 'What makes Expedia
    a leader in the travel industry', then you should set target_stock to Expedia's stock identifier.
    By default, the function returns up to {DEFAULT_CATEGORY_LIMIT} criteria. However, an optional
    'limit' parameter can be passed in to adjust the number of outputted criteria based on the user input.
    Note that these criteria are very general and only appropriate as a high level rubric for evalulating
    stocks in the context of a competitive analysis.
    If multiple stocks are mentioned by name in the input, you must pick the most prominently mentioned stock as
    the target stock. For example, if the input is `Is Expedia a market leader compared to PriceLine and
    and Kayak`, your target stock would be Expedia. You may only have one target stock.
    Never use this tool in the context of due diligence, use the summary tool to write due diligence criteria.
    """,
    category=ToolCategory.COMPETITIVE_ANALYSIS,
    tool_registry=default_tool_registry(),
    use_cache=True,
    cache_key_fn=category_cache_key_fn,
    cache_ttl=CATEGORY_CACHE_TTL,
    cache_backend=PostgresCacheBackend(),
)
async def get_criteria_for_competitive_analysis(
    args: CriteriaForCompetitiveAnalysis, context: PlanRunContext
) -> List[Category]:
    if not args.market:
        raise ValueError("A market must be provided")

    llm = GPT(context=None, model=GPT4_O)
    categories = await get_categories_for_stock_impl(
        llm=llm,
        context=context,
        market=args.market,
        must_include_criteria_names=args.must_include_criteria,
        target_stock=args.target_stock,
        limit=args.limit,
    )

    await tool_log(
        log=f"Found {len(categories)} criteria for market {args.market}", context=context
    )

    return categories


async def get_categories_for_stock_impl(
    llm: GPT,
    context: PlanRunContext,
    market: str,
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
    output = None
    retry_count = 0
    while output is None and retry_count < MAX_RETRIES:
        try:
            initial_categories_gpt_resp = await llm.do_chat_w_sys_prompt(
                sys_prompt=GET_CATEGORIES_FOR_STOCK_SYS_PROMPT.format(limit=limit),
                main_prompt=GET_CATEGORIES_FOR_STOCK_MAIN_PROMPT.format(
                    market=market,
                    chat_context=context.chat,
                    must_include_criteria_names_str=must_include_criteria_names_str,
                    company_description_str=company_description_str,
                ),
                no_cache=True,
                output_json=True,
            )
            categories = json.loads(repair_json_if_needed(initial_categories_gpt_resp))
            output = [Category(**category) for category in categories]
        except pydantic_core._pydantic_core.ValidationError:
            retry_count += 1

    if not output:
        raise EmptyOutputError("Unable to generate appropriate criteria for competitive analysis")
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
    categories_input = CriteriaForCompetitiveAnalysis(
        market="data centers",
        target_stock=StockID(
            gbi_id=5766, symbol="INTC", isin="US4581401001", company_name="Intel Corporation"
        ),
        must_include_criteria=["age"],
        limit=7,
    )
    categories: List[Category] = await get_criteria_for_competitive_analysis(
        args=categories_input, context=plan_context
    )  # type: ignore
    for category in categories:
        print(category.to_markdown_string())


if __name__ == "__main__":
    asyncio.run(main())
