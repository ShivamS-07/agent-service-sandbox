from collections import Counter
from typing import Dict, List, Optional, Set

from agent_service.GPT.constants import (
    FILTER_CONCURRENCY,
    GPT4_O,
    GPT35_TURBO,
    MAX_TOKENS,
    NO_PROMPT,
)
from agent_service.GPT.requests import GPT
from agent_service.GPT.tokens import GPTTokenizer
from agent_service.io_type_utils import HistoryEntry
from agent_service.io_types.stock import StockID
from agent_service.io_types.stock_aligned_text import StockAlignedTextGroups
from agent_service.io_types.text import Text
from agent_service.tool import ToolArgs, ToolCategory, ToolRegistry, tool
from agent_service.tools.stock_metadata import (
    GetStockDescriptionInput,
    get_company_descriptions,
)
from agent_service.types import PlanRunContext
from agent_service.utils.async_utils import gather_with_concurrency
from agent_service.utils.prefect import get_prefect_logger
from agent_service.utils.prompt_utils import Prompt

logger = get_prefect_logger(__name__)


SEPARATOR = "###"
CHEAP_LLM = GPT35_TURBO
# Number of rounds to perform GPT call and get results based on majority
CONSISTENCY_ROUNDS = 3


STOCK_PRODUCT_FILTER1_MAIN_PROMPT = Prompt(
    name="STOCK_PRODUCT_FILTER1_MAIN_PROMPT",
    template=(
        "Your task is to determine whether a company is a producer of a given Product "
        "based on Product and company description below. "
        "\n- The Product description below includes a description of the Product and "
        "a list of synonyms and similar terms that might be used interchangeably for the Product. "
        "\n- {must_include_stocks_prompt}"
        "\nSome guidelines to consider: "
        "\n- If the company produce components or services that are closely related to the Product, "
        "the decision MUST be 'YES'. "
        "\n- The Product can be a service (e.g. retail) or physical products. "
        "\n- Your response MUST start with 2 to 3 sentences justification of your decsion, "
        "then your decision 'YES' or 'NO', all delineated with the separator {separator}."
        "\n### Product Description\n"
        "{product_description}"
        "\n### Company Description\n"
        "{company_description}"
        "\nNow, please provide your response."
    ),
)

STOCK_PRODUCT_FILTER2_MAIN_PROMPT = Prompt(
    name="STOCK_PRODUCT_FILTER2_MAIN_PROMPT",
    template=(
        "Your task is to determine whether a company is a producer of a given Product "
        "based on Product and company description below. "
        "\n- The Product description below includes a description of the Product and "
        "a list of synonyms and similar terms that might be used interchangeably for the Product. "
        "\n- {must_include_stocks_prompt}"
        "\n\nSome guidelines to consider: "
        "\n- If the company produce components or services that are used in the production "
        "of the Product, and not producing the product itself the decision MUST be 'NO'. "
        "\n- Product can be service (e.g. retail) or physical product. "
        "\n- Be very strict in your decision, ONLY if the company is a direct producer of the Product, "
        "the decision can be 'YES', Otherwise, the decision MUST be 'NO'. "
        "\n- Your response MUST start with 2 to 3 sentences justification of your decsion, "
        "then your decision 'YES' or 'NO', all delineated with the separator {separator}."
        "\n### Product Description\n"
        "{product_description}"
        "\n### Company Description\n"
        "{company_description}"
        "\nNow, please provide your response."
    ),
)
MUST_INCLUDE_STOCKS_PROMPT = Prompt(
    name="MUST_INCLUDE_STOCKS_PROMPT",
    template=(
        "If the given company is in the following list, you MUST answer 'YES' no matter what. "
        "List of companies: {must_include_stocks}"
    ),
)

PRODUCT_DESCRIPTION_PROMPT = Prompt(
    name="PRODUCT_DESCRIPTION_PROMPT",
    template=(
        "Please provide a short description and all synonyms and similar terms that might be used interchangeably for "
        "the given product or service below. "
        ""
        "\n- Your response should be three parts: 1) a short description of the product or service, and 2) a list of "
        "synonyms and similar terms. "
        "Ensure your response is concise and less than 500 words."
        "\n\n### Product: {product_str}\n"
        "Now, please provide your response."
    ),
)


class FilterStocksByProductOrServiceInput(ToolArgs):
    stock_ids: List[StockID]
    product_str: str
    must_include_stocks: Optional[List[str]] = None


@tool(
    description=(
        "This tool can be used to determine whether a given list of stocks or companies produces a service "
        "or product based on given product_str. This function should only be used if it "
        "is plausible that the user is asking particularly for providers of products or services. "
        "Do not use this function in cases where the user might be looking for consumers of such products "
        "or services, or companies otherwise linked to the product via supply chains. "
        "In these complex cases, use the filter_stocks_by_profile tool instead. "
        "This tool is only for simple cases where it should be clear whether a company counts as a provider"
        "of the product or service based on a company description. "
        "Note that retail should be considered a general service, and then retailers that sell specific kinds "
        "of products are a more specific kind of service. "
        "\n 'stock_ids' is a list of stock ids to filter. "
        "\n 'product_str' is a string representing the product or service to filter by. "
        "For example, 'electric vehicles', 'cloud computing', 'solar panels', 'smartphones', 'AI chips', "
        "'Online retail', etc. "
    ),
    category=ToolCategory.STOCK,
    tool_registry=ToolRegistry,
    is_visible=True,
    enabled=True,
)
async def filter_stocks_by_product_or_service(
    args: FilterStocksByProductOrServiceInput, context: PlanRunContext
) -> List[StockID]:
    # get company/stock descriptions
    description_texts = await get_company_descriptions(
        GetStockDescriptionInput(
            stock_ids=args.stock_ids,
        ),
        context,
    )

    # create aligned stock text groups and get all the text strings
    aligned_text_groups = StockAlignedTextGroups.from_stocks_and_text(
        args.stock_ids, description_texts  # type: ignore
    )
    stock_description_map: Dict[StockID, str] = await Text.get_all_strs(  # type: ignore
        aligned_text_groups.val, include_header=True, text_group_numbering=True
    )
    # filter out those with no data
    stocks = [stock for stock in args.stock_ids if stock in stock_description_map]

    # first round of filtering
    # initiate GPT llm models
    llm_gpt35 = GPT(model=CHEAP_LLM)
    llm_gpt4 = GPT(model=GPT4_O)

    # get must include stocks prompt
    if args.must_include_stocks:
        must_include_stocks_prompt = MUST_INCLUDE_STOCKS_PROMPT.format(
            must_include_stocks=args.must_include_stocks
        ).filled_prompt
    else:
        must_include_stocks_prompt = ""

    # run CONSISTENCY_ROUNDS times to get a more stable result
    tasks = []
    for _ in range(CONSISTENCY_ROUNDS):
        tasks.append(
            filter_stocks_round1(
                product_str=args.product_str,
                stocks=stocks,
                stock_description_map=stock_description_map,
                must_include_stocks_prompt=must_include_stocks_prompt,
                llm_gpt35=llm_gpt35,
                llm_gpt4=llm_gpt4,
            )
        )
    filtered_stocks1_set_list: List[Set[StockID]] = await gather_with_concurrency(
        tasks, n=FILTER_CONCURRENCY
    )

    # get the union if the results
    filtered_stocks1 = list(set.union(*filtered_stocks1_set_list))

    if not filtered_stocks1:
        raise ValueError(
            f"No stocks are a good match for the given product/service: '{args.product_str}'"
        )

    # run CONSISTENCY_ROUNDS times to get a more stable result
    tasks = []
    for _ in range(CONSISTENCY_ROUNDS):
        tasks.append(
            filter_stocks_round2(
                product_str=args.product_str,
                filtered_stocks1=filtered_stocks1,
                stock_description_map=stock_description_map,
                must_include_stocks_prompt=must_include_stocks_prompt,
                llm_gpt4=llm_gpt4,
            )
        )
    filtered_stocks2_dict_list: List[Dict[StockID, str]] = await gather_with_concurrency(
        tasks, n=FILTER_CONCURRENCY
    )

    # Count occurrences of each stock across all sets
    stock_counter = Counter(
        stock for stock_set in filtered_stocks2_dict_list for stock in stock_set
    )

    # Determine the majority threshold
    majority_threshold = CONSISTENCY_ROUNDS // 2

    # Select stocks that are mentioned in the majority of the runs
    filtered_stocks2 = [
        stock for stock, count in stock_counter.items() if count > majority_threshold
    ]
    # add explanation to the filtered stocks history
    res = []
    for stock in filtered_stocks2:
        for filtered_stocks2_dict in filtered_stocks2_dict_list:
            # find the explanation for the stock in the filtered_stocks1_dict
            if stock in filtered_stocks2_dict:
                explanation = filtered_stocks2_dict[stock]
                break
        res.append(
            stock.inject_history_entry(
                HistoryEntry(
                    explanation=explanation,
                    title=f"Connection to '{args.product_str}'",
                    task_id=context.task_id,
                )
            )
        )

    logger.info(
        (
            f"args.product_str: {args.product_str}"
            f"\nfiltered_stocks1: {[stock.company_name for stock in filtered_stocks1]}"
            f"\nfiltered_stocks2: {[stock.company_name for stock in filtered_stocks2]}"
        )
    )
    return res


# Helper functions
async def filter_stocks_round1(
    product_str: str,
    stocks: List[StockID],
    stock_description_map: Dict[StockID, str],
    must_include_stocks_prompt: str,
    llm_gpt35: GPT,
    llm_gpt4: GPT,
) -> Set[StockID]:
    # get product description from GPT
    product_description = await llm_gpt4.do_chat_w_sys_prompt(
        main_prompt=PRODUCT_DESCRIPTION_PROMPT.format(product_str=product_str),
        sys_prompt=NO_PROMPT,
    )
    # create GPT call tasks
    tasks = []
    filtered_stock_set: Set[StockID] = set()
    for stock in stocks:
        # check token length
        main_prompt = STOCK_PRODUCT_FILTER1_MAIN_PROMPT.format(
            must_include_stocks_prompt=must_include_stocks_prompt,
            separator=SEPARATOR,
            company_description=stock_description_map[stock],
            product_description=product_description,
        )
        main_prompt_token_size = GPTTokenizer(CHEAP_LLM).get_token_length(
            input=main_prompt.filled_prompt
        )
        # if token size is less than CHEAP_LLM, use CHEAP_LLM, else use GPT4_O
        if main_prompt_token_size < MAX_TOKENS[CHEAP_LLM]:
            tasks.append(
                llm_gpt35.do_chat_w_sys_prompt(
                    main_prompt=main_prompt,
                    sys_prompt=NO_PROMPT,
                )
            )
        else:
            tasks.append(
                llm_gpt4.do_chat_w_sys_prompt(
                    main_prompt=main_prompt,
                    sys_prompt=NO_PROMPT,
                )
            )
    results = await gather_with_concurrency(tasks, n=FILTER_CONCURRENCY)
    for stock, result in zip(stocks, results):
        if "yes" in result.lower():
            filtered_stock_set.add(stock)
    return filtered_stock_set


async def filter_stocks_round2(
    product_str: str,
    filtered_stocks1: List[StockID],
    stock_description_map: Dict[StockID, str],
    must_include_stocks_prompt: str,
    llm_gpt4: GPT,
) -> Dict[StockID, str]:
    # get product description from GPT
    product_description = await llm_gpt4.do_chat_w_sys_prompt(
        main_prompt=PRODUCT_DESCRIPTION_PROMPT.format(product_str=product_str),
        sys_prompt=NO_PROMPT,
    )
    tasks = []
    filtered_stock_dict: Dict[StockID, str] = {}
    for stock in filtered_stocks1:
        tasks.append(
            llm_gpt4.do_chat_w_sys_prompt(
                STOCK_PRODUCT_FILTER2_MAIN_PROMPT.format(
                    must_include_stocks_prompt=must_include_stocks_prompt,
                    separator=SEPARATOR,
                    company_description=stock_description_map[stock],
                    product_description=product_description,
                ),
                sys_prompt=NO_PROMPT,
            )
        )
    results = await gather_with_concurrency(tasks, n=FILTER_CONCURRENCY)
    for stock, result in zip(filtered_stocks1, results):
        explanation, _ = result.split(SEPARATOR)
        if "yes" in result.lower():
            filtered_stock_dict[stock] = explanation.strip()
    return filtered_stock_dict
