from collections import Counter
from typing import Any, Dict, List, Optional

from agent_service.GPT.constants import GPT4_O, GPT4_O_MINI, NO_PROMPT
from agent_service.GPT.requests import GPT
from agent_service.io_type_utils import HistoryEntry, dump_io_type
from agent_service.io_types.stock import StockID
from agent_service.io_types.stock_aligned_text import StockAlignedTextGroups
from agent_service.io_types.text import Text
from agent_service.planner.errors import NonRetriableError
from agent_service.tool import (
    TOOL_DEBUG_INFO,
    ToolArgs,
    ToolCategory,
    ToolRegistry,
    tool,
)
from agent_service.tools.stock_metadata import (
    GetStockDescriptionInput,
    get_company_descriptions,
)
from agent_service.tools.tool_log import tool_log
from agent_service.types import PlanRunContext
from agent_service.utils.async_utils import gather_with_concurrency
from agent_service.utils.gpt_logging import GptJobIdType, GptJobType, create_gpt_context
from agent_service.utils.prefect import get_prefect_logger
from agent_service.utils.prompt_utils import Prompt
from agent_service.utils.tool_diff import get_prev_run_info

SEPARATOR = "###"
FILTER1_LLM = GPT4_O_MINI  # can't be GPT3.5 because it may pass the token limit
FILTER2_LLM = GPT4_O
logger = get_prefect_logger(__name__)
# Number of rounds to perform GPT call and get results based on majority
CONSISTENCY_ROUNDS_FILTER1 = 3
CONSISTENCY_ROUNDS_FILTER2 = 3
STOCK_NUM_THRESHOLD = 1000
CUNCURRENT_CALLS_FILTER1 = 300
CUNCURRENT_CALLS_FILTER2 = 300


STOCK_PRODUCT_FILTER1_MAIN_PROMPT = Prompt(
    name="STOCK_PRODUCT_FILTER1_MAIN_PROMPT",
    template=(
        "Your task is to determine whether a company is a producer of a given Product "
        "based on Product and company description below. "
        "\n- The Product description below includes a description of the Product and "
        "a list of synonyms and similar terms that might be used interchangeably for the Product. "
        "\n- Based on company description, if it implies that the company is a producer of the Product, "
        "your response MUST be 'YES'. "
        "\n- The Product can be a service (e.g. retail) or physical products. "
        "\n- Your response MUST start with 2 to 3 sentences justification of your decsion, "
        "then your decision 'YES' or 'NO', all delineated with the separator {separator}. "
        "For example: '<justification> {separator} YES/NO'."
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
        "Your task is to determine whether a given company is a producer of a given Product, "
        "based on the Product and the company description below. "
        "\n- The Product description below includes a description of the Product and "
        "a list of synonyms and similar terms that might be used interchangeably for the Product. "
        "\n\nSome guidelines to consider: "
        "\n- If the company produce components or services that are used in the production "
        "of the Product, and not producing the product itself the decision MUST be 'NO'. "
        "\n- Product can be service (e.g. retail) or physical product. "
        "\n- Be very strict in your decision, ONLY if the company is a direct producer of the Product, "
        "the decision can be 'YES', Otherwise, the decision MUST be 'NO'. "
        "\n- Your response MUST start with 2 to 3 sentences justification of your decsion, "
        "then your decision 'YES' or 'NO', all delineated with the separator {separator}. "
        "For example: '<justification> {separator} YES/NO'."
        "\n### Product Description\n"
        "{product_description}"
        "\n### Company Description\n"
        "{company_description}"
        "\n\nNow, please provide your response."
    ),
)


PRODUCT_DESCRIPTION_PROMPT = Prompt(
    name="PRODUCT_DESCRIPTION_PROMPT",
    template=(
        "Please provide a short description and all synonyms and similar terms that might be used interchangeably for "
        "the given product or service below. "
        "\n- Your response should be two parts: 1) a short description of the product or service, and 2) a list of "
        "synonyms and similar terms. "
        "Ensure your response is concise and less than 500 words."
        "\n\n### Product: {product_str}\n"
        "\nNow, please provide your response."
    ),
)


class FilterStocksByProductOrServiceInput(ToolArgs):
    stock_ids: List[StockID]
    texts: List[Text]  # we are including this to help the planner not be stupid, not used!
    product_str: str
    must_include_stocks: Optional[List[StockID]] = None


@tool(
    description=(
        "This tool is used to filter a given list of stocks to companies which offer the service "
        "or product refered to by the product_str. This function should only be used the client wants to identify "
        "companies which provide 'product_str' to others. It must not be used to identify those others, i.e. "
        "do NOT use this function in cases where the user is looking for consumers of a product "
        "or service, or companies otherwise linked to the product via supply chains. This tool only "
        "finds suppliers of the product or service indicated by product_str. "
        "Do NOT use this function if the client is asking for filtering based on properties more "
        "complicated than just what specific product or service they sell, even when products or services "
        "are mentioned. For every company X selected by this function, it will be true that "
        "`Company X offers product_str`, if that is NOT what the user is asking for, DO NOT USE THIS FUNCTION! "
        "You must never, ever use this function if someone asks the equivalent of `who uses X in their products?` "
        "If the client need is more complicated that finding companies that are the providers of the specific "
        "products or services indicated by product_str, use filter_stocks_by_profile tool instead. If you are "
        "at all unsure, use the filter_stocks_by_profile tool. "
        "You must always get a list of stocks using a function such as get_stock_universe to pass in as the stock_ids"
        "You should NEVER pass an empty list, or a list with only one or a handful of stocks. The purpose of this "
        "tool is to filter a large list of stocks. If a specific stock is mentioned in the request, you will often "
        "want to include it in the `must_include_stocks` instead."
        "This tool is only for simple cases where it should be clear whether a company counts as a provider"
        "of the product or service based on a company description. "
        "For this product filter tool ONLY, you should always pass in company descriptions as the texts. "
        "If the user specifically mentions a sector that is covered by the sector_identifier tool, use "
        "that tool instead, but some things the user refers to as sectors will be covered by this tool."
        "Note that retail should be considered a general service, and then retailers that sell specific kinds "
        "of products are a more specific kind of service. "
        "\n 'stock_ids' is a list of stock ids to filter, it should be a large list like a stock universe. "
        "Never pass a single stock."
        "\n Again, 'product_str' is a string representing the product or service the filtered companies provide. "
        "For example, 'electric vehicles' (Tesla), 'cloud computing' (Amazon), 'solar panels' (First Solar), "
        " 'smartphones' (Apple), 'AI chips' (Nvidia), 'online retail' (Amazon), etc. "
        "\n must_include_stocks is a list of companies that the output of tool must have, for instance if the "
        "\n client asks 'Which of QCOM, IRDM, FBIN, FAST are the leader in industrial IoT', then those companies "
        "should be passed in a must_include_stocks and the output will include all of them. "
        "If the user has not specified any specific universe of stocks, the S&P500 is a good default "
        "to pass to this tool. If the user has mentioned a specific stock whose main market is domestic, "
        "you should pass an region-specific index which contains that stock. "
        "However, if the client has not mentioned a region and either there is a mention of a stock "
        "which has an major international presence, or the product market involves major international "
        "competition, you should instead use the Vanguard World Stock ETF instead of the S&P 500, for example "
        "if the product_str was cars, you should use the world stocks because of major Japanese manufacturers "
        "like Toyota, Honda, etc."
    ),
    category=ToolCategory.STOCK,
    tool_registry=ToolRegistry,
    is_visible=True,
)
async def filter_stocks_by_product_or_service(
    args: FilterStocksByProductOrServiceInput, context: PlanRunContext
) -> List[StockID]:
    debug_info: Dict[str, Any] = {}
    TOOL_DEBUG_INFO.set(debug_info)

    prev_run_info = None
    try:  # since everything associated with diffing is optional, put in try/except
        prev_run_info = await get_prev_run_info(context, "filter_stocks_by_profile_match")
        if prev_run_info is not None:
            prev_args = FilterStocksByProductOrServiceInput.model_validate_json(
                prev_run_info.inputs_str
            )

            # only include the stocks that were in the previous run as well as in the current input
            stock_ids_set = set(args.stock_ids)
            prev_output: List[StockID] = [
                stock for stock in prev_run_info.output if stock in stock_ids_set  # type:ignore
            ]

            # start by finding the differences in the input stocks
            diff_stock_ids = stock_ids_set - set(prev_args.stock_ids)

            # we are only going to run the tool on the stocks that are different
            args.stock_ids = list(diff_stock_ids)

    except Exception as e:
        logger.warning(f"Error including stock ids from previous run: {e}")

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
    logger.info(f"Number of stocks to be filtered: {len(stocks)}")

    # first round of filtering
    # initiate GPT llm models
    gpt_context = create_gpt_context(
        GptJobType.AGENT_TOOLS, context.agent_id, GptJobIdType.AGENT_ID
    )
    llm_cheap = GPT(model=FILTER1_LLM, context=gpt_context)
    llm_big = GPT(model=FILTER2_LLM, context=gpt_context)

    # run CONSISTENCY_ROUNDS times to get a more stable result
    await tool_log(
        log="Filtering round 1...",
        context=context,
    )

    # filter stocks in round 1
    if STOCK_NUM_THRESHOLD < len(stocks):
        consistency_rounds = 1
    else:
        consistency_rounds = CONSISTENCY_ROUNDS_FILTER1

    filtered_stocks1_dict: Dict[StockID, str] = await filter_stocks_round1(
        product_str=args.product_str,
        stocks=stocks,
        stock_description_map=stock_description_map,
        llm_cheap=llm_cheap,
        llm_big=llm_big,
        consistency_rounds=consistency_rounds,
    )
    filtered_stocks1 = list(filtered_stocks1_dict.keys())

    if not filtered_stocks1_dict:
        raise ValueError(
            f"No stocks are a good match for the given product/service: '{args.product_str}'"
        )
    debug_info["filtered_stocks1"] = filtered_stocks1
    await tool_log(
        log=(f"Number of stocks after first round of filtering: {len(filtered_stocks1)}"),
        context=context,
    )
    await tool_log(
        log="Filtering round 2...",
        context=context,
    )
    # filter stocks in round 2
    filtered_stocks2_dict: Dict[StockID, str] = await filter_stocks_round2(
        product_str=args.product_str,
        stocks=filtered_stocks1,
        stock_description_map=stock_description_map,
        llm_big=llm_big,
    )
    filtered_stocks2 = list(filtered_stocks2_dict.keys())

    debug_info["filtered_stock2"] = dump_io_type(filtered_stocks2)
    await tool_log(
        log=(f"Number of stocks after second round of filtering: {len(filtered_stocks2)}"),
        context=context,
    )
    # add must_include_stocks to the result
    if args.must_include_stocks:
        for stock in args.must_include_stocks:
            if stock not in filtered_stocks2_dict:
                filtered_stocks2_dict[stock] = "Must include company"

    # add explanation to the filtered stocks history
    res = []
    for stock in filtered_stocks2_dict:
        res.append(
            stock.inject_history_entry(
                HistoryEntry(
                    explanation=filtered_stocks2_dict[stock],
                    title=f"Connection to '{args.product_str}'",
                    task_id=context.task_id,
                )
            )
        )
    # add previous run stocks to the result
    if prev_run_info is not None:
        res.extend(prev_output)

    logger.info(
        (
            f"args.product_str: {args.product_str}"
            f"\nfiltered stocks in first round: {len((filtered_stocks1))}"
            f"\n{[stock.company_name for stock in filtered_stocks1]}"
            f"\nfiltered stocks in second round: {len(filtered_stocks2)}"
            f"\n{[stock.company_name for stock in filtered_stocks2]}"
        )
    )
    if len(res) == 0:
        raise NonRetriableError(
            message="Stock product/service filter resulted in an empty list of stocks"
        )
    return res


# Helper functions
async def filter_stocks_round1(
    product_str: str,
    stocks: List[StockID],
    stock_description_map: Dict[StockID, str],
    llm_cheap: GPT,
    llm_big: GPT,
    consistency_rounds: int = CONSISTENCY_ROUNDS_FILTER1,
) -> Dict[StockID, str]:
    # get product description from GPT
    product_description = await llm_big.do_chat_w_sys_prompt(
        main_prompt=PRODUCT_DESCRIPTION_PROMPT.format(product_str=product_str),
        sys_prompt=NO_PROMPT,
    )
    # create GPT call tasks
    tasks = []

    for stock in stocks:
        main_prompt = STOCK_PRODUCT_FILTER1_MAIN_PROMPT.format(
            separator=SEPARATOR,
            company_description=stock_description_map[stock],
            product_description=product_description,
        )
        for _ in range(consistency_rounds):
            # run multiple calls to get a more stable result
            tasks.append(
                llm_cheap.do_chat_w_sys_prompt(
                    main_prompt=main_prompt,
                    sys_prompt=NO_PROMPT,
                    no_cache=True,
                )
            )
    import time

    start_time = time.time()
    gpt_responses = await gather_with_concurrency(tasks, n=CUNCURRENT_CALLS_FILTER1)
    logger.info(f"gpt call duration in minutes: {(time.time() - start_time) / 60}")

    # separate the gpt_responses into a list of results
    results_list = [gpt_responses[i::consistency_rounds] for i in range(consistency_rounds)]

    # prepare result as mapper of filtered stocks to their explanations
    filtered_stocks1_dict: Dict[StockID, str] = {}
    company_names = set()
    for results in results_list:
        for stock, result in zip(stocks, results):
            try:
                explanation = result.split(SEPARATOR)[0]
                decision = result.split(SEPARATOR)[1]
                if "yes" in decision.lower() and stock.company_name not in company_names:
                    filtered_stocks1_dict[stock] = explanation
                    company_names.add(stock.company_name)
            except Exception as e:
                logger.error(
                    f"Error while processing result: {result} for stock: {stock} in round 1.\nError: {e}"
                )

    return filtered_stocks1_dict


async def filter_stocks_round2(
    product_str: str,
    stocks: List[StockID],
    stock_description_map: Dict[StockID, str],
    llm_big: GPT,
    consistency_rounds: int = CONSISTENCY_ROUNDS_FILTER2,
) -> Dict[StockID, str]:
    # get product description from GPT
    product_description = await llm_big.do_chat_w_sys_prompt(
        main_prompt=PRODUCT_DESCRIPTION_PROMPT.format(product_str=product_str),
        sys_prompt=NO_PROMPT,
    )
    tasks = []
    for stock in stocks:
        for _ in range(consistency_rounds):
            tasks.append(
                llm_big.do_chat_w_sys_prompt(
                    STOCK_PRODUCT_FILTER2_MAIN_PROMPT.format(
                        separator=SEPARATOR,
                        company_description=stock_description_map[stock],
                        product_description=product_description,
                    ),
                    sys_prompt=NO_PROMPT,
                    no_cache=True,
                )
            )
    gpt_responses = await gather_with_concurrency(tasks, n=CUNCURRENT_CALLS_FILTER2)
    # separate the results_all into a list of results
    results_list = [gpt_responses[i::consistency_rounds] for i in range(consistency_rounds)]

    # filter out stocks that are a good match based on the results_list
    filtered_stocks2_dict_list = []
    for results in results_list:
        filtered_stock_dict: Dict[StockID, str] = {}
        for stock, result in zip(stocks, results):
            try:
                explanation = result.split(SEPARATOR)[0]
                decision = result.split(SEPARATOR)[1]
                if "yes" in decision.lower():
                    filtered_stock_dict[stock] = explanation
            except Exception as e:
                logger.error(
                    f"Error while processing result: {result} for stock: {stock} in round 1.\nError: {e}"
                )

        filtered_stocks2_dict_list.append(filtered_stock_dict)

    # Count occurrences of each stock across all sets
    stock_counter = Counter(
        stock for stock_set in filtered_stocks2_dict_list for stock in stock_set
    )

    # Determine the majority threshold
    majority_threshold = consistency_rounds // 2

    # Select stocks that are mentioned in the majority of the runs
    filtered_stocks2 = [
        stock for stock, count in stock_counter.items() if count > majority_threshold
    ]

    # prepare result as mapper of filtered stocks to their explanations
    res = {}
    for stock in filtered_stocks2:
        for filtered_stocks2_dict in filtered_stocks2_dict_list:
            # find the explanation for the stock in the filtered_stocks1_dict
            if stock in filtered_stocks2_dict:
                res[stock] = filtered_stocks2_dict[stock]
                break

    return res
