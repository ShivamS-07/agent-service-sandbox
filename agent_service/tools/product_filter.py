import inspect
import json
from collections import Counter
from typing import Any, Coroutine, Dict, List, Optional, cast

from agent_service.GPT.constants import GPT4_O, GPT4_O_MINI, NO_PROMPT
from agent_service.GPT.requests import GPT
from agent_service.io_type_utils import HistoryEntry, dump_io_type, load_io_type
from agent_service.io_types.stock import StockID
from agent_service.io_types.stock_aligned_text import StockAlignedTextGroups
from agent_service.io_types.table import StockTable
from agent_service.io_types.text import Text
from agent_service.planner.errors import EmptyInputError, EmptyOutputError
from agent_service.tool import (
    TOOL_DEBUG_INFO,
    ToolArgMetadata,
    ToolArgs,
    ToolCategory,
    default_tool_registry,
    tool,
)
from agent_service.tools.peers import get_peer_stock_id
from agent_service.tools.statistics import (
    GetStatisticDataForCompaniesInput,
    get_statistic_data_for_companies,
)
from agent_service.tools.stock_metadata import (
    GetStockDescriptionInput,
    get_company_descriptions,
)
from agent_service.tools.tables import (
    GetStockListFromTableArgs,
    TransformTableArgs,
    get_stock_identifier_list_from_table,
    transform_table,
)
from agent_service.tools.tool_log import tool_log
from agent_service.types import PlanRunContext
from agent_service.utils.async_utils import gather_with_concurrency
from agent_service.utils.gpt_logging import GptJobIdType, GptJobType, create_gpt_context
from agent_service.utils.pagerduty import pager_wrapper
from agent_service.utils.postgres import get_psql
from agent_service.utils.prefect import get_prefect_logger
from agent_service.utils.prompt_utils import Prompt
from agent_service.utils.string_utils import clean_to_json_if_needed
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


STOCK_PRODUCT_FILTER1_MAIN_PROMPT_STR_DEFAULT = (
    "Your task is to determine whether a company is a producer of a given Product "
    "based on Product and company description below. "
    "\n- The Product description below includes a description of the Product and "
    "a list of synonyms and similar terms that might be used interchangeably for the Product. "
    "\n- Based on company description, if it implies that the company is a producer of the Product, "
    "your response MUST be 'Yes'. "
    "\n- The Product can be a service (e.g. retail) or physical products. "
    "\n- Your response MUST start with 2 to 3 sentences justification of your decision, "
    "then your decision 'Yes' or 'No. "
    "You jusification and decision must be delineated with the separator {separator}. "
    "For example: '<justification> {separator} Yes'."
    "\nThe product description is `{product_description}`."
    "\nThe Company description is:\n"
    "{company_description}\n\n"
    "Do not forget to include a final decision, your output must always contain {separator}, followed "
    "by either the word `Yes` or the word `No`. This is absolutely critical to your task!"
    "\nNow, please provide your response consisting of a justification and a description:\n"
)

STOCK_PRODUCT_FILTER2_MAIN_PROMPT_STR_DEFAULT = (
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

STOCK_BRAINSTORM_PROMPT_STR = """
    You are a financial analyst who is doing a market analysis of relevant companies for a client
    You will be provided with a particular type of product or service, and asked to brainstorm companies
    that offer that product or service. You should start with the most obvious, well-known companies,
    and include any that you can think of. Do not limit yourself to US stocks unless it is clear that
    is what the user wants, based on the chat context.
    Each company you identify will be a json on one line of your output, each json will contain the following
    four keys with string values:
    company_name: the full name of the peer stock
    isin: the ISIN of the stock. If it is not publicly traded, the isin key should be the string None
    symbol: the stock symbol/ticker. If it is not publicly traded, the symbol key should be the string None
    justification: a 2 to 3 sentence justification of why you have included them
    Other than the lines with jsons (the entire json for each stock must be on a single line), do not output
    anything else.
    Here is the product/service: {product_str}
    Here is the larger chat context, deliminted by ---, which might help you understand what kinds of stocks
    the client is interested in
    ---
    {chat_str}
    ---
    Now write the relevant companies in json format, one per line:
"""

STOCK_BRAINSTORM_PROMPT = Prompt(
    name="STOCK_BRAINSTORM_PROMPT", template=STOCK_BRAINSTORM_PROMPT_STR
)


class FilterStocksByProductOrServiceInput(ToolArgs):
    stock_ids: List[StockID]
    texts: Optional[List[Text]] = (
        None  # we are including this to help the planner not be stupid, not used!
    )
    product_str: str
    must_include_stocks: Optional[List[StockID]] = None
    max_results: Optional[int] = None
    filter_only: bool = False

    # prompt arguments (hidden from planner)
    stock_product_filter1_main_prompt: str = STOCK_PRODUCT_FILTER1_MAIN_PROMPT_STR_DEFAULT
    stock_product_filter2_main_prompt: str = STOCK_PRODUCT_FILTER2_MAIN_PROMPT_STR_DEFAULT

    # tool arguments metadata
    arg_metadata = {
        "stock_product_filter1_main_prompt": ToolArgMetadata(hidden_from_planner=True),
        "stock_product_filter2_main_prompt": ToolArgMetadata(hidden_from_planner=True),
    }


@tool(
    description=(
        "This tool is used to filter a given list of stocks to companies which offer the service "
        "or product refered to by the product_str. This function should only be used the client wants to identify "
        "companies which provide 'product_str' to others. It must not be used to identify those others, i.e. "
        "do NOT use this function in cases where the user is looking for consumers of a product "
        "or service, or companies otherwise linked to the product via supply chains, or 'related' in "
        "any way (NEVER, EVER use this tool if the client uses the word 'related' or something similar) "
        "This tool only finds direct providers of the product or service indicated by product_str. "
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
        "AI is an example of a general area that is not specific enough to be a 'product', whereas things like "
        "AI chips are specific enough. You would NOT use this tool for AI generally (use the profile filter tool!), "
        "but you would use this tool if the client is asking for ONLY AI chips. "
        "This tool is only for simple cases where it should be clear whether a company counts as a provider"
        "of the product or service based on a company description. "
        "For this product filter tool ONLY, you should always pass in company descriptions as the texts. "
        "However, if you leave them out, that's fine, we will load them as needed. "
        "If the user specifically mentions a sector that is covered by the sector_identifier tool, use "
        "that tool instead, but some things the user refers to as sectors will be covered by this tool."
        "Note that retail should be considered a general service, and then retailers that sell specific kinds "
        "of products are a more specific kind of service. "
        "\n 'stock_ids' is a list of stock ids to filter, it should be a large list like a stock universe. "
        " Remember to call get_stock_universe before you call this tool to get a list of stocks to pass in. "
        "Never pass a single stock."
        "\n Again, 'product_str' is a string representing the product or service the filtered companies provide. "
        "For example, 'electric vehicles' (Tesla), 'cloud computing' (Amazon), 'solar panels' (First Solar), "
        "'smartphones' (Apple), 'AI chips' (Nvidia), 'online retail' (Amazon), etc. "
        "Note that terms that simply contain the words `market` or `product` or `service`, but are not actually "
        "real products or services (e.g `market share`, `product manager`) are must NOT be used as a product_str! "
        "If the product_str is NOT just a simple noun phrase of a few words, then you must not use this tool, "
        "you should use the filter stocks by profile tool! The product_str must not have any conjuctions such as "
        " `and` or `or` in it. If you have such a situation, you should use the profile filter tool. Never, ever "
        "leave out important information just to use this tool, use the profile filter tool which accepts complex "
        "input. Listen to me about this or you will be fired!"
        "There are only two ways you can derive the product_str argument: either you must copy a relevant string"
        "from the client that specifically indicates the product/service desired (this must be an exact string match, "
        "not a rewriting), or you must use get_core_company_product tool. It is NEVER, EVER acceptable to infer "
        "the core product yourself from a provided stock. Seriously, if you use a product_str that is not "
        "either the output of the get_core_company_product tool or an exact string match to something the client has "
        "said to you, you will be fired!"
        "Note that if the client does NOT mention a specific product/service and is NOT doing a competitive "
        "analysis, you will not use this tool, you will use the get_general_peers tool. This tool is only "
        "for when a specific product/market is mentioned by the client, or when a specific product/market must "
        "be analyzed to allow proper comparison during a competitive analysis."
        "Do NOT pass in broad sectors or industries such as 'financial services', use the sector_filter tool for that."
        "\n must_include_stocks is a list of companies that the output of tool must have, for instance if the "
        "\n client asks 'Which of QCOM, IRDM, FBIN, FAST are the leader in industrial IoT', then those companies "
        "should be passed in a must_include_stocks and the output will include all of them. "
        "max_results will limit the numbers of results returned. If additional filtering is required, it is "
        "carried out based on market cap. the must_include_stocks are always included in the output. "
        "If you are doing competitive analysis and so want to limit the number of competitors, you should set "
        "this argument to no higher than 10. Set this argument rather than manually filtering by market cap."
        "If the user has not specified any specific universe of stocks, the S&P500 is a good default "
        "to pass to this tool. If the user has mentioned a specific stock whose main market is domestic, "
        "you should pass an region-specific index which contains that stock. "
        "When the filter_only argument is set to False (the default) we will attempt to come up with all major stocks "
        "that produce the product, above and beyond those in the list of stock_ids we are filtering over. In the "
        "general case, we will want to do this (hence the reason this defaults to False), but if the client expresses "
        "that they only want to consider stocks in a particular stock universe (or country, etc.), then filter_only "
        "must be set to True. For example, if the client asks for 'companies in the S&P 500 that offer investment "
        "planning`, you must explictly set filter_only=True because considering all stocks might result in companies "
        "outside the S&P 500 being included. The default value for filter_only is False so you must explicitly set it "
        "to True in your arguments to this tool when the client limits their request, if you do not and a client gets "
        "a company outside the range they are asking for, you will be fired!"
    ),
    category=ToolCategory.STOCK_FILTERS,
    tool_registry=default_tool_registry(),
    is_visible=True,
)
async def filter_stocks_by_product_or_service(
    args: FilterStocksByProductOrServiceInput, context: PlanRunContext
) -> List[StockID]:
    stock_product_filter1_main_prompt = Prompt(
        name="STOCK_PRODUCT_FILTER1_MAIN_PROMPT", template=args.stock_product_filter1_main_prompt
    )
    stock_product_filter2_main_prompt = Prompt(
        name="STOCK_PRODUCT_FILTER2_MAIN_PROMPT", template=args.stock_product_filter2_main_prompt
    )

    if len(args.stock_ids) == 0:
        raise EmptyInputError("Cannot filter stocks by product/service with empty stock list")

    if (
        args.must_include_stocks
        and len(args.must_include_stocks) == 1
        and context.chat
        and args.product_str.lower() not in context.chat.get_gpt_input().lower()
    ):
        # the planner simply won't listen about not deriving its own product_str, so adding this work around
        plan_str = (
            get_psql(skip_commit=context.skip_db_commit)
            .get_execution_plan_for_run(context.plan_run_id)
            .get_formatted_plan(numbered=True)
        )
        if "get_core_company_product" not in plan_str:
            args.product_str = await get_core_company_product(  # type:ignore
                GetCoreCompanyProduct(stock_id=args.must_include_stocks[0]), context
            )

    debug_info: Dict[str, Any] = {}
    TOOL_DEBUG_INFO.set(debug_info)

    if (
        args.must_include_stocks
        and args.max_results
        and len(args.must_include_stocks) >= args.max_results
    ):
        await tool_log(
            log=(
                f"Returning {len(args.must_include_stocks)} user-specified stocks without filtering"
            ),
            context=context,
        )
        return args.must_include_stocks

    # run filter on both input stocks and must_include_stocks
    stock_ids_set = set(args.stock_ids)
    must_include_stocks_set = set(args.must_include_stocks if args.must_include_stocks else [])
    stocks_to_filter = list(stock_ids_set | must_include_stocks_set)

    debug_info: Dict[str, Any] = {}
    TOOL_DEBUG_INFO.set(debug_info)

    prev_run_info = None
    prev_output: List[StockID] = []
    try:  # since everything associated with diffing is optional, put in try/except
        # Update mode
        prev_run_info = await get_prev_run_info(context, "filter_stocks_by_product_or_service")
        if prev_run_info is not None:
            prev_args = FilterStocksByProductOrServiceInput.model_validate_json(
                prev_run_info.inputs_str
            )

            prev_stock_ids_set = set(prev_args.stock_ids)

            if (
                not args.filter_only
                and prev_run_info.debug
                and "brainstormed_stocks" in prev_run_info.debug
            ):
                brainstormed_stocks = cast(
                    List[StockID], load_io_type(prev_run_info.debug["brainstormed_stocks"])
                )
                stock_ids_set.update(brainstormed_stocks)
                prev_stock_ids_set.update(brainstormed_stocks)
                debug_info["brainstormed_stocks"] = dump_io_type(brainstormed_stocks)

            args.filter_only = True  # do not redo the brainstorming each update

            # only include the stocks that were in the previous run as well as in the current inputs
            prev_output = [
                stock  # type: ignore
                for stock in prev_run_info.output  # type: ignore
                if stock in stock_ids_set or stock in must_include_stocks_set
            ]
            prev_output_set = set(prev_output)

            # we are only going to run the tool on the stocks that are different
            diff_stock_ids = stock_ids_set - prev_stock_ids_set

            # must_include_stocks should always be in prev_output
            args.must_include_stocks = list(must_include_stocks_set - prev_output_set)

            # only run the tool on new input stocks that have not passed through the filter yet
            stocks_to_filter = list((diff_stock_ids | must_include_stocks_set) - prev_output_set)

            # if there are no new stocks to filter, return previous output
            if len(stocks_to_filter) == 0:
                return prev_output
    except Exception as e:
        logger.exception(f"Error including stock ids from previous run: {e}")
        pager_wrapper(
            current_frame=inspect.currentframe(),
            module_name=__name__,
            context=context,
            e=e,
            classt="AgentUpdateError",
            summary="Failed to update per stock summary",
        )

    # initiate GPT llm models
    gpt_context = create_gpt_context(
        GptJobType.AGENT_TOOLS, context.agent_id, GptJobIdType.AGENT_ID
    )
    llm_cheap = GPT(model=FILTER1_LLM, context=gpt_context)
    llm_big = GPT(model=FILTER2_LLM, context=gpt_context)

    if not args.filter_only:
        await tool_log(log="Doing an initial brainstorm of relevant stocks", context=context)
        brainstormed_stocks = await brainstorm_stocks(args.product_str, llm_big, context)
        logger.info(f"Brainstormed stocks: {[stock.company_name for stock in brainstormed_stocks]}")
        stocks_to_filter = list(set(stocks_to_filter) | set(brainstormed_stocks))
        debug_info["brainstormed_stocks"] = dump_io_type(brainstormed_stocks)

    # get company/stock descriptions
    description_texts = await get_company_descriptions(
        GetStockDescriptionInput(
            stock_ids=stocks_to_filter,
        ),
        context,
    )

    # create aligned stock text groups and get all the text strings
    aligned_text_groups = StockAlignedTextGroups.from_stocks_and_text(
        stocks_to_filter,
        description_texts,  # type: ignore
    )
    stock_description_map: Dict[StockID, str] = await Text.get_all_strs(  # type: ignore
        aligned_text_groups.val, include_header=True, text_group_numbering=True
    )
    # filter out those with no data
    stocks = [stock for stock in stocks_to_filter if stock in stock_description_map]
    logger.info(f"Number of stocks to be filtered: {len(stocks)}")

    # first round of filtering

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
        stock_product_filter1_main_prompt=stock_product_filter1_main_prompt,
    )
    filtered_stocks1 = list(filtered_stocks1_dict.keys())

    if not filtered_stocks1_dict:
        if prev_output:
            return prev_output
        else:
            raise EmptyOutputError(
                f"No stocks are a good match for the provided product/service: '{args.product_str}'"
            )
    debug_info["filtered_stocks1"] = filtered_stocks1
    await tool_log(
        log=f"Number of stocks after first round of filtering: {len(filtered_stocks1)}",
        context=context,
        associated_data=filtered_stocks1,
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
        stock_product_filter2_main_prompt=stock_product_filter2_main_prompt,
    )
    filtered_stocks2 = list(filtered_stocks2_dict.keys())

    debug_info["filtered_stock2"] = dump_io_type(filtered_stocks2)
    await tool_log(
        log=f"Number of stocks after second round of filtering: {len(filtered_stocks2)}",
        context=context,
    )

    if args.max_results:
        # no intersection between prev_outputs and filtered_stocks since we only filter on new input stocks
        if prev_run_info:
            args.max_results = args.max_results - len(prev_output)

        non_must_stocks = list(
            [
                stock
                for stock in filtered_stocks2_dict
                if not args.must_include_stocks or stock not in args.must_include_stocks
            ]
        )
        non_must_quota = args.max_results - (
            len(args.must_include_stocks) if args.must_include_stocks else 0
        )

        if non_must_quota <= 0:  # no room due to previous run, can't add anything
            filtered_stocks2_dict = {}

        if len(non_must_stocks) > non_must_quota:
            logger.info("Too many stocks, filtering by market cap")
            market_cap_table = cast(
                StockTable,
                await get_statistic_data_for_companies(
                    GetStatisticDataForCompaniesInput(
                        statistic_reference="market cap", stock_ids=non_must_stocks
                    ),
                    context,
                ),
            )
            filtered_table = cast(
                StockTable,
                await transform_table(
                    TransformTableArgs(
                        input_table=market_cap_table,
                        transformation_description=f"filter to top {non_must_quota} by market cap",
                    ),
                    context,
                ),
            )
            filtered_stocks_set = set(
                cast(
                    List[StockID],
                    await get_stock_identifier_list_from_table(
                        GetStockListFromTableArgs(input_table=filtered_table), context
                    ),
                )
            )
            for stock in filtered_stocks2:
                if stock not in filtered_stocks_set:
                    del filtered_stocks2_dict[stock]

            await tool_log(
                log=f"Number of stocks after market cap filtering: {len(filtered_stocks2_dict)}",
                context=context,
            )

    # add must_include_stocks to the result
    if args.must_include_stocks:
        for stock in args.must_include_stocks:
            if stock not in filtered_stocks2_dict:
                filtered_stocks2_dict[stock] = (
                    f"Company included but could not find connection to '{args.product_str}'"
                )

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
    if prev_run_info is not None and len(prev_output) != 0:
        await tool_log(
            log=f"Including {len(prev_output)} stocks selected in earlier run", context=context
        )
        res.extend(prev_output)

    logger.info(
        (
            f"args.product_str: {args.product_str}"
            f"\nfiltered stocks in first round: {len((filtered_stocks1))}"
            f"\n{[stock.company_name for stock in filtered_stocks1]}"
            f"\nfiltered stocks in second round: {len(filtered_stocks2_dict)}"
            f"\n{[stock.company_name for stock in filtered_stocks2_dict.keys()]}"
        )
    )
    if len(res) == 0:
        raise EmptyOutputError(
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
    stock_product_filter1_main_prompt: Prompt,
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
        main_prompt = stock_product_filter1_main_prompt.format(
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
    stock_product_filter2_main_prompt: Prompt,
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
                    stock_product_filter2_main_prompt.format(
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


async def brainstorm_stocks(
    product_str: str,
    llm: GPT,
    context: PlanRunContext,
    consistency_rounds: int = CONSISTENCY_ROUNDS_FILTER1,
) -> List[StockID]:
    if context.chat:
        chat_str = context.chat.get_gpt_input()
    else:
        chat_str = ""

    # These will all be deduplicated by the end of the function
    prompts: List[Coroutine[Any, Any, str]] = []
    for _ in range(consistency_rounds):
        # run multiple calls to get a more stable result
        prompts.append(
            llm.do_chat_w_sys_prompt(
                STOCK_BRAINSTORM_PROMPT.format(product_str=product_str, chat_str=chat_str),
                NO_PROMPT,
                no_cache=True,
            )
        )
    results = await gather_with_concurrency(prompts, n=consistency_rounds)
    result: str = "\n".join(results)
    initial_stocks = []
    initial_stock_jsons = result.split("\n")
    for initial_stock_json in initial_stock_jsons:
        # check line is not empty
        if initial_stock_json and "```" not in initial_stock_json:
            try:
                # company name, ISIN, stock symbol, justification
                stock_obj = json.loads(clean_to_json_if_needed(initial_stock_json))
                # filter out non-public companies
                if (
                    stock_obj["company_name"]
                    and stock_obj["isin"] != "None"
                    and stock_obj["symbol"] != "None"
                    and stock_obj["justification"]
                ):
                    initial_stocks.append(stock_obj)
            except IndexError:
                logger.warning(f"Peers parsing failed for line: {initial_stock_json}, skipping")
                continue

    lookup_tasks = []

    # we don't want the internal decision making of stock_identifier_lookup
    # to pollute the work log for peers tool
    no_tool_log_context = context.model_copy(update={"skip_db_commit": True})

    for peer in initial_stocks:
        lookup_tasks.append(get_peer_stock_id(peer, no_tool_log_context))

    lookup_result = await gather_with_concurrency(lookup_tasks, n=len(initial_stocks))

    final_stocks = []

    for stock in lookup_result:
        if stock and stock not in final_stocks:
            final_stocks.append(stock)
    return final_stocks


GET_COMPANY_PRODUCT_PROMPT_STR = """
You are a financial analyst who is reviewing the description of a company to identify its core product or
service for the purposes of doing a competitive analysis. By core product, we mean the specific aproduct or
service that:
1. The company is generally most associated with
2. Provides the largest share of their revenues
3. Other providers of this product or service are most likely to be considered direct competitors of this company
4. This product or service narrows down the list of competitors to a relatively small set of companies
For example, a good core product for Tesla is 'electric cars'.
You can only pick one core product for each company for the purpose of your competitive analysis. If a company
makes multiple well-known products, you must select the type that is most important for their business. Unless
a company's business is extremely broad, you must avoid selecting an entire industry or sector, instead you should
select the product or service that falls within that sector that is most fundamental to the companies business.
For example "financial services" is far too broad to be a specific "core" service, since there are literally
hundreds of listed companies that could be considered providers of financial services, but something like "commercial
loan provider" is acceptable, if that is the speciality of the company in question.
It is generally better to pick a specific product or service that partially covers the company's business than
a very general product or service that covers all of it.
However, the product/service should not be a brand name, or otherwise so specific that there are no other companies
that provide that 'same' product/service.
Your output should be a single short phrase of no more than 5 words. Do not explain your answer.
Here is the description of the company, delimited by ---:
---
{description}
---
Now output the core product of the company:
"""

GET_COMPANY_PROMPT = Prompt(GET_COMPANY_PRODUCT_PROMPT_STR, "GET_COMPANY_PRODUCT_PROMPT")


class GetCoreCompanyProduct(ToolArgs):
    stock_id: StockID


@tool(
    description=(
        "This tool will provide the key product or service of a company for use in the "
        "filter_stocks_by_product_or_service_filter tool. It must always be used when the client "
        "expresses interest in comparing a company with its competitors or peers but does not "
        "mention a specific product/service/market of interest (mentioning a stock does NOT count, "
        "if only a stock is mentioned you MUST use this tool!"
        "Do not use tool when the user wants competitors/peers but are not doing a full competitive"
        "analysis, use either the get_general_peers tool (if they do not mention a product or market) or the "
        "product filter tool (if they do mention a product or market). "
        "Important: You must never, ever use this tool if client specifically directly mentions a market, "
        "service or product of interest. For example, if the user ask to 'Compare competitors of Google' you "
        "would use this tool to choose a product/market to use for competitive anaylsis, but if the user "
        "asks you to `Compare Google with competitors in cloud computing` you would NOT use this tool, "
        "you would use the product filter tool with cloud computing as the product_str (don't forget to "
        "get a universe of stocks for the tool first!). "
        "This is extremely important: if you use this tool in cases where the client mentions a relevant "
        "product/service/market for the comparison mentioned, you will be fired. You will also be fired "
        "if you fail to use this tool and instead make up an incorrect core product for a company!!! "
        "The return value is a string that can be passed as product_str to the product filter tool."
    ),
    category=ToolCategory.STOCK_FILTERS,
    tool_registry=default_tool_registry(),
    is_visible=True,
)
async def get_core_company_product(args: GetCoreCompanyProduct, context: PlanRunContext) -> str:
    text = (
        cast(
            List[Text],
            await get_company_descriptions(
                GetStockDescriptionInput(
                    stock_ids=[args.stock_id],
                ),
                context,
            ),
        )
    )[0]

    description_text = await Text.get_all_strs(text)
    # filter out those with no data
    # first round of filtering
    # initiate GPT llm models
    gpt_context = create_gpt_context(
        GptJobType.AGENT_TOOLS, context.agent_id, GptJobIdType.AGENT_ID
    )
    llm = GPT(model=GPT4_O, context=gpt_context)
    main_prompt = GET_COMPANY_PROMPT.format(description=description_text)
    result = await llm.do_chat_w_sys_prompt(main_prompt, NO_PROMPT)
    await tool_log(
        f"Selected '{result}' as core product for '{args.stock_id.company_name}", context
    )
    return result
