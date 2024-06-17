import copy
import json
from typing import Any, Dict, List, Optional

from pydantic import field_validator

from agent_service.external.discover_svc_client import (
    get_score_from_rating,
    get_temporary_discover_block_data,
)
from agent_service.external.investment_policy_svc import (
    get_all_stock_investment_policies,
)
from agent_service.GPT.constants import FILTER_CONCURRENCY, GPT4_O
from agent_service.GPT.requests import GPT
from agent_service.GPT.tokens import GPTTokenizer
from agent_service.io_type_utils import HistoryEntry, Score
from agent_service.io_types.stock import StockID
from agent_service.io_types.stock_aligned_text import StockAlignedTextGroups
from agent_service.io_types.text import StockText, Text
from agent_service.tool import ToolArgs, ToolCategory, ToolRegistry, tool
from agent_service.tools.other_text import (
    GetAllTextDataForStocksInput,
    get_all_text_data_for_stocks,
)
from agent_service.tools.stocks import GetStockUniverseInput, get_stock_universe
from agent_service.tools.tool_log import tool_log
from agent_service.types import PlanRunContext
from agent_service.utils.async_utils import gather_with_concurrency, identity
from agent_service.utils.gpt_logging import GptJobIdType, GptJobType, create_gpt_context
from agent_service.utils.prefect import get_prefect_logger
from agent_service.utils.prompt_utils import Prompt
from agent_service.utils.string_utils import clean_to_json_if_needed

# Define as the template. When you want to use it, please DEEPCOPY it!
SETTINGS_TEMPLATE: Dict[str, Any] = {
    "horizon_settings": {"news_horizon": "1W"},
    "ism_settings": {
        "ism_id": None,
        "match_labels": ["Perfect Match"],
        "weight": 0.5,
    },
    "block_type": "stock",
    "universe_ids": [],
    "sector_settings": {"sector_ids": []},
    "rating_settings": {"boundary": {"lb": None, "ub": None}, "weight": 1.0},
    "strategy_settings": {  # these are exclusive
        "all_strategies": False,
        "subscribed_strategies": True,
        "strategy_ids": None,
    },
    "theme_settings": {
        "has_recent_major_developments": False,
        "recent_major_developments_days": 7,
        "themes": [],
    },
    "news_settings": {
        "sentiment_boundaries": [],
        "prev_sentiment_boundaries": [],
        "weight": 0.5,
    },
    "earnings_settings": {
        "eps": {"at_least": None, "consecutive": None, "exact": None},
        "revenue": {"at_least": None, "consecutive": None, "exact": None},
    },
    "gbi_ids": [],  # if not empty, `universe_ids` will be ignored
}

RECOMMENDATION_SYS_PROMPT_STR = """
You are a financial analyst which needs to use the provided data about a stock to provide reasoning that
supports a particular investment decision. {direction} It is critical that what you write is consistent
with this investment outlook, even if the data you are provided with is biased in a different direction.
You will be provided with some combination
of recent news articles, the latest 10K/Q SEC filing, and a summary of the recent earnings call. There will
also be a description of the company, which you should use to understand what is important to the company
but you should not use as direct evidence in your argument. In order
to support your argument, you should put forward evidence that seems the strongest based your own intuition
about the kind of event that moves the stock market as well as multiple mentions across the different
documents you have access to. You may mention data that is not consistent with your argument, but you should
minimize such evidence or discuss mitigating factors. You should only use provided data as evidence for your
argument, do not include any additional information you might know. You will also be provided which a chat
between you and your client who  wants this recommendation, if the client has mentioned any specific focus in
the context of getting recommendations, you should try to follow their instructions. For example, if the
client has specifically asked for a quantitive analysis, you should focus on individual numbers mentioned in
the documents which tell the story you are trying to tell. Your total outupt should be a medium length
paragraph of 2-4 sentences, and definitely no more than about 200 words, brevity is good provided you make a
compelling argument. Individual texts in your collection are delimited by ***, and each one starts with a
Text Number. When you have finished your argument, on the last line, you must write a list of integers and
nothing else (e.g. `[2, 5, 9]`) which corresponds to source texts for your argument. Please be selective,
list only those texts from which you directly pulled information. You should never include the description
among your sources. Important: You must not cite any sources in the body of the argument. You should only cite
your sources in this list at the end.
"""

RECOMMENDATION_MAIN_PROMPT_STR = """
Write an argument which supports an investment decision. The company in question is {company_name}. Here are the
documents about the company, delimited by ---
---
{documents}
---
Here is the chat context with your investment client. You should look at the part of the context
that discusses recommendations and integrate any particular instructions the user may have
---
{chat_context}
---
Now provide your reasoning for the investment decision in a short paragraph: """

BUY_DIRECTION = (
    "In particular, you must write an argument that focuses on evidence for buying the stock."
)
SELL_DIRECTION = "In particular, you must write an argument that focuses on evidence for selling/shorting the stock."
SCORE_DIRECTION = (
    "In particular, you should write an argument which is compatible with a score of {score} on"
    " a 0 to 1 scale. If the score is above 0.6 you should focus on evidence for investing in the"
    " stock, and if it is below 0.4 you should focus on evidence that would disuade investors."
)


RECOMMENDATION_MAIN_PROMPT = Prompt(RECOMMENDATION_MAIN_PROMPT_STR, "RECOMMENDATION_MAIN_PROMPT")
RECOMMENDATION_SYS_PROMPT = Prompt(RECOMMENDATION_SYS_PROMPT_STR, "RECOMMENDATION_SYS_PROMPT")


def map_input_to_closest_horizon(input_horizon: str, supported_horizons: List[str]) -> str:
    if input_horizon in supported_horizons:
        return input_horizon

    days_lookup = {"D": 1, "W": 7, "M": 30, "Y": 365}
    input_days = int(input_horizon[:-1]) * days_lookup[input_horizon[-1]]
    supported_horizon_to_days = {
        horizon: int(horizon[:-1]) * days_lookup[horizon[-1]] for horizon in supported_horizons
    }

    min_pair = min(supported_horizon_to_days.items(), key=lambda x: abs(x[1] - input_days))
    return min_pair[0]


async def add_scores_and_rationales_to_stocks(
    score_dict: Dict[StockID, Score], is_buy: Optional[bool], context: PlanRunContext
) -> List[StockID]:

    stocks = list(score_dict)
    texts: List[StockText] = await get_all_text_data_for_stocks(  # type: ignore
        GetAllTextDataForStocksInput(stock_ids=stocks), context
    )
    aligned_text_groups = StockAlignedTextGroups.from_stocks_and_text(stocks, texts)
    str_lookup: Dict[StockID, str] = Text.get_all_strs(  # type: ignore
        aligned_text_groups.val, include_header=True, text_group_numbering=True
    )

    gpt_context = create_gpt_context(
        GptJobType.AGENT_PLANNER, context.agent_id, GptJobIdType.AGENT_ID
    )
    llm = GPT(context=gpt_context, model=GPT4_O)
    tokenizer = GPTTokenizer(GPT4_O)
    used = tokenizer.get_token_length(
        "\n".join(
            [
                RECOMMENDATION_SYS_PROMPT.template,
                RECOMMENDATION_MAIN_PROMPT.template,
                SCORE_DIRECTION,
            ]
        )
    )

    tasks = []
    for stock, score in score_dict.items():
        text_str = str_lookup[stock]
        text_str = tokenizer.chop_input_to_allowed_length(text_str, used)
        if text_str == "":  # no text, skip
            tasks.append(identity(""))
        if is_buy is None:
            direction_str = SCORE_DIRECTION.format(score=score.val)
        elif is_buy:
            direction_str = BUY_DIRECTION
        else:
            direction_str = SELL_DIRECTION

        if context.chat:
            chat_context = context.chat.get_gpt_input()
        else:
            chat_context = ""
        tasks.append(
            llm.do_chat_w_sys_prompt(
                RECOMMENDATION_MAIN_PROMPT.format(
                    company_name=stock.company_name, documents=text_str, chat_context=chat_context
                ),
                RECOMMENDATION_SYS_PROMPT.format(direction=direction_str),
            )
        )
    results = await gather_with_concurrency(tasks, n=FILTER_CONCURRENCY)

    stocks_with_rec = []

    for result, (stock, score), text_group in zip(
        results, score_dict.items(), aligned_text_groups.val.values()
    ):

        try:
            rationale, citations = result.strip().replace("\n\n", "\n").split("\n")
            citation_idxs = json.loads(clean_to_json_if_needed(citations))
            citations = text_group.get_citations(citation_idxs)
        except ValueError:
            rationale = ""
            citations = []

        stocks_with_rec.append(
            stock.inject_history_entry(
                HistoryEntry(
                    explanation=rationale,
                    title="Recommendation",
                    score=score,
                    citations=citations,
                )
            )
        )

    return stocks_with_rec


class GetStockRecommendationsInput(ToolArgs):
    """Note: This is to find the recommended stocks from the provided stock list. It takes into
    many factors like news, ISM, ratings, etc.
    """

    stock_ids: Optional[List[StockID]] = None  # if None, we will default to use SP500
    filter: bool = True
    buy: Optional[bool] = None  # whether to get buy or sell recommendations
    horizon: str = "1M"  # 1M, 3M, 1Y
    delta_horizon: str = "1M"  # 1W, 1M, 3M, 6M, 9M, 1Y
    # news_horizon: str = "1W"  # 1W, 1M, 3M
    num_stocks_to_return: int = 5

    @field_validator("horizon", mode="before")
    @classmethod
    def validate_horizon(cls, value: Any) -> Any:
        if not isinstance(value, str):
            raise ValueError("horizon must be a string")

        return map_input_to_closest_horizon(value.upper(), supported_horizons=["1M", "3M", "1Y"])

    @field_validator("delta_horizon", mode="before")
    @classmethod
    def validate_delta_horizon(cls, value: Any) -> Any:
        if not isinstance(value, str):
            raise ValueError("delta horizon must be a string")

        return map_input_to_closest_horizon(
            value.upper(), supported_horizons=["1W", "1M", "3M", "6M", "9M", "1Y"]
        )

    # @field_validator("news_horizon", mode="before")
    # @classmethod
    # def validate_news_horizon(cls, value: Any) -> Any:
    #     if not isinstance(value, str):
    #         raise ValueError("news horizon must be a string")
    #     return map_input_to_closest_horizon(value.upper(), supported_horizons=["1W", "1M", "3M"])


@tool(
    description=(
        "This function provides stock recommendations and/or text justifying those recommendations"
        "There are two major modes, controlled by the filter boolean. If filter is on (True), the function"
        "will filter the provided stock list (or the S&P 500, if no stock list is provided) to a list of "
        "recommended buys (if buy = True) or recommended sells/shorts (if buy = False), the selection of "
        "stock for this is based on a machine learning algorithm. "
        "Note that the buy variable must NOT be None if filter is True. The num_stocks_to_return variable is "
        "used to control how many stocks are returned in filter mode (it is not used when filter is False). "
        "An example of a request that would use this function with filter=True is: "
        "`Give me 10 stocks you like/don't like`. "
        "Each stock selected with this function will include reasoning about why the stock is a buy or sell."
        "Filter should be set to False if the client is asking for recommendations for a specific, fixed set "
        " of stocks. "
        "When filter is false, this function returns the same set of stockIDs as its input, but "
        "a recommendation rationale will be included. If buy = True, a rationale for buying each stock will "
        "be added, if buy = False, a rationale for selling/shorting the stock will be added. If buy == None, "
        "then the rationale will be based on justifying the score from a machine learning algorithm. "
        "So, if a user says to `give me a reason to buy/sell NVDA`, you would use filter = False, and "
        "buy = true/false, respectively "
        "but if the user asks `Should I buy NVDA?` filter is still False, but buy should be None, and the "
        "rationale will reflect a post hoc rationalization of the machine-provided rating for the stock. "
        "Investment horizon and delta horizon are used for in the ML algorithm to decide how far into "
        "The future (investment horizon) or into the past (delta) to consider, you should increase them from the "
        "defaults only when the user express some specific interest in a longer term view. "
        "If no stock ID's are provided (which should only happen when filter=True), the S&P 500 stocks are used"
    ),
    category=ToolCategory.STOCK,
    tool_registry=ToolRegistry,
    reads_chat=True,
)
async def get_stock_recommendations(
    args: GetStockRecommendationsInput, context: PlanRunContext
) -> List[StockID]:
    # NOTE: You can't use `get_dummy()` to create a dummy context for this tool because it requires
    # an actual user ID to make the gRPC call (specifically for PA SVC)

    logger = get_prefect_logger(__name__)

    if args.stock_ids:
        if args.filter:
            if len(args.stock_ids) < args.num_stocks_to_return:
                raise ValueError(
                    "The number of stocks to return is greater than the number of stocks provided."
                )
            elif len(args.stock_ids) == args.num_stocks_to_return:
                logger.warning(
                    "The number of stocks to return is equal to the number of stocks provided. Return directly"  # noqa
                )
                return args.stock_ids
        stock_ids = args.stock_ids
    else:
        if not args.filter:
            raise Exception("Get recommended stocks called in non-filter mode but no stocks passed")
        # we perhaps can store the SP500 stocks as a log output but not for now as they are GBI IDs
        await tool_log(log="No stock IDs provided. Using S&P 500 stocks.", context=context)
        stock_ids: List[StockID] = await get_stock_universe(  # type: ignore
            args=GetStockUniverseInput(universe_name="SPDR S&P 500 ETF Trust"), context=context
        )

    ism_resp = await get_all_stock_investment_policies(context.user_id)
    ism_id = None
    if ism_resp.investment_policies:
        ism = max(ism_resp.investment_policies, key=lambda x: x.last_updated.ToDatetime())
        ism_id = ism.investment_policy_id.id
        await tool_log(log=f'Using Investment Style "{ism.name}" to search stocks', context=context)

    settings_blob = copy.deepcopy(SETTINGS_TEMPLATE)
    settings_blob["ism_settings"]["ism_id"] = ism_id
    settings_blob["gbi_ids"] = [stock.gbi_id for stock in stock_ids]
    settings_blob["ism_settings"]["match_labels"] = [
        "Perfect Match",
        "Strong Match",
        "Medium Match",
    ]

    if not args.filter:  # don't want to filter on matches
        settings_blob["ism_settings"]["match_labels"].extend(["Weak Match", "Poor Match"])

    if args.buy is None or args.buy:
        if args.filter:
            settings_blob["rating_settings"]["boundary"]["lb"] = 2.5
        else:
            settings_blob["rating_settings"]["boundary"]["lb"] = 0.0001
        # settings_blob["news_settings"]["sentiment_boundaries"] = [{"lb": 0, "ub": 1}]
    else:
        if args.filter:
            settings_blob["rating_settings"]["boundary"]["ub"] = 2.5
        else:
            settings_blob["rating_settings"]["boundary"]["ub"] = 4.9999
        # settings_blob["news_settings"]["sentiment_boundaries"] = [{"lb": -1, "ub": 0}]

    await tool_log(log=f"Getting stock ratings for {len(stock_ids)} stocks", context=context)

    resp = await get_temporary_discover_block_data(
        context.user_id, settings_blob, args.horizon, args.delta_horizon
    )
    if args.filter and len(resp.rows) < args.num_stocks_to_return:
        # TODO: We can loose the constraints, but this is already very loose. What to do?
        raise ValueError(
            f"Cannot find enough stocks to meet the requirement: {len(resp.rows)} / {args.num_stocks_to_return}"
        )

    logger.info(f"Got scores for {len(resp.rows)} stocks")

    if args.filter:
        rows = resp.rows[: args.num_stocks_to_return]
        logger.info(f"Filtered to {args.num_stocks_to_return} stocks")
    else:
        rows = list(resp.rows)

    if len(rows) == 0:
        raise Exception("Could not get ratings for any stocks")

    gbi_id_to_stock = {stock.gbi_id: stock for stock in stock_ids}
    score_dict = {}
    for row in rows:
        score_dict[gbi_id_to_stock[row.gbi_id]] = get_score_from_rating(row.rating_and_delta)

    await tool_log(log=f"Writing reasoning for {len(score_dict)} stocks", context=context)

    return await add_scores_and_rationales_to_stocks(score_dict, args.buy, context)
