import copy
import json
from typing import Any, Dict, List, Optional

from pydantic import field_validator

from agent_service.external.discover_svc_client import (
    get_news_sentiment_score,
    get_recommendation_score,
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
from agent_service.tools.news import (
    GetNewsDevelopmentsAboutCompaniesInput,
    get_all_news_developments_about_companies,
)
from agent_service.tools.other_text import (
    GetAllTextDataForStocksInput,
    get_all_text_data_for_stocks,
)
from agent_service.tools.stocks import GetStockUniverseInput, get_stock_universe
from agent_service.tools.tool_log import tool_log
from agent_service.types import PlanRunContext
from agent_service.utils.async_utils import gather_with_concurrency, identity
from agent_service.utils.date_utils import (
    convert_horizon_to_date,
    convert_horizon_to_days,
)
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

    input_days = convert_horizon_to_days(input_horizon)
    supported_horizon_to_days = {
        horizon: convert_horizon_to_days(horizon) for horizon in supported_horizons
    }

    min_pair = min(supported_horizon_to_days.items(), key=lambda x: abs(x[1] - input_days))
    return min_pair[0]


async def add_scores_and_rationales_to_stocks(
    ranked_stocks: List[StockID],
    score_dict: Dict[StockID, Score],
    is_buy: Optional[bool],
    context: PlanRunContext,
    news_horizon: Optional[str] = None,
) -> List[StockID]:

    if news_horizon:
        horizon_date = convert_horizon_to_date(news_horizon)
        texts: List[StockText] = await get_all_news_developments_about_companies(  # type: ignore
            GetNewsDevelopmentsAboutCompaniesInput(
                stock_ids=ranked_stocks, start_date=horizon_date
            ),
            context,
        )

    else:
        texts: List[StockText] = await get_all_text_data_for_stocks(  # type: ignore
            GetAllTextDataForStocksInput(stock_ids=ranked_stocks), context
        )
    aligned_text_groups = StockAlignedTextGroups.from_stocks_and_text(ranked_stocks, texts)
    str_lookup: Dict[StockID, str] = await Text.get_all_strs(  # type: ignore
        aligned_text_groups.val, include_header=True, text_group_numbering=True
    )

    ranked_stocks = [
        stock for stock in ranked_stocks if stock in str_lookup
    ]  # filter out those with no data

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
    for stock in ranked_stocks:
        text_str = str_lookup[stock]
        text_str = tokenizer.chop_input_to_allowed_length(text_str, used)
        if text_str == "":  # no text, skip
            tasks.append(identity(""))
        if is_buy is None:
            score = score_dict[stock]
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

    for result, stock in zip(results, ranked_stocks):
        text_group = aligned_text_groups.val[stock]
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
                    score=score_dict[stock],
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
    news_horizon: str = "1M"  # 1W, 1M, 3M
    news_only: bool = False
    num_stocks_to_return: Optional[int] = None
    star_rating_threshold: Optional[float] = None

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

    @field_validator("news_horizon", mode="before")
    @classmethod
    def validate_news_horizon(cls, value: Any) -> Any:
        if not isinstance(value, str):
            raise ValueError("news horizon must be a string")
        return map_input_to_closest_horizon(value.upper(), supported_horizons=["1W", "1M", "3M"])


@tool(
    description=(
        "This function provides stock recommendations and/or text justifying those recommendations"
        "There are two major modes, controlled by the filter boolean. If filter is on (True), the function"
        "will filter the provided stock list (or the S&P 500, if no stock list is provided) to a list of "
        "recommended buys (if buy = True) or recommended sells/shorts (if buy = False), the rankings of the "
        "outputted stocks for this is based on a machine learning algorithm. "
        "Note that the buy variable must NEVER, EVER be None if filter is True. "
        "Each stock selected with this function will include reasoning about why the stock is a buy or sell."
        "Either num_stock_to_return or star_rating_threshold can be used to define how the filter works. If "
        "The client indicates some number of recommendations they want, then num_stocks_to_return should be set "
        "to that number (or some reasonable interpretation, if vague, like `a few` can be mapped to 3)"
        "An example of a request that would use this function with filter=True and num_stocks_to_return is set is: "
        "`Give me 10 stocks you like/don't like`. "
        "If the client just asks directly for some recommendations, you should set num_stocks_to_return to 5"
        "If the client instead indicates they just want all positive or negative recommendations within "
        "some set of stocks, the star_rating_threshold should be set to 2.5, the middle of a 5 point star "
        "rating scale. "
        "If the client indicates a particular number of stars (or a score), then use that as the "
        "threshold, otherwise, if it is more vague, an appropriate corresponding rating should be selected. "
        "If the client indicates they want stocks above the star rating, use buy=True, else buy=False "
        "For example, if a client says they want to see only `deeply troubled` stocks, then filter should be True, "
        "buy must be set to False, and a star_rating_threshold choice of 1.0 is appropriate. "
        "Always express the star_rating_threshold as a float. "
        "Generally you will only use either start_rating_threshold or num_stocks_to_return. If both are set "
        "both kinds of filters will be applied."
        "When filter is false, this function returns the same set of stockIDs as its input, but "
        "a recommendation rationale will be included. If buy = True, a rationale for buying each stock will "
        "be added, if buy = False, a rationale for selling/shorting the stock will be added. If buy == None, "
        "then the rationale will be based on justifying the score from a machine learning algorithm. "
        "So, if a client says to `give me a reason to buy/sell NVDA`, you would use filter = False, and "
        "buy = true/false, respectively "
        "but if the client asks `Should I buy NVDA?` filter is still False, but buy should be None, and the "
        "rationale will reflect a post hoc rationalization of the machine-provided rating for the stock. "
        "Investment horizon and delta horizon are used for in the ML algorithm to decide how far into "
        "The future (investment horizon) or into the past (delta) to consider, you should increase them from the "
        "defaults only when the client expresses some specific interest in a longer term view. "
        "By default, the ML algorithm uses a mixture of quantitative information and news sentiment. "
        "You should use also use this function (and not filter_stocks_by_profile!) with the news_only flag "
        "when the client wants to filter stocks based on news semtiment, ."
        "When you get such a request, e.g. `filter to stocks with only positive "
        "news sentiment`, the optional news_only should be set to True, and only news information will be used in "
        "rating and corresponding rationale. "
        "This function looks up the news for the relevant stocks internally, you do not need to run "
        "the get_all_news_developments_about_companies function before this one!"
        "In news only mode, the news_horizon specifically controls how far back the algorithm looks for news "
        "when doing the sentiment calculation and the rationale."
        "If no stock ID's are provided (which must only happen when filter=True!), the S&P 500 stocks are used"
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
        if args.filter and args.num_stocks_to_return:
            if len(args.stock_ids) < args.num_stocks_to_return:
                raise ValueError(
                    "The number of stocks to return is greater than the number of stocks provided."
                )
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

    if not args.filter or args.news_only:  # don't want ism filter or news only requests
        settings_blob["ism_settings"]["match_labels"].extend(["Weak Match", "Poor Match"])

    if args.buy is None or args.buy:
        settings_blob["rating_settings"]["boundary"]["lb"] = 0.0001
        settings_blob["news_settings"]["sentiment_boundaries"] = [{"lb": -0.9999, "ub": 1}]
    else:
        settings_blob["rating_settings"]["boundary"]["ub"] = 4.9999
        settings_blob["news_settings"]["sentiment_boundaries"] = [{"lb": -1, "ub": 0.9999}]

    await tool_log(log=f"Getting stock ratings for {len(stock_ids)} stocks", context=context)

    resp = await get_temporary_discover_block_data(
        context.user_id, settings_blob, args.horizon, args.delta_horizon
    )

    rows = list(resp.rows)

    logger.info(f"Got scores for {len(rows)} stocks")

    if len(rows) == 0:
        raise Exception("Could not get ratings for any of the stocks provided")

    gbi_id_to_stock = {stock.gbi_id: stock for stock in stock_ids}
    score_dict = {}
    for row in rows:
        if args.news_only:
            score = get_news_sentiment_score(row)
        else:
            score = get_recommendation_score(row)
        score_dict[gbi_id_to_stock[row.gbi_id]] = score

    ranked_stocks = sorted(
        score_dict, key=lambda x: score_dict[x], reverse=args.buy or args.buy is None
    )

    if args.filter:
        if (
            args.num_stocks_to_return is None and args.star_rating_threshold is None
        ):  # if GPT is stupid
            args.star_rating_threshold = 2.5

        if args.star_rating_threshold:
            threshold = args.star_rating_threshold / 5
            if args.buy:
                ranked_stocks = [
                    stock for stock in ranked_stocks if score_dict[stock].val > threshold
                ]
            else:
                ranked_stocks = [
                    stock for stock in ranked_stocks if score_dict[stock].val < threshold
                ]

        if args.num_stocks_to_return:
            ranked_stocks = ranked_stocks[: args.num_stocks_to_return]

        logger.info(f"Filtered to {len(ranked_stocks)} stocks")

    if not args.news_only:
        news_horizon = None
    else:
        news_horizon = args.news_horizon

    await tool_log(log="Writing reasoning", context=context)

    final_stocks = await add_scores_and_rationales_to_stocks(
        ranked_stocks, score_dict, args.buy, context, news_horizon=news_horizon
    )

    await tool_log(log=f"Finished recommendation for {len(ranked_stocks)} stocks", context=context)

    return final_stocks
