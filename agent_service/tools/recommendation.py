import copy
import datetime
import inspect
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from pa_portfolio_service_proto_v1.investment_policy_match_pb2 import (
    StockInvestmentPolicySummary,
)
from pydantic import field_validator

from agent_service.external.discover_svc_client import (
    get_news_sentiment_score,
    get_rating_score,
    get_temporary_discover_block_data,
)
from agent_service.external.investment_policy_svc import (
    get_all_stock_investment_policies,
)
from agent_service.GPT.constants import (
    DEFAULT_CHEAP_MODEL,
    FILTER_CONCURRENCY,
    GPT4_O,
    NO_PROMPT,
    SONNET,
)
from agent_service.GPT.requests import GPT
from agent_service.GPT.tokens import GPTTokenizer
from agent_service.io_type_utils import HistoryEntry, Score, dump_io_type, load_io_type
from agent_service.io_types.dates import DateRange
from agent_service.io_types.stock import StockID
from agent_service.io_types.stock_aligned_text import StockAlignedTextGroups
from agent_service.io_types.text import StockText, Text
from agent_service.planner.errors import EmptyInputError, EmptyOutputError
from agent_service.tool import (
    TOOL_DEBUG_INFO,
    ToolArgs,
    ToolCategory,
    default_tool_registry,
    tool,
)
from agent_service.tools.LLM_analysis.prompts import CITATION_PROMPT, CITATION_REMINDER
from agent_service.tools.LLM_analysis.utils import extract_citations_from_gpt_output
from agent_service.tools.news import (
    GetNewsDevelopmentsAboutCompaniesInput,
    get_all_news_developments_about_companies,
)
from agent_service.tools.other_text import (
    GetAllTextDataForStocksInput,
    get_default_text_data_for_stocks,
)
from agent_service.tools.stocks import GetStockUniverseInput, get_stock_universe
from agent_service.tools.tool_log import tool_log
from agent_service.types import PlanRunContext
from agent_service.utils.async_utils import gather_with_concurrency, identity
from agent_service.utils.date_utils import (
    convert_horizon_to_date,
    convert_horizon_to_days,
    get_now_utc,
)
from agent_service.utils.gpt_logging import GptJobIdType, GptJobType, create_gpt_context
from agent_service.utils.pagerduty import pager_wrapper
from agent_service.utils.prefect import get_prefect_logger
from agent_service.utils.prompt_utils import Prompt
from agent_service.utils.tool_diff import get_prev_run_info


@dataclass
class RecommendationScores:
    news_score: Optional[Score] = None
    rating_score: Optional[Score] = None
    # maybe more later

    def get_overall(self) -> Score:
        return Score.average([score for score in [self.news_score, self.rating_score] if score])

    def add_score_history(self, stock: StockID) -> StockID:
        if self.news_score:
            stock = stock.inject_history_entry(
                HistoryEntry(title="News Rating", score=self.news_score)
            )
        if self.rating_score:
            stock = stock.inject_history_entry(
                HistoryEntry(title="Quant Rating", score=self.rating_score)
            )
        return stock


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
    "rating_settings": {
        "boundary": {"lb": None, "ub": None},
        "weight": 1.0,
        "filter_by_ratings": True,
    },
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
compelling argument.
"""
RECOMMENDATION_SYS_PROMPT_STR += CITATION_PROMPT

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
"""
RECOMMENDATION_MAIN_PROMPT_STR += CITATION_REMINDER
RECOMMENDATION_MAIN_PROMPT_STR += (
    "\n\nNow provide your reasoning for the investment decision in a short paragraph:\n"
)

BUY_DIRECTION = (
    "In particular, you must write an argument that focuses on evidence for buying the stock."
)
SELL_DIRECTION = "In particular, you must write an argument that focuses on evidence for selling/shorting the stock."
SCORE_DIRECTION = (
    "In particular, you should write an argument which is compatible with a score of {score} on"
    " a 0 to 1 scale. If the score is above 0.6 you should focus on evidence for investing in the"
    " stock, and if it is below 0.4 you should focus on evidence that would disuade investors. You "
    " must never directly mention this score in your output!"
)


RECOMMENDATION_MAIN_PROMPT = Prompt(RECOMMENDATION_MAIN_PROMPT_STR, "RECOMMENDATION_MAIN_PROMPT")
RECOMMENDATION_SYS_PROMPT = Prompt(RECOMMENDATION_SYS_PROMPT_STR, "RECOMMENDATION_SYS_PROMPT")

POLICY_MAIN_PROMPT_STR = (
    "You are an investment advisor trying to match a client's mention of an investment style/policy "
    "to a pre-existing list. You will be given the name mentioned by the client, and the list of styles, simply "
    "output the element of the list that best matches the mention, or return `None` if there "
    "is nothing in the list that has any relation to the match. Here is the client mention: {policy_ref}. "
    "And here is the list:\n{list_of_policies}\nNow output the best match:\n"
)

POLICY_MAIN_PROMPT = Prompt(POLICY_MAIN_PROMPT_STR, "POLICY_MAIN_PROMPT")


REC_ADD_DIFF_MAIN_PROMPT = Prompt(
    name="REC_ADD_DIFF_MAIN_PROMPT",
    template="""
You are a financial analyst that carries out periodic analysis of stocks and provide lists of stocks
to your client. Your current goal is to explain why you've added a particular stock to a list of
buy or sell recommendation.
You will be provided with a company name, a kind of recommendation (buy or sell), current scores about
the company, and older scores about the company from you previous analysis. There are two scores in each set,
one is news score which is a reflection of recent news sentiment about the company, and the other is a quant
score which uses a quantitive machine learning model with features corresponding to typical quantiative analyst
metrics to predict future stock performance.
Both scores have a range of 0-5, 5 indicating a strong positive sentiment and 0 indicating a strong
negative sentiment, the two are weighted equally in the final score which determines the ranking of stocks
for recommendation. Generally we would expect higher scores for buy recommendations, and lower scores for
sell recommendations.
In a single sentence, briely explain why we have added the stock in this pass after
we excluded it in the previous one. Usually this can be explained directly by simply stating any major
changes in the scores, make sure you mention both the new and old value for the releven score. For example,
`Nvida was added in our buys due to a major increase in its news sentiment score, from 2.5 to 4.5,
since the last analysis.`
If only one score changed significantly, focus only on that score.
If neither score changed hardly at all (less than 0.1), or if the result is non-intuitive (a stock decreases
in scores and yet gets added to a buy recommendation list, you must NOT state score change as the reason this
occurred, instead you should explain this by vague reference to even more extreme changes from other stocks
(although Nvida went up, other stocks went up even more.) Keep it brief and professional.
Here is the company name:  {company_name}.
Here is the current scores for the company: {curr_scores}.
Here are the stats from your previous analysis: {prev_scores}.
Here is the kind of stock recommendation you are explaining: {buy_or_sell}
{news}
Now write your explanation of the change:""",
)

REC_REMOVE_DIFF_MAIN_PROMPT = Prompt(
    name="REC_REMOVE_DIFF_MAIN_PROMPT",
    template="""
You are a financial analyst that carries out periodic analysis of stocks and provide lists of stocks
to your client. Your current goal is to explain why you've removed a particular stock from a list of
buys or sells.
You will be provided with a company name, a class of recommendation you are making (buy or sell), current scores about
the company, and older scores about the company from you previous analysis. There are two scores in each set,
one is news score which is a reflection of recent news sentiment about the company, and the other is a quant
score which uses a quantitive machine learning model with features corresponding to typical quantiative analyst
metrics to predict future stock performance.
Both scores have a range of 0-5, 5 indicating a strong positive sentiment and 0 indicating a strong
negative sentiment, the two are weighted equally in the final score which determines the ranking of stocks
for recommendation. Generally we would expect higher scores for buy recommendations, and lower scores for
sell recommendations.
In a single sentence, briely explain why we have removed the stock in this pass after
we included it in the previous one. Usually this can be explained directly by simply stating any major
changes in the scores, make sure you mention both the new and old value for the releven score. For example,
`Nvida was removed from our buys due to a major drop in its news sentiment score, from 4.5 to 2.5,
since the last analysis.`
If only one score changed significantly, focus only on that score.
If neither score changed hardly at all (less than 0.1), or the result is non-intuitive (a stock increases in
scores and yet fell out of a buy recommendation list), you must NOT state the score change as the reason this
occurred, but instead explain the change by vague reference to even more extreme changes from other stocks
(although Nvida went down, other stocks went down even more.) Keep it brief and professional.
Here is the company name: {company_name}.
Here is the current scores for the company: {curr_scores}.
Here are the stats from your previous analysis: {prev_scores}.
Here is the kind of stock recommendation you are explaining: {buy_or_sell}
{news}
Now write your explanation of the change:""",
)


NEWS_TEMPLATE = (
    "Finally, here is some news that this company that has occurred since the last analysis. If you are focusing "
    "on the news score as a cause of the change in recommendation and any of this news can be used to sensibly "
    "explain the change in the news score, please add a sentence which speculates on the cause, however you "
    "should avoid any firm conclusions. Make sure you only highlight news that that is very likely to cause a major "
    "change in the recommendations for a stock and is compatible with the change in news score; if it is not possible "
    "to make reasonable case with this news, it is much better to not mention the news at all. "
    "Refer to the news as `the latest news`. Here is the news:\n---\n{news}\n---\n"
)


async def pick_investment_style(
    policy_ref: str, isms: List[StockInvestmentPolicySummary], context: PlanRunContext
) -> Optional[StockInvestmentPolicySummary]:
    gpt_context = create_gpt_context(
        GptJobType.AGENT_PLANNER, context.agent_id, GptJobIdType.AGENT_ID
    )
    llm = GPT(context=gpt_context, model=DEFAULT_CHEAP_MODEL)
    list_of_policies = "\n".join([ism.name for ism in isms])
    main_prompt = POLICY_MAIN_PROMPT.format(
        policy_ref=policy_ref, list_of_policies=list_of_policies
    )
    result = (await llm.do_chat_w_sys_prompt(main_prompt, NO_PROMPT)).strip()
    for ism in isms:
        if ism.name == result:
            return ism
    return None


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
    score_dict: Dict[StockID, RecommendationScores],
    is_buy: Optional[bool],
    context: PlanRunContext,
    news_horizon: Optional[str] = None,
    task_id: Optional[str] = None,
    date: Optional[DateRange] = None,
) -> List[StockID]:
    logger = get_prefect_logger(__name__)

    if date is None:
        end_date = get_now_utc().date()
    else:
        end_date = date.end_date

    if news_horizon:
        start_date = convert_horizon_to_date(
            news_horizon if news_horizon else "1M", end_date=end_date
        )
        texts: List[StockText] = await get_all_news_developments_about_companies(  # type: ignore
            GetNewsDevelopmentsAboutCompaniesInput(
                stock_ids=ranked_stocks,
                date_range=(DateRange(start_date=start_date, end_date=end_date)),
            ),
            context,
        )

    else:
        start_date = convert_horizon_to_date("1Q", end_date=end_date)
        texts: List[StockText] = await get_default_text_data_for_stocks(  # type: ignore
            GetAllTextDataForStocksInput(
                stock_ids=ranked_stocks,
                date_range=DateRange(start_date=start_date, end_date=end_date),
            ),
            context,
        )
    aligned_text_groups = StockAlignedTextGroups.from_stocks_and_text(ranked_stocks, texts)
    str_lookup: Dict[StockID, str] = await Text.get_all_strs(  # type: ignore
        aligned_text_groups.val, include_header=True, text_group_numbering=True
    )

    ranked_stocks = [
        stock for stock in ranked_stocks if stock in str_lookup
    ]  # filter out those with no data

    gpt_context = create_gpt_context(
        GptJobType.AGENT_TOOLS, context.agent_id, GptJobIdType.AGENT_ID
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
            continue
        if is_buy is None:
            score = score_dict[stock].get_overall()
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
        if not result:
            stocks_with_rec.append(stock)
        text_group = aligned_text_groups.val[stock]
        scores = score_dict[stock]
        try:
            rationale, citations = await extract_citations_from_gpt_output(
                result, text_group, context
            )
        except Exception as e:
            logger.exception(f"Could not extract rationale with citations due to {e}")
            rationale = ""
            citations = []

        if citations is None:
            citations = []

        stock = stock.inject_history_entry(
            HistoryEntry(
                explanation=rationale,
                title="Recommendation Reasoning",
                citations=citations,  # type:ignore
                task_id=task_id,
            )
        )

        stock = scores.add_score_history(stock)

        stocks_with_rec.append(stock)

    return stocks_with_rec


async def recommendation_filter_added_diff_info(
    added_stocks: List[StockID],
    news_texts: List[StockText],
    curr_score_dict: Dict[StockID, RecommendationScores],
    prev_score_dict: Dict[StockID, RecommendationScores],
    buy_or_sell: str,
    agent_id: str,
) -> Dict[StockID, str]:
    gpt_context = create_gpt_context(GptJobType.AGENT_TOOLS, agent_id, GptJobIdType.AGENT_ID)
    llm = GPT(context=gpt_context, model=SONNET)

    aligned_text_groups = StockAlignedTextGroups.from_stocks_and_text(added_stocks, news_texts)
    str_lookup: Dict[StockID, str] = await Text.get_all_strs(  # type: ignore
        aligned_text_groups.val, include_header=False, text_group_numbering=False
    )

    tasks = []
    for stock in added_stocks:
        if stock in str_lookup and str_lookup[stock]:
            news_str = NEWS_TEMPLATE.format(news=str_lookup[stock])
        else:
            news_str = ""

        curr_news = curr_score_dict[stock].news_score
        curr_quant = curr_score_dict[stock].rating_score
        prev_news = prev_score_dict[stock].news_score
        prev_quant = prev_score_dict[stock].rating_score

        tasks.append(
            llm.do_chat_w_sys_prompt(
                REC_ADD_DIFF_MAIN_PROMPT.format(
                    company_name=stock.company_name,
                    news=news_str,
                    buy_or_sell=buy_or_sell,
                    curr_scores={
                        "news": (curr_news.val * 5 if curr_news is not None else None),
                        "quant": (curr_quant.val * 5 if curr_quant is not None else None),
                    },
                    prev_scores={
                        "news": (prev_news.val * 5 if prev_news is not None else None),
                        "quant": (prev_quant.val * 5 if prev_quant is not None else None),
                    },
                ),
                NO_PROMPT,
            )
        )

    results = await gather_with_concurrency(tasks)
    return {stock: explanation for stock, explanation in zip(added_stocks, results)}


async def recommendation_filter_removed_diff_info(
    removed_stocks: List[StockID],
    news_texts: List[StockText],
    curr_score_dict: Dict[StockID, RecommendationScores],
    prev_score_dict: Dict[StockID, RecommendationScores],
    buy_or_sell: str,
    agent_id: str,
) -> Dict[StockID, str]:
    gpt_context = create_gpt_context(GptJobType.AGENT_TOOLS, agent_id, GptJobIdType.AGENT_ID)
    llm = GPT(context=gpt_context, model=SONNET)

    aligned_text_groups = StockAlignedTextGroups.from_stocks_and_text(removed_stocks, news_texts)
    str_lookup: Dict[StockID, str] = await Text.get_all_strs(  # type: ignore
        aligned_text_groups.val, include_header=False, text_group_numbering=False
    )

    tasks = []
    for stock in removed_stocks:
        if stock in str_lookup and str_lookup[stock]:
            news_str = NEWS_TEMPLATE.format(news=str_lookup[stock])
        else:
            news_str = ""

        curr_news = curr_score_dict[stock].news_score
        curr_quant = curr_score_dict[stock].rating_score
        prev_news = prev_score_dict[stock].news_score
        prev_quant = prev_score_dict[stock].rating_score

        tasks.append(
            llm.do_chat_w_sys_prompt(
                REC_REMOVE_DIFF_MAIN_PROMPT.format(
                    company_name=stock.company_name,
                    news=news_str,
                    buy_or_sell=buy_or_sell,
                    curr_scores={
                        "news": (curr_news.val * 5 if curr_news is not None else None),
                        "quant": (curr_quant.val * 5 if curr_quant is not None else None),
                    },
                    prev_scores={
                        "news": (prev_news.val * 5 if prev_news is not None else None),
                        "quant": (prev_quant.val * 5 if prev_quant is not None else None),
                    },
                ),
                NO_PROMPT,
            )
        )

    results = await gather_with_concurrency(tasks)
    return {stock: explanation for stock, explanation in zip(removed_stocks, results)}


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
    investment_style: Optional[str] = None
    date: Optional[DateRange] = None
    # date is currently only partially functional, the scores are from today since disco blocks
    # does not have PIT scores, however at least the text explanation is from the correct period
    # TODO: add PIT scores

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
        "This function provides stock recommendations, including ratings/scores for each stock outputted, "
        "and text justifying those recommendations. The scores and the justification are displayed "
        "in a table if the stock list is shown to the client. When a user asks for just a 'score' or a 'rating' "
        "for stocks, you will use this tool."
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
        "Look carefully for some mention of best N or top N in the client input, and if it exists, don't forget to "
        "assign num_stocks_to_return to N, otherwise you will return all positive/negative stocks, and there "
        "is no other way for the client to further filter these stocks!"
        "When filter is false, this function returns the same set of stockIDs as its input, but they will be "
        "sorted by their score and a recommendation rationale will be included."
        "You should use filter = False if the client is asking for just a ranking and/or recommendation for "
        "a specific list of stocks, without filtering. This includes asking for recommendations for all "
        "stock in a users portfolio for another stock universe. Do not filter if the client clearly wants "
        "output for all stocks, not just subset; if they mention `all` stocks or `each` stock, then "
        "you probably want filter=False"
        "If buy = True, a rationale for buying each stock will "
        "be added, if buy = False, a rationale for selling/shorting the stock will be added. If buy = None, "
        "then the rationale will be based on justifying the score from a machine learning algorithm. "
        "So, if a client says to `give me a reason to buy/sell NVDA`, you would use filter = False, and "
        "buy = true/false, respectively "
        "but if the client asks `Should I buy NVDA?` or `Tell me if I should short NVDA` then"
        "filter is still False, but buy should be None, not True or False, and the "
        "rationale will reflect a post hoc rationalization of the machine-provided rating for the stock. "
        "Again, make sure to use buy=None if the user wants you to choose whether or not to recommend "
        "a particular stock or stocks. If the user is asking a direct question about whether to buy or "
        "sell a stock (or both!), then they are asking for ML recommendations, and buy should be None. "
        "If the user asks for a news sentiment of a stock or stocks, use this tool with the filter off"
        "and buy= None. If they ask for a specific date, pass a DateRange to the date."
        "Investment horizon and delta horizon are used for in the ML algorithm to decide how far into "
        "The future (investment horizon) or into the past (delta) to consider, you should increase them from the "
        "defaults only when the client expresses some specific interest in a longer term view. "
        "By default, the ML algorithm uses a mixture of quantitative information and news sentiment. "
        "If the client asks for a quant rating or quant score for a set of stocks, you should use this function"
        "with filter=False and news_only=False"
        "If the client asks for news sentiment scores in a table, use this tool with news_only=True. Do not use the "
        "get_news_sentiment_time_series tool unless the client asks specifically for sentiment graph."
        "You should use also use this function (and not filter_stocks_by_profile!) with the news_only flag "
        "when the client wants to filter stocks based on news sentiment."
        "When you get such a request, e.g. `filter to stocks with only positive "
        "news sentiment`, the optional news_only should be set to True, and only news information will be used in "
        "rating and corresponding rationale. "
        "If the client asks for recommendations AND includes a requirement that news be positive, it is "
        "perfectly reasonable to run this tool twice, first with news_only=True, and the second time "
        "with news_only=False. You should never use the filter by profile function for filtering by simple "
        "news sentiment!!!!"
        "This function looks up the news for the relevant stocks internally, you do not need to run "
        "the get_all_news_developments_about_companies function before this one!"
        "In news only mode, the news_horizon specifically controls how far back the algorithm looks for news "
        "when doing the sentiment calculation and the rationale."
        "Valid horizons for the horizon arguments are 1W, 1M, 3M, and 1Y, do not pass anything else!"
        "If the client mentions an investment policy/style, it should be passed in as investment style. "
        "If no stock ID's are provided (which must only happen when filter=True!), the S&P 500 stocks are used"
        "You must NOT use this tool when the client asks directly for `analyst expectations`, "
        "analyst expectations is a statistic that accessible through the get_statistic_data tool! "
        "However, analyst expectations should only be retrieved when someone uses exactly that wording. "
        "If the client asks for a ranking of stocks or scores for stocks without mentioning any other specific "
        "statistic to rank on, e.g. `give me a table with stocks ranked by score` use this tool and "
        "NOT transform table and/or the statistic tool."
        "If the client mentions Boosted rec, boosted recommendation,"
        "boosted.Ai rec or any other variation they mean this tool"
        "You must NEVER pass the output of this tool to the transform_table function. If the user wants to filter "
        "On the top/bottom n stocks, you must use the num_stocks_to_return argument. When the list of stocks "
        "outputed by this tool are displayed it looks like a table to the client, so you do not need to take any "
        "other steps to convert the output to a table!!!"
    ),
    category=ToolCategory.STOCK_SENTIMENT,
    tool_registry=default_tool_registry(),
    reads_chat=True,
    update_instructions=(
        "Since the most of the important functionalities of this function are controlled by arguments "
        "to the function, nearly any change that involves the operation of this function requires a "
        "full replan. This includes anything that directly or indirectly affects the selection of the "
        "of recommended stocks. The only circumstance where a rerun is appropriate is if the user has "
        "asked only for a change in the style or focus of the recommendation texts."
    ),
)
async def get_stock_recommendations(
    args: GetStockRecommendationsInput, context: PlanRunContext
) -> List[StockID]:
    # NOTE: You can't use `get_dummy()` to create a dummy context for this tool because it requires
    # an actual user ID to make the gRPC call (specifically for PA SVC)

    logger = get_prefect_logger(__name__)

    debug_info: Dict[str, Any] = {}
    TOOL_DEBUG_INFO.set(debug_info)

    prev_args = None
    prev_output = None
    prev_scores = None
    prev_time = None

    try:  # since everything here is optional, put in try/except
        prev_run_info = await get_prev_run_info(context, "get_stock_recommendations")
        if prev_run_info is not None:
            prev_args = GetStockRecommendationsInput.model_validate_json(prev_run_info.inputs_str)
            prev_output: List[StockID] = prev_run_info.output  # type:ignore
            prev_other: Dict[str, str] = prev_run_info.debug  # type:ignore
            prev_time: datetime = prev_run_info.timestamp  # type:ignore
            if prev_other:
                prev_score_dict = (
                    load_io_type(prev_other["score_dict"]) if prev_other.get("score_dict") else {}
                )
                prev_scores = {  # type: ignore
                    stock: RecommendationScores(news_score=news_score, rating_score=rating_score)
                    for stock, news_score, rating_score in prev_score_dict  # type: ignore
                }
    except Exception as e:
        logger.exception(f"Error getting info from previous run: {e}")
        pager_wrapper(
            current_frame=inspect.currentframe(),
            module_name=__name__,
            context=context,
            e=e,
            classt="AgentUpdateError",
            summary="Failed to get previous run info",
        )

    if args.stock_ids:
        if args.filter and args.num_stocks_to_return:
            if len(args.stock_ids) < args.num_stocks_to_return:
                await tool_log(
                    (
                        f"Tried to find {args.num_stocks_to_return} stocks, "
                        f"but only {len(args.stock_ids)} stocks passed in."
                    ),
                    context=context,
                )
        stock_ids = args.stock_ids
    else:
        if not args.filter:
            raise EmptyInputError(
                "Get recommended stocks called in non-filter mode but no stocks passed"
            )
        # we perhaps can store the SP500 stocks as a log output but not for now as they are GBI IDs
        await tool_log(log="No stock IDs provided. Using S&P 500 stocks.", context=context)
        stock_ids: List[StockID] = await get_stock_universe(  # type: ignore
            args=GetStockUniverseInput(universe_name="SPDR S&P 500 ETF Trust"), context=context
        )

    ism_resp = await get_all_stock_investment_policies(context.user_id)
    ism_id = None
    if ism_resp.investment_policies:
        ism = None
        if args.investment_style:
            ism = await pick_investment_style(
                args.investment_style, list(ism_resp.investment_policies), context
            )
            if ism:
                ism_id = ism.investment_policy_id.id

        # Removing this for now, maybe we want some way to do something like it later though
        # if ism_id is None:
        #     ism = max(ism_resp.investment_policies, key=lambda x: x.last_updated.ToDatetime())
        #     ism_id = ism.investment_policy_id.id
        if ism:
            await tool_log(
                log=f'Using Investment Style "{ism.name}" to search stocks', context=context
            )

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
        settings_blob["rating_settings"]["filter_by_ratings"] = False

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
        raise EmptyOutputError("Could not get ratings for any of the stocks provided")

    gbi_id_to_stock = {stock.gbi_id: stock for stock in stock_ids}
    score_dict: Dict[StockID, RecommendationScores] = {}
    for row in rows:
        stock_id = gbi_id_to_stock[row.gbi_id]
        rec_scores = RecommendationScores()
        rec_scores.news_score = get_news_sentiment_score(row)
        if not args.news_only:
            rating_score = get_rating_score(row)
            rec_scores.rating_score = rating_score if rating_score.val > 0.0 else None
        score_dict[stock_id] = rec_scores

    debug_info["score_dict"] = dump_io_type(
        [
            [stock_id, recommendation.news_score, recommendation.rating_score]
            for stock_id, recommendation in score_dict.items()
        ]
    )

    ranked_stocks = sorted(
        score_dict,
        key=lambda x: score_dict[x].get_overall().val,
        reverse=args.buy or args.buy is None,
    )

    if args.filter:
        if (
            args.num_stocks_to_return is None and args.star_rating_threshold is None
        ):  # if GPT is stupid
            args.star_rating_threshold = 2.5

        if args.star_rating_threshold:
            threshold = args.star_rating_threshold / 5
            num_before = len(ranked_stocks)
            if args.buy:
                ranked_stocks = [
                    stock
                    for stock in ranked_stocks
                    if score_dict[stock].get_overall().val > threshold
                ]
                await tool_log(
                    f"Filtered out {num_before - len(ranked_stocks)} stocks below a {args.star_rating_threshold} star rating",  # noqa
                    context=context,
                )
            else:
                ranked_stocks = [
                    stock
                    for stock in ranked_stocks
                    if score_dict[stock].get_overall().val < threshold
                ]
                await tool_log(
                    f"Filtered out {num_before - len(ranked_stocks)} stocks above a {args.star_rating_threshold} star rating",  # noqa
                    context=context,
                )

        if args.num_stocks_to_return:
            if args.buy:
                await tool_log(
                    f"Keeping only the {args.num_stocks_to_return} highest-rated stocks",
                    context=context,
                )
            else:
                await tool_log(
                    f"Keeping only the {args.num_stocks_to_return} lowest-rated stocks",
                    context=context,
                )
            ranked_stocks = ranked_stocks[: args.num_stocks_to_return]

        logger.info(f"Filtered to {len(ranked_stocks)} stocks")

    if not args.news_only:
        news_horizon = None
    else:
        news_horizon = args.news_horizon

    await tool_log(log="Writing reasoning", context=context)

    final_stocks = await add_scores_and_rationales_to_stocks(
        ranked_stocks,
        score_dict,
        args.buy,
        context,
        news_horizon=news_horizon,
        task_id=context.task_id,
        date=args.date,
    )

    try:  # since everything here is optional, put in try/except
        if (
            context.diff_info is not None
            and prev_output is not None
            and prev_args
            and prev_scores
            and args.filter
        ):
            shared_stocks = set(score_dict) & set(prev_scores)  # use score dict stocks
            curr_stocks = set(final_stocks)
            prev_stocks = set(prev_output)
            added_stocks = (curr_stocks - prev_stocks) & shared_stocks
            removed_stocks = (prev_stocks - curr_stocks) & shared_stocks

            changed_stocks = list(added_stocks | removed_stocks)
            try:
                news_texts: StockText = await get_all_news_developments_about_companies(  # type: ignore
                    GetNewsDevelopmentsAboutCompaniesInput(
                        stock_ids=changed_stocks,
                        date_range=DateRange(
                            start_date=prev_time.date(), end_date=datetime.date.today()
                        ),
                    ),
                    context,
                )
            except Exception as e:
                logger.warning(f"No news data available for changed stocks: {e}")
                news_texts = []

            buy_or_sell = "buy" if args.buy else "sell"

            added_diff_info = await recommendation_filter_added_diff_info(
                added_stocks, news_texts, score_dict, prev_scores, buy_or_sell, context.agent_id
            )

            removed_diff_info = await recommendation_filter_removed_diff_info(
                removed_stocks, news_texts, score_dict, prev_scores, buy_or_sell, context.agent_id
            )

            context.diff_info[context.task_id] = {
                "added": added_diff_info,
                "removed": removed_diff_info,
            }

    except Exception as e:
        logger.warning(f"Error doing diff from previous run: {e}")

    await tool_log(log=f"Finished recommendation for {len(ranked_stocks)} stocks", context=context)

    return final_stocks
