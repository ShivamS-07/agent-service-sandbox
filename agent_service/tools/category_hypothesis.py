import json
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

from gpt_service_proto_v1.service_grpc import GPTServiceStub

from agent_service.GPT.constants import GPT4_O, NO_PROMPT, SONNET
from agent_service.GPT.requests import GPT, _get_gpt_service_stub
from agent_service.io_type_utils import (
    ComplexIOBase,
    HistoryEntry,
    IOType,
    Score,
    io_type,
)
from agent_service.io_types.stock import StockID
from agent_service.io_types.text import (
    StockDescriptionText,
    StockEarningsSummaryPointText,
    StockEarningsSummaryText,
    StockNewsDevelopmentText,
    StockText,
    Text,
    TextCitation,
)
from agent_service.tool import ToolArgs, ToolCategory, ToolRegistry, tool
from agent_service.tools.category import Categories, Category
from agent_service.tools.tool_log import tool_log
from agent_service.types import PlanRunContext
from agent_service.utils.async_utils import gather_with_concurrency
from agent_service.utils.gpt_logging import GptJobIdType, GptJobType, create_gpt_context
from agent_service.utils.hypothesis.hypothesis_pipeline import HypothesisPipeline
from agent_service.utils.hypothesis.types import (
    CompanyEarningsTopicInfo,
    CompanyNewsTopicInfo,
)
from agent_service.utils.hypothesis.utils import get_earnings_topics, get_news_topics
from agent_service.utils.postgres import get_psql
from agent_service.utils.prefect import get_prefect_logger
from agent_service.utils.prompt_utils import Prompt
from agent_service.utils.string_utils import clean_to_json_if_needed


@io_type
class CategoricalHypothesisTexts(ComplexIOBase):
    val: List[Text]
    categories: Categories

    async def split_into_components(self) -> List[IOType]:
        return [self.val[0], self.categories] + self.val[1:]  # type: ignore


class AnalyzeHypothesisWithCategoriesInput(ToolArgs):
    hypothesis: str
    categories: List[Category]
    stocks: List[StockID]
    all_text_data: List[StockText]
    # if only 1 stock mentioned in `hypothesis`, it will be assigned. If no stock or more than 1
    # stocks are mentioned, it will be None.
    target_stock: Optional[StockID] = None


@tool(
    description=(
        "Given a list of relevant text data and a list of categories used to break hypothesis down, "
        "this function generates a score indicating the extent to which the provided hypothesis is "
        "supported, and a short summary which explains the score with reference to "
        "the information in the provided text data."
        "This hypothesis must be focused on a specific stock or a small group of stocks, this function "
        "must NOT be used to filter stocks more generally! (i.e. Do not use it for "
        " `Give/find me stocks...` type queries, use the filter by profile tool). "
        "If the hypothesis specifies a company, then it should be the target stock. Otherwise, leave it to None."
    ),
    category=ToolCategory.HYPOTHESIS,
    tool_registry=ToolRegistry,
)
async def analyze_hypothesis_with_categories(
    args: AnalyzeHypothesisWithCategoriesInput, context: PlanRunContext
) -> CategoricalHypothesisTexts:
    logger = get_prefect_logger(__name__)

    # Step: Group texts by sources
    logger.info("Grouping texts by sources")
    company_descriptions = [t for t in args.all_text_data if isinstance(t, StockDescriptionText)]
    news_developments = [t for t in args.all_text_data if isinstance(t, StockNewsDevelopmentText)]
    earnings_summaries = [t for t in args.all_text_data if isinstance(t, StockEarningsSummaryText)]

    earnings_summary_points = await StockEarningsSummaryPointText.init_from_earnings_texts(
        earnings_summaries
    )

    stocks = list(
        {t.stock_id for t in company_descriptions if t.stock_id}
        | {t.stock_id for t in news_developments if t.stock_id}
        | {t.stock_id for t in earnings_summaries if t.stock_id}
    )

    gpt_service_stub = _get_gpt_service_stub()[0]

    # Step: Revise hypothesis to remove company specific information
    logger.info("Revising hypothesis to remove any company specific information")
    revised_hypothesis = await revise_hypothesis(args.hypothesis, context, gpt_service_stub)
    logger.info(f"Revised hypothesis: {revised_hypothesis}")  # Not visible to users

    # Step: Filter texts by relevant topics
    logger.info("Filtering relevant topics by revised hypothesis")
    news_topics, earnings_topics = await find_relevant_topics_by_hypothesis(
        revised_hypothesis, news_developments, earnings_summary_points, gpt_service_stub
    )
    await tool_log(
        log=(
            f"Found {len(news_topics)} relevant news topics and {len(earnings_topics)} "
            f"relevant earnings topics"
        ),
        context=context,
    )

    # Step: Classify news/earnings topics into categories
    logger.info("Classifying news and earnings topics into categories")

    gbi_ids = [t.stock_id.gbi_id for t in company_descriptions if t.stock_id]
    descriptions = get_psql().get_short_company_descriptions_for_gbi_ids(gbi_ids)
    gbi_id_to_short_description = {
        gbi_id: desc for gbi_id, (desc, _) in descriptions.items() if desc
    }

    categories = sorted(args.categories, key=lambda x: x.weight, reverse=True)
    category_to_topics = await classify_news_and_earnings_topics_into_category(
        revised_hypothesis,
        categories,
        news_topics,
        earnings_topics,
        gbi_id_to_short_description,
        context,
        gpt_service_stub,
    )

    category_topic_log = ""
    for category_idx, topics in category_to_topics.items():
        category = args.categories[category_idx]
        category_topic_log += f"Category {category.name}: {len(topics)} topics; "
    await tool_log(log=category_topic_log, context=context)

    # Step: Rank and summarize for each category
    logger.info("Ranking and summarizing for each category")
    candidate_target_stock = args.target_stock
    category_to_result = await rank_and_summarize_for_each_category(
        candidate_target_stock,
        stocks,
        revised_hypothesis,
        categories,
        category_to_topics,
        gbi_id_to_short_description,
        context,
        gpt_service_stub,
    )
    await tool_log(
        log="Ranked all the relevant stocks for each category and created summaries",
        context=context,
    )

    # Step: Calculate weighted average scores and determine the real target stock
    actual_target_stock, total_scores = calculate_weighted_average_scores(
        candidate_target_stock, stocks, categories, category_to_result
    )

    # Step: Overall summary
    # we need to use the original hypothesis here
    logger.info("Generating the final summary")
    final_summary = await overall_summary(
        actual_target_stock,
        args.hypothesis,
        categories,
        category_to_result,
        total_scores,
        context,
        gpt_service_stub,
    )
    await tool_log(log="Generated the final summary", context=context)

    # Step: Prepare outputs
    output = await prepare_categorical_hypothesis_outputs(
        actual_target_stock,
        final_summary,
        total_scores[actual_target_stock.symbol],  # type: ignore
        categories,
        category_to_topics,
        category_to_result,
        news_developments,
        earnings_summary_points,
    )

    return output


####################################################################################################
# Utils
####################################################################################################
async def revise_hypothesis(
    hypothesis: str, context: PlanRunContext, gpt_service_stub: Optional[GPTServiceStub] = None
) -> str:
    prompt_str = """
        Given the hypothesis, revise it to make it non company-specific. \
        For example, if the original hypothesis is 'Is Expedia the leader in the travel industry?', \
        the revised hypothesis should be 'Who is the leader in the travel industry?'. \
        If no stock/company is mentioned in the hypothesis, return the original hypothesis.
        Here is the original hypothesis: {hypothesis}
    """
    prompt = Prompt(template=prompt_str, name="REVISE_HYPOTHESIS_SYS_PROMPT")

    gpt_context = create_gpt_context(
        GptJobType.AGENT_TOOLS, context.agent_id, GptJobIdType.AGENT_ID
    )
    return await GPT(
        context=gpt_context, model=GPT4_O, gpt_service_stub=gpt_service_stub
    ).do_chat_w_sys_prompt(
        main_prompt=prompt.format(hypothesis=hypothesis),
        sys_prompt=NO_PROMPT,
    )


async def find_relevant_topics_by_hypothesis(
    hypothesis: str,
    news_developments: List[StockNewsDevelopmentText],
    earnings_summary_points: List[StockEarningsSummaryPointText],
    gpt_service_stub: Optional[GPTServiceStub] = None,
) -> Tuple[List[CompanyNewsTopicInfo], List[CompanyEarningsTopicInfo]]:
    logger = get_prefect_logger(__name__)

    # FIXME: Parallelize

    # segerate news/earnings topics by stocks
    stock_to_developments: Dict[StockID, List[StockNewsDevelopmentText]] = defaultdict(list)
    for development in news_developments:
        if development.stock_id is None:
            continue

        stock_to_developments[development.stock_id].append(development)

    stock_to_earnings_points: Dict[StockID, List[StockEarningsSummaryPointText]] = defaultdict(list)
    for point in earnings_summary_points:
        if point.stock_id is None:
            continue

        stock_to_earnings_points[point.stock_id].append(point)

    # create hypothesis pipeline objects
    stock_id_to_pipeline: Dict[StockID, HypothesisPipeline] = {}
    embedding: List[float] = None  # type: ignore
    hypothesis_breakdown: Dict[str, Any] = None  # type: ignore
    for stock_id in stock_to_developments.keys():
        pipeline = HypothesisPipeline(
            stock_id.gbi_id, hypothesis, gpt_service_stub=gpt_service_stub
        )
        if embedding is None or hypothesis_breakdown is None:
            await pipeline.initial_hypothesis_processing()  # only need to do it once
            embedding = pipeline.hypothesis.embedding
            hypothesis_breakdown = pipeline.hypothesis.hypothesis_breakdown
        else:
            pipeline.hypothesis.embedding = embedding
            pipeline.hypothesis.hypothesis_breakdown = hypothesis_breakdown

        stock_id_to_pipeline[stock_id] = pipeline

    # prepare LLM inputs
    topic_ids = [development.id for development in news_developments]
    news_topics = get_news_topics(topic_ids)
    topic_id_to_news_topic = {topic.topic_id: topic for topic in news_topics}

    summary_ids = list({point.summary_id for point in earnings_summary_points})
    earnings_topics = get_earnings_topics(summary_ids)
    id_to_earnings_topic = {
        (t.topic_id, t.summary_type.value, t.summary_index): t for t in earnings_topics
    }

    relevant_news_topics: List[CompanyNewsTopicInfo] = []
    relevant_earnings_topics: List[CompanyEarningsTopicInfo] = []
    for stock_id, pipeline in stock_id_to_pipeline.items():
        each_news_topics = []
        for dev in stock_to_developments[stock_id]:
            topic = topic_id_to_news_topic[dev.id]
            topic.gbi_id = stock_id.gbi_id  # Reset gbi_id for topics to avoid `-1` confusion
            each_news_topics.append(topic)

        logger.info(
            f"Reviewing {len(each_news_topics)} NEWS topics "
            f"for stock {stock_id.symbol} for relevance"
        )
        for start in range(0, len(each_news_topics), 100):
            news_topics_batch = each_news_topics[start : start + 100]
            news_topics_mask = await pipeline.llm.check_hypothesis_relevant_topics(
                news_topics_batch
            )
            relevant_news_topics.extend(
                [
                    topic
                    for topic, is_related in zip(news_topics_batch, news_topics_mask)
                    if is_related
                ]
            )

        earnings_point_list = stock_to_earnings_points[stock_id]
        each_earnings_topics = [
            id_to_earnings_topic[(point.summary_id, point.summary_type, point.summary_idx)]
            for point in earnings_point_list
        ]

        logger.info(
            f"Reviewing {len(each_earnings_topics)} EARNINGS topics "
            f"for stock {stock_id.symbol} for relevance"
        )
        for start in range(0, len(each_earnings_topics), 100):
            earnings_topics_batch = each_earnings_topics[start : start + 100]
            earnings_topics_mask = await pipeline.llm.check_hypothesis_relevant_topics(
                earnings_topics_batch
            )
            relevant_earnings_topics.extend(
                [
                    topic
                    for topic, is_related in zip(earnings_topics_batch, earnings_topics_mask)
                    if is_related
                ]
            )

    return relevant_news_topics, relevant_earnings_topics


async def classify_news_and_earnings_topics_into_category(
    hypothesis: str,
    categories: List[Category],
    news_topics: List[CompanyNewsTopicInfo],
    earnings_topics: List[CompanyEarningsTopicInfo],
    gbi_id_to_description: Dict[int, str],
    context: PlanRunContext,
    gpt_service_stub: Optional[GPTServiceStub] = None,
) -> Dict[int, List[Union[CompanyNewsTopicInfo, CompanyEarningsTopicInfo]]]:
    # Prompt
    main_prompt_str = """
    You are a financial analyst who is evaluating whether a news topic is directly relevant to any
    of the provided category that can be used to evaluate a hypothesis.
    The output should be in a key-value JSON format with 2 keys.
    The first key should be 'indices' and the value is a list of integers for the indices of
    the most relevant categories, e.g. '[0,1]'. The indices should be 0-based. If you find that this
    topic can't fall into any of the category, you should return an empty list.
    The second key should be 'reason' and the value is a short sentence of explanation for why you chose
    those categories.
    Your choice should be as conservative as possible and return no more than 2 relevant categories.
    When there are multiple categories that are equally relevant, you should take categories' importances
    into account and choose the more important ones.
    Here is the company description: {company_description}.
    Here is the hypothesis: {hypothesis}.
    Here are the categories and their explanations: {categories}.
    Here is the news topic: {topic}.
    """
    main_prompt = Prompt(template=main_prompt_str, name="CLASSIFY_NEWS_TOPIC_SYS_PROMPT")

    gpt_context = create_gpt_context(
        GptJobType.AGENT_TOOLS, context.agent_id, GptJobIdType.AGENT_ID
    )
    gpt = GPT(context=gpt_context, model=SONNET, gpt_service_stub=gpt_service_stub)

    # Create GPT tasks
    category_str = Category.multi_to_gpt_input(categories)

    news_tasks = []
    for news_topic in news_topics:
        company_description = gbi_id_to_description.get(news_topic.gbi_id, "No company description")
        news_tasks.append(
            gpt.do_chat_w_sys_prompt(
                main_prompt=main_prompt.format(
                    company_description=company_description,
                    hypothesis=hypothesis,
                    categories=category_str,
                    topic=news_topic.to_gpt_input(),
                ),
                sys_prompt=NO_PROMPT,
            )
        )

    earnings_tasks = []
    for earnings_topic in earnings_topics:
        company_description = gbi_id_to_description[earnings_topic.gbi_id]
        earnings_tasks.append(
            gpt.do_chat_w_sys_prompt(
                main_prompt=main_prompt.format(
                    company_description=company_description,
                    hypothesis=hypothesis,
                    categories=category_str,
                    topic=earnings_topic.to_gpt_input(),
                ),
                sys_prompt=NO_PROMPT,
            )
        )

    results: List[str] = await gather_with_concurrency(
        tasks=news_tasks + earnings_tasks, n=min(50, len(news_tasks) + len(earnings_tasks))
    )

    # Divide results
    category_to_topics: Dict[int, List[Union[CompanyNewsTopicInfo, CompanyEarningsTopicInfo]]] = (
        defaultdict(list)
    )
    for idx, result in enumerate(results):
        cleaned_result: Dict = json.loads(clean_to_json_if_needed(result))
        relevant_category_idxs: List[int] = cleaned_result.get("indices", [])
        if not relevant_category_idxs:
            continue

        if idx < len(news_tasks):
            obj = news_topics[idx]
        else:
            obj = earnings_topics[idx - len(news_tasks)]  # type: ignore

        for category_idx in relevant_category_idxs:
            category_to_topics[category_idx].append(obj)

    return category_to_topics


async def rank_and_summarize_for_each_category(
    target_stock: Optional[StockID],
    stocks: List[StockID],
    hypothesis: str,
    categories: List[Category],
    category_idx_to_topics: Dict[int, List[Union[CompanyNewsTopicInfo, CompanyEarningsTopicInfo]]],
    gbi_id_to_description: Dict[int, str],
    context: PlanRunContext,
    gpt_service_stub: Optional[GPTServiceStub] = None,
) -> Dict[int, Dict]:
    logger = get_prefect_logger(__name__)

    # FIXME: Parallelize

    # Rank + Summarize for each category
    rank_by_category_sys_prompt_str = """
        You are a financial analyst who is creating a ranking list for a few companies for a hypothesis
        on a specific financial success criteria. You will be provided with the hypothesis, the criteria,
        the descriptions of these companies, a list of companies' news topics and earnings topics relevant
        to the criteria.
        For example, if the criteria is 'Revenue', then you should look through the topics related to
        revenue, such as 'revenue growth', 'revenue diversification', 'revenue forecast', and analyze
        how well these companies perform comprehensively in the criteria.
    """
    rank_by_category_sys_prompt = Prompt(
        template=rank_by_category_sys_prompt_str, name="RANK_BY_CATEGORY_SYS_PROMPT"
    )

    rank_by_category_main_prompt_str = """
        Analyze the following information about the companies and rank them based on how well they perform \
        compared to each other.
        The output should be in the format of a deserializeable JSON (key-value pairs).
        The first key should be 'ranking' and the value should be a list of objects, where each object has \
        the following keys:
            - 'symbol': a string of the stock symbol
            - 'score': an integer in the range of 0 to 10 which represents how well the company performs \
                in the criteria. 0 means the company performs the worst, 10 means the company performs the best.
            - 'explanation': a string of of 3 to 4 sentences that explains why the company is ranked here. \
                You should conclude based on the provided descriptions and topics. You should \
                not only look at the company itself's information, but also compare it with other companies. \
                Your explanation should match the score you give to the company but not explicitly mention the score.
        The 'ranking' list should contain all companies in the order of the ranking. If there are 3 or more \
        companies in the ranking, the score of the bottom ranked company should be no higher than 3. \
        You should also be conservative about the top company's score. If it is the undoubtedly best, you should score \
        it 9 or 10. However, if it's the top 1 but doesn't show a significant lead, its score should be \
        no higher than 8. \
        The difference of two companies' scores should reflect the difference of their performance in the criteria. \
        1 point difference should represent a small difference in performance, 3 or more points difference should be \
        a significant difference. \
        If there is no topic related to any company, they should be ranked at the bottom with a score \
        of -1 to distinguish and an explanation that there is not enough information to rank the company.
        {summary_str}
        The third key should be 'citations' and the value should be a list of integers that represents the \
        indices of the topics that you used to rank the companies. The indices should be 0-based. You should \
        give at least 1 citation for low ranked companies and at least 2 citations for top ranked companies. \
        But no more than 3 citations for any company.
        Here is the hypothesis: {hypothesis}\n
        Here is the main criteria you should use to evaluate the hypothesis:{category}\n
        Here are the other criteria which are only used to provide context and should not be considered
        to rank the companies, nor mentioned in the explanation:{other_categories}\n
        Here are the companies' descriptions:\n{company_descriptions}\n
        Here are the topics you should use to rank the companies:\n{topics}\n
        Now, generate the ranking list:
    """
    rank_by_category_main_prompt = Prompt(
        template=rank_by_category_main_prompt_str, name="RANK_BY_CATEGORY_MAIN_PROMPT"
    )

    if target_stock is not None:
        summary_str = (
            "The second key should be 'summary' and the value should be a string that focuses on "
            f"the target stock ({target_stock.symbol}) to answer why it's positioned in that order. "
            "It should also try to mention other companies and concisely compare them with the target stock."
        )
    else:
        summary_str = (
            "The second key should be 'summary' and the value should be a string that focuses on "
            "the stock with the highest score to answer why it's ranked top. "
            "It should also try to mention other companies and concisely compare them with the target stock."
        )

    gpt_context = create_gpt_context(
        GptJobType.AGENT_TOOLS, context.agent_id, GptJobIdType.AGENT_ID
    )
    gpt = GPT(context=gpt_context, model=GPT4_O, gpt_service_stub=gpt_service_stub)
    gbi_id_to_stock = {stock.gbi_id: stock for stock in stocks}
    category_idx_to_result: Dict[int, Dict] = {}
    for category_idx, topics in category_idx_to_topics.items():
        logger.info(f"Ranking and summarizing for category {categories[category_idx].name}")

        gbi_ids = {t.gbi_id for t in topics}
        company_description_list = []
        for gbi_id in gbi_ids:
            stock = gbi_id_to_stock[gbi_id]
            description = gbi_id_to_description[gbi_id]
            company_description_list.append(f"Symbol: {stock.symbol}. Description: {description}")
        company_description_str = "\n".join(company_description_list)

        topics_str = "\n".join((f"- {idx}: {text}" for idx, text in enumerate(topics)))

        category_str = await categories[category_idx].to_gpt_input()

        other_categories = [categories[i] for i in range(len(categories)) if i != category_idx]
        other_categories_str = Category.multi_to_gpt_input(other_categories)

        resp = await gpt.do_chat_w_sys_prompt(
            main_prompt=rank_by_category_main_prompt.format(
                hypothesis=hypothesis,
                category=category_str,
                other_categories=other_categories_str,
                company_descriptions=company_description_str,
                topics=topics_str,
                summary_str=summary_str,
            ),
            sys_prompt=rank_by_category_sys_prompt.format(),
        )

        result = json.loads(clean_to_json_if_needed(resp))
        category_idx_to_result[category_idx] = result

    return category_idx_to_result


def calculate_weighted_average_scores(
    candidate_target_stock: Optional[StockID],
    stocks: List[StockID],
    categories: List[Category],
    category_idx_to_result: Dict[int, Dict],
) -> Tuple[StockID, Dict[str, float]]:
    scores_mapping = {}
    for category_idx, result in category_idx_to_result.items():
        weight = categories[category_idx].weight
        for ranking in result["ranking"]:
            if ranking["symbol"] not in scores_mapping:
                scores_mapping[ranking["symbol"]] = [ranking["score"] * weight, weight]
            else:
                scores_mapping[ranking["symbol"]][0] += ranking["score"] * weight
                scores_mapping[ranking["symbol"]][1] += weight

    final_scores = {symbol: score / weight for symbol, (score, weight) in scores_mapping.items()}

    if candidate_target_stock is not None:
        target_stock = candidate_target_stock
    else:
        symbol = max(final_scores.items(), key=lambda x: x[1])[0]
        symbol_to_stocks = {stock.symbol: stock for stock in stocks}
        target_stock = symbol_to_stocks[symbol]

    return target_stock, final_scores


async def overall_summary(
    target_stock: StockID,
    hypothesis: str,
    categories: List[Category],
    category_idx_to_result: Dict[int, Dict],
    final_scores: Dict[str, float],
    context: PlanRunContext,
    gpt_service_stub: Optional[GPTServiceStub] = None,
) -> str:
    # Prompt
    prompt_str = """
    You'll be given a hypothesis and your job is to interpret the hypothesis and answer it
    based on the information provided.
    The hypothesis is broken down into several categories, each with explanations, justifications, and weights.
    For each category, you will also be provided with a ranking list of the companies and the explanations
    why they are ranked in that order from the perspective of that category. You will also be provided with
    a weighted-average score for each company based on the rankings in each category.
    Return a string that consists of 3 to 5 sentences that focuses on the target company to answer the hypothesis.
    Again, the summary should be consistent with the weighted-average scores, but you should never mention the scores,
    nor repeat the hypothesis in the summary.
    Also, you should mention the categories with the high weights more often in the summary.
    You should try to also mention other companies in the summary, but not too much.
    Here is the hypothesis: {hypothesis}
    Here is the target company: {target_stock}
    Here is the weighted-average scores for each company: {scores}
    Here are the categories and their explanations: {categories}
    Here are the rankings for each category: {rankings}
    """
    prompt = Prompt(template=prompt_str, name="OVERALL_RANK_AND_SUMMARY_SYS_PROMPT")

    # Prepare prompt input
    category_str = Category.multi_to_gpt_input(categories)

    scores_str = "\n".join(
        (
            f"- {symbol}: {score}"
            for symbol, score in sorted(final_scores.items(), key=lambda x: -x[1])
        )
    )

    rankings_list = []
    for category_idx, result in category_idx_to_result.items():
        rankings_list.append(
            f"- Rankings for {categories[category_idx].name}:\n{result['ranking']}"
        )
    rankings_str = "\n".join(rankings_list)

    gpt_context = create_gpt_context(
        GptJobType.AGENT_TOOLS, context.agent_id, GptJobIdType.AGENT_ID
    )
    gpt = GPT(context=gpt_context, model=GPT4_O, gpt_service_stub=gpt_service_stub)
    resp = await gpt.do_chat_w_sys_prompt(
        main_prompt=prompt.format(
            hypothesis=hypothesis,
            target_stock=target_stock.symbol,
            scores=scores_str,
            categories=category_str,
            rankings=rankings_str,
        ),
        sys_prompt=NO_PROMPT,
    )

    return resp


async def prepare_categorical_hypothesis_outputs(
    actual_target_stock: StockID,
    final_summary: str,
    final_score: float,
    categories: List[Category],
    category_idx_to_topics: Dict[int, List[Union[CompanyNewsTopicInfo, CompanyEarningsTopicInfo]]],
    category_idx_to_result: Dict[int, Dict],
    news_developments: List[StockNewsDevelopmentText],
    earnings_summary_points: List[StockEarningsSummaryPointText],
) -> CategoricalHypothesisTexts:
    output = CategoricalHypothesisTexts(
        val=[], categories=Categories(val=categories, title="Categories")
    )

    # top summary
    output.val.append(
        Text(
            val=final_summary,
            history=[HistoryEntry(score=Score.scale_input(final_score, lb=0, ub=10))],
            title="Summary",  # FIXME: use GPT to generate a title
        )
    )

    # each category ranking widget
    id_to_news_dev = {dev.id: dev for dev in news_developments}
    key_to_earnings_points = {
        (obj.summary_id, obj.summary_type, obj.summary_idx): obj for obj in earnings_summary_points
    }
    for category_idx, result in sorted(
        category_idx_to_result.items(), key=lambda x: -categories[x[0]].weight
    ):
        # score
        score = 0
        for each_ranking in result["ranking"]:
            if each_ranking["symbol"] == actual_target_stock.symbol:
                score = each_ranking["score"]
                break

        # text
        rankings_list = [f"{result['summary']}"]
        for each_ranking in result["ranking"]:
            rankings_list.append(
                (
                    f"- {each_ranking['symbol']}\n"
                    f"  - Score: {each_ranking['score']}\n"
                    f"  - Explanation: {each_ranking['explanation']}"
                )
            )
        rankings_list_str = "\n".join(rankings_list)

        # citations
        citation_idxs = result["citations"]
        topics = category_idx_to_topics[category_idx]
        citations = []
        for idx in citation_idxs:
            topic = topics[idx]
            if isinstance(topic, CompanyNewsTopicInfo):
                dev = id_to_news_dev[topic.topic_id]
                citations.append(TextCitation(source_text=dev))
            elif isinstance(topic, CompanyEarningsTopicInfo):
                key = (topic.topic_id, topic.summary_type.value, topic.summary_index)
                point = key_to_earnings_points[key]
                citations.append(TextCitation(source_text=point))

        output.val.append(
            Text(
                val=rankings_list_str,
                history=[HistoryEntry(score=Score.scale_input(score, lb=0, ub=10), citations=citations)],  # type: ignore # noqa
                title=f"Analysis - {categories[category_idx].name}",
            )
        )

    return output


if __name__ == "__main__":

    async def main(hypothesis: str, main_stock: str) -> None:
        import datetime

        from agent_service.tools.category import CategoriesForStockInput, get_categories
        from agent_service.tools.lists import CombineListsInput, add_lists
        from agent_service.tools.other_text import (
            GetAllTextDataForStocksInput,
            get_all_text_data_for_stocks,
        )
        from agent_service.tools.peers import (
            GeneralPeersForStockInput,
            get_general_peers,
        )
        from agent_service.tools.stocks import (
            StockIdentifierLookupInput,
            stock_identifier_lookup,
        )

        context = PlanRunContext.get_dummy(user_id="6953b640-16f9-4757-914e-02de6b79fab4")

        # Get stock ID and peers' IDs
        stock_id = await stock_identifier_lookup(
            StockIdentifierLookupInput(stock_name=main_stock), context
        )
        peers = await get_general_peers(GeneralPeersForStockInput(stock_id=stock_id), context)  # type: ignore # noqa
        stocks = await add_lists(CombineListsInput(list1=[stock_id], list2=peers), context)  # type: ignore # noqa

        all_texts: List[StockText] = await get_all_text_data_for_stocks(  # type: ignore
            GetAllTextDataForStocksInput(stock_ids=stocks, start_date=datetime.date(2024, 4, 1)),  # type: ignore # noqa
            context,
        )

        categories = await get_categories(CategoriesForStockInput(prompt=hypothesis), context)

        output = await analyze_hypothesis_with_categories(
            AnalyzeHypothesisWithCategoriesInput(
                hypothesis=hypothesis,
                categories=categories,  # type: ignore
                stocks=stocks,  # type: ignore
                all_text_data=all_texts,
                target_stock=stock_id,  # type: ignore
            ),
            context,
        )
        print(output)

    import asyncio

    from agent_service.utils.logs import init_stdout_logging

    init_stdout_logging()
    asyncio.run(main(hypothesis="Is NVDA the leader in the AI chips space?", main_stock="NVDA"))
