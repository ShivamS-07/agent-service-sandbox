import json
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union

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
    StockEarningsText,
    StockNewsDevelopmentText,
    StockSecFilingSectionText,
    StockSecFilingText,
    StockText,
    Text,
    TextCitation,
)
from agent_service.tool import ToolArgs, ToolCategory, ToolRegistry, tool
from agent_service.tools.category import Category
from agent_service.tools.tool_log import tool_log
from agent_service.types import PlanRunContext
from agent_service.utils.async_utils import gather_with_concurrency
from agent_service.utils.gpt_logging import GptJobIdType, GptJobType, create_gpt_context
from agent_service.utils.postgres import get_psql
from agent_service.utils.prefect import get_prefect_logger
from agent_service.utils.prompt_utils import Prompt
from agent_service.utils.string_utils import clean_to_json_if_needed


@io_type
class MixedTopics(ComplexIOBase):
    # all these objects must have content filled in
    news_topics: List[StockNewsDevelopmentText] = []
    earnings_topics: List[StockEarningsSummaryPointText] = []
    sec_filing_sections: List[StockSecFilingSectionText] = []

    async def to_gpt_input(self, use_abberivated_output: bool = True) -> str:
        self._cut_input_topics(target_num=200)

        texts = []
        idx = 0
        for topic in self.sec_filing_sections:
            text = await topic.to_gpt_input()
            texts.append(f"- {idx}: ({topic.stock_id.symbol}) {text}")  # type: ignore
            idx += 1

        for topic in self.earnings_topics:
            texts.append(f"- {idx}: ({topic.stock_id.symbol}) {topic.val}")  # type: ignore
            idx += 1

        for topic in self.news_topics:
            texts.append(f"- {idx}: ({topic.stock_id.symbol}) {topic.val}")  # type: ignore
            idx += 1

        return "\n".join(texts)

    def _cut_input_topics(self, target_num: int = 200) -> None:
        # if there are too many topics passed to GPT, it may cause the output being chopped off
        # so we need to reduce the number of topics
        if target_num >= self.total:
            return

        # if there are too many news topics, we only reduce the number of news topics
        if len(self.news_topics) > self.total * 0.6:
            num_news_to_del = self.total - target_num
            news_ratio = 1 - num_news_to_del / len(self.news_topics)
            stock_to_news_topics: Dict[StockID, List[StockNewsDevelopmentText]] = defaultdict(list)
            for topic in self.news_topics:
                if topic.stock_id:
                    stock_to_news_topics[topic.stock_id].append(topic)

            self.news_topics = []
            for topics in stock_to_news_topics.values():
                # FIXME: should sort these news topics by date and then cut once we have that info
                self.news_topics.extend(topics[: int(len(topics) * news_ratio)])
            return

        # otherwise we reduce the number of topics for each type and each company proportionally
        stock_to_topics: Dict[StockID, MixedTopics] = defaultdict(MixedTopics)
        for topic in self.news_topics + self.earnings_topics + self.sec_filing_sections:  # type: ignore
            if topic.stock_id:
                stock_to_topics[topic.stock_id].insert_topic(topic)

        # reset the lists
        ratio = target_num / self.total
        self.news_topics = []
        self.earnings_topics = []
        self.sec_filing_sections = []
        for mixed_topics in stock_to_topics.values():
            if len(mixed_topics.news_topics) > mixed_topics.total * 0.6:
                # if this company has too many news topics, we also only reduce news topics
                mixed_topics._cut_input_topics(target_num=int(mixed_topics.total * ratio))
                num_news_to_keep = len(mixed_topics.news_topics)
                num_earnings_to_keep = len(mixed_topics.earnings_topics)
                num_filings_to_keep = len(mixed_topics.sec_filing_sections)
            else:
                # otherwise, proportionally keep at least 1 topic for each type
                num_news_to_keep = max(1, int(len(mixed_topics.news_topics) * ratio))
                num_earnings_to_keep = max(1, int(len(mixed_topics.earnings_topics) * ratio))
                num_filings_to_keep = max(1, int(len(mixed_topics.sec_filing_sections) * ratio))

            self.news_topics.extend(mixed_topics.news_topics[:num_news_to_keep])
            self.earnings_topics.extend(mixed_topics.earnings_topics[:num_earnings_to_keep])
            self.sec_filing_sections.extend(mixed_topics.sec_filing_sections[:num_filings_to_keep])

    @property
    def total(self) -> int:
        return len(self.news_topics) + len(self.earnings_topics) + len(self.sec_filing_sections)

    def insert_topic(
        self,
        topic: Union[
            StockNewsDevelopmentText, StockEarningsSummaryPointText, StockSecFilingSectionText
        ],
    ) -> None:
        if isinstance(topic, StockNewsDevelopmentText):
            self.news_topics.append(topic)
        elif isinstance(topic, StockEarningsSummaryPointText):
            self.earnings_topics.append(topic)
        else:
            self.sec_filing_sections.append(topic)

    def get_topic(
        self, idx: int
    ) -> Union[StockNewsDevelopmentText, StockEarningsSummaryPointText, StockSecFilingSectionText]:
        if idx < len(self.sec_filing_sections):
            return self.sec_filing_sections[idx]
        idx -= len(self.sec_filing_sections)
        if idx < len(self.earnings_topics):
            return self.earnings_topics[idx]
        idx -= len(self.earnings_topics)
        return self.news_topics[idx]


@io_type
class HypothesisAnalysisByCategory(ComplexIOBase):
    actual_target_stock: StockID
    final_scores: Dict[str, float]
    ranked_categories: List[Category]
    category_idx_to_topics: Dict[int, MixedTopics]
    category_idx_to_result: Dict[int, Dict]
    val: List[Text] = []

    async def split_into_components(self) -> List[IOType]:
        self.prepare_categorical_hypothesis_outputs()
        return self.val  # type: ignore

    def prepare_categorical_hypothesis_outputs(self) -> None:
        # each category ranking widget
        for category_idx, result in sorted(
            self.category_idx_to_result.items(), key=lambda x: -self.ranked_categories[x[0]].weight
        ):
            # score
            score = 0
            for each_ranking in result["ranking"]:
                if each_ranking["symbol"] == self.actual_target_stock.symbol:
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
            mixed_topics = self.category_idx_to_topics[category_idx]
            citations = []
            for idx in citation_idxs:
                topic = mixed_topics.get_topic(idx)
                topic.reset_value()  # remove the content to save DB space
                citations.append(TextCitation(source_text=topic))

            self.val.append(
                Text(
                    val=rankings_list_str,
                    history=[
                        HistoryEntry(
                            score=Score.scale_input(score, lb=0, ub=10), citations=citations  # type: ignore
                        )
                    ],
                    title=f"Analysis - {self.ranked_categories[category_idx].name}",
                )
            )


class AnalyzeHypothesisWithCategoriesInput(ToolArgs):
    hypothesis: str
    categories: List[Category]
    stocks: List[StockID]
    all_text_data: List[StockText]
    # if only 1 stock mentioned in `hypothesis`, it will be assigned. If no stock or more than 1
    # stock are mentioned, it will be None.
    target_stock: Optional[StockID] = None


@tool(
    description=(
        "Given a list of relevant text data and a list of categories used to break market analysis down, "
        "this function generates scores and their explanations for each category indicating the extent to which "
        "it is supported with reference to the information in the provided text data."
        "This analysis must be focused on a specific stock or a small group of stocks, this function "
        "must NOT be used to filter stocks more generally! (i.e. Do not use it for "
        " `Give/find me stocks...` type queries, use the filter by profile tool). "
        "If a specific company is mentioned in the user input, then an identifier for that "
        "company MUST be passed as the target stock. Only leave target_stock empty if no "
        "company or more than one company is mentioned in the user question/hypothesis. "
        "For example, if the question is, `Is NVDA a leader in AI chips?, you MUST pass NVDA in "
        "as the target stock. But if the question is `Who is the leader in AI chips?`, "
        "or `Is NVDA or AMD the leader in AI chips?`, you must leave target_stock empty."
    ),
    category=ToolCategory.HYPOTHESIS,
    tool_registry=ToolRegistry,
)
async def analyze_hypothesis_with_categories(
    args: AnalyzeHypothesisWithCategoriesInput, context: PlanRunContext
) -> HypothesisAnalysisByCategory:
    logger = get_prefect_logger(__name__)

    # Step: Download content for text data
    logger.info("Downloading content for text data")
    (
        gbi_id_to_short_description,
        news_devs_with_text,
        earnings_points_with_text,
        sec_filings_sections_with_text,
    ) = await download_content_for_text_data(args.all_text_data)

    stocks = list(
        {t.stock_id for t in news_devs_with_text if t.stock_id}
        | {t.stock_id for t in earnings_points_with_text if t.stock_id}
        | {t.stock_id for t in sec_filings_sections_with_text if t.stock_id}
    )

    gpt_service_stub = _get_gpt_service_stub()[0]

    # Step: Revise hypothesis to remove company specific information
    logger.info("Revising hypothesis to remove any company specific information")
    revised_hypothesis = await revise_hypothesis(args.hypothesis, context, gpt_service_stub)
    logger.info(f"Revised hypothesis: {revised_hypothesis}")  # Not visible to users

    # Step: Classify topics into categories
    logger.info("Classifying news, earnings, SEC topics into categories")
    categories = sorted(args.categories, key=lambda x: x.weight, reverse=True)
    category_idx_to_mixed_topics = await filter_and_classify_topics_into_category(
        revised_hypothesis,
        categories,
        gbi_id_to_short_description,
        news_devs_with_text,
        earnings_points_with_text,
        sec_filings_sections_with_text,
        context,
        gpt_service_stub,
    )

    # Step: Rank and summarize for each category
    logger.info("Ranking and summarizing for each category")
    candidate_target_stock = args.target_stock
    category_to_result = await rank_and_summarize_for_each_category(
        candidate_target_stock,
        stocks,
        revised_hypothesis,
        categories,
        category_idx_to_mixed_topics,
        gbi_id_to_short_description,
        context,
        gpt_service_stub,
    )

    # Step: Calculate weighted average scores and determine the real target stock
    actual_target_stock, total_scores = calculate_weighted_average_scores(
        candidate_target_stock, stocks, categories, category_to_result
    )

    # Step: Prepare outputs
    output = HypothesisAnalysisByCategory(
        actual_target_stock=actual_target_stock,
        final_scores=total_scores,
        ranked_categories=categories,
        category_idx_to_topics=category_idx_to_mixed_topics,
        category_idx_to_result=category_to_result,
    )

    return output


class GenerateSummaryForHypothesisWithCategoriesInput(ToolArgs):
    hypothesis: str
    hypothesis_analysis_by_category: HypothesisAnalysisByCategory


@tool(
    description=(
        "Given a market analysis broken down by categories, for a specific stock or group of stocks, "
        "this function generates a short summary for the analysis. This function must be called after "
        "analyze_hypothesis_with_categories."
    ),
    category=ToolCategory.HYPOTHESIS,
    tool_registry=ToolRegistry,
)
async def generate_summary_for_hypothesis_with_categories(
    args: GenerateSummaryForHypothesisWithCategoriesInput, context: PlanRunContext
) -> Text:
    logger = get_prefect_logger(__name__)

    gpt_service_stub = _get_gpt_service_stub()[0]

    logger.info("Generating the final summary")
    final_summary = await overall_summary(
        target_stock=args.hypothesis_analysis_by_category.actual_target_stock,
        hypothesis=args.hypothesis,
        categories=args.hypothesis_analysis_by_category.ranked_categories,
        category_idx_to_result=args.hypothesis_analysis_by_category.category_idx_to_result,
        final_scores=args.hypothesis_analysis_by_category.final_scores,
        context=context,
        gpt_service_stub=gpt_service_stub,
    )
    await tool_log(log="Generated the final summary", context=context)
    final_score = args.hypothesis_analysis_by_category.final_scores[
        args.hypothesis_analysis_by_category.actual_target_stock.symbol  # type: ignore
    ]
    return Text(
        val=final_summary, history=[HistoryEntry(score=Score.scale_input(final_score, lb=0, ub=10))]
    )


####################################################################################################
# Utils
####################################################################################################
async def download_content_for_text_data(all_text_data: List[StockText]) -> Tuple[
    Dict[int, str],
    List[StockNewsDevelopmentText],
    List[StockEarningsSummaryPointText],
    List[StockSecFilingSectionText],
]:
    """
    1. Download the actual text content for these text objects
    2. Assign the text content to the corresponding text objects's `val` field
    """

    logger = get_prefect_logger(__name__)

    logger.info("Separate text data by sources")
    company_descriptions = [t for t in all_text_data if isinstance(t, StockDescriptionText)]

    news_developments = [t for t in all_text_data if isinstance(t, StockNewsDevelopmentText)]

    # Currently ignoring earning texts containing transcripts instead of summaries
    earnings_summaries: List[StockEarningsText] = [
        t for t in all_text_data if isinstance(t, StockEarningsSummaryText)
    ]
    earnings_summary_points = await StockEarningsSummaryPointText.init_from_earnings_texts(
        earnings_summaries
    )

    # it does the downloading of the content inside `init_from_filings`
    logger.info("Downloading content for 10K/10Q and split into sections")
    sec_filings = [t for t in all_text_data if isinstance(t, StockSecFilingText)]
    sec_filing_sections_with_text = await StockSecFilingSectionText.init_from_filings(sec_filings)
    logger.info(
        f"Got {len(sec_filing_sections_with_text)} sections from {len(sec_filings)} filings"
    )

    logger.info("Downloading content for company descriptions")
    gbi_ids = [t.stock_id.gbi_id for t in company_descriptions if t.stock_id]
    descriptions = get_psql().get_short_company_descriptions_for_gbi_ids(gbi_ids)
    gbi_id_to_short_description = {
        gbi_id: desc for gbi_id, (desc, _) in descriptions.items() if desc
    }

    # NOTE: meaningless to parallelize as they are using the same PSQL object
    logger.info("Downloading content for news developments")
    topic_id_to_news_dev = await StockNewsDevelopmentText._get_strs_lookup(news_developments)
    news_devs_with_text = []
    for news_dev in news_developments:
        if news_dev.id in topic_id_to_news_dev:
            news_dev.val = topic_id_to_news_dev[news_dev.id]
            news_devs_with_text.append(news_dev)

    logger.info("Downloading content for earnings summary points")
    hash_id_to_earnings_point = await StockEarningsSummaryPointText._get_strs_lookup(
        earnings_summary_points
    )
    earnings_points_with_text = []
    for earnings_point in earnings_summary_points:
        if earnings_point.id in hash_id_to_earnings_point:
            earnings_point.val = hash_id_to_earnings_point[earnings_point.id]
            earnings_points_with_text.append(earnings_point)

    return (
        gbi_id_to_short_description,
        news_devs_with_text,
        earnings_points_with_text,
        sec_filing_sections_with_text,
    )


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


async def filter_and_classify_topics_into_category(
    hypothesis: str,
    categories: List[Category],
    gbi_id_to_description: Dict[int, str],
    news_devs_with_text: List[StockNewsDevelopmentText],
    earnings_points_with_text: List[StockEarningsSummaryPointText],
    sec_filings_sections_with_text: List[StockSecFilingSectionText],
    context: PlanRunContext,
    gpt_service_stub: Optional[GPTServiceStub] = None,
) -> Dict[int, MixedTopics]:
    logger = get_prefect_logger(__name__)

    # Prompt
    main_prompt_str = """
You are a financial analyst who is evaluating whether a text about a stock is directly relevant to any \
of the provided categories that can be used to answer a question. \
For example, if the question is 'Who is the leader in the industry?' and you have one category \
`Market Share` and the text says 'Company A's CEO's wealth increased due to the increase in \
company's market cap', then the point of the text is the CEO's wealth, and it is not relevant to \
the category 'Market Share'. However, a text which says 'Company A's sales up compared to competitors' \
is relevant to `Market Share`.
All changes related to a particular category are equally relevant. For example, in the example above \
you must say relevant to both texts which implying 'High Market Share' as well as 'Low Market Share' \
You absolutely must not think a text is relevant just because the text and the category are both good (or bad) \
for the company, there must be a clear topical connection between the text and the category \
that goes well beyond a general positive or negative effect on the company.
Unless there is a specific category associated with stock performance, a text which refers to the \
company's stock performance or stock split in general is NOT relevant. For example, topics talking about \
stock prices, stock splits, market cap/value, or stock performance in general are NOT relevant to any \
other categories than stock performance category. Read and interpret the text carefully! \
Topics like 'Stock Price Record Highs', 'Market Cap surged to $1 trillion' are ONLY relevant to stock \
performance, and should not fall into other categories even if they mention some seemingly relevant keywords.
If the category is about Innovation or Efficiency, topics with concrete examples are relevant, \
which should usually mention a specific product or service of the company, or mention the comparison \
or improvement to the predecessors. \
Topics that mention more than one company are preferred in general. You should tell if they are competitors \
or partners, and classify them into the relevant categories like 'Competition', 'Ecosystem', 'Partership', etc.
You should be very conservative, you must say irrelevant if there's no clear reason that there might \
be a measurable connection between the text and the categories.
For example, if there is a category about a specific product, texts that indicate a change in supply, \
demand, or cost of that product would all be considered directly relevant.
The output should be in a key-value JSON format with 2 keys.
The first key should be 'reason' and the value is a short sentence of explanation for what categories \
you chose and why. Absolutely no more than 50 words. You must explitly say whether the text is primarily \
related to stock performance or not. \
The second key should be 'relevant_categories' and the value is a list of integers for the indices of \
the MOST relevant categories, e.g. '[0,3]'. You must NOT include this topic if you think it's primarily \
related to stock performance. Do not choose more than 2 categories. The indices should be 0-based. \
If this text is not relevant to any of the categories, this should be any empty list.
Most texts will be relevant to no categories, or only one. You should be very conservative about including more \
than one.
Here is the company description:
{company_description}.
Here is the question:
{hypothesis}.
Here are the categories and their explanations:
{categories}
Here is the text:
{topic}
    """
    main_prompt = Prompt(template=main_prompt_str, name="CLASSIFY_NEWS_TOPIC_SYS_PROMPT")

    gpt_context = create_gpt_context(
        GptJobType.AGENT_TOOLS, context.agent_id, GptJobIdType.AGENT_ID
    )
    gpt = GPT(context=gpt_context, model=SONNET, gpt_service_stub=gpt_service_stub)

    # Create GPT tasks
    # Don't want categories' weights to confuse GPT
    category_str = Category.multi_to_gpt_input(categories, include_weight=False)

    news_tasks = []
    for news_dev in news_devs_with_text:
        company_description = gbi_id_to_description.get(
            news_dev.stock_id.gbi_id, "No company description"  # type: ignore
        )
        news_tasks.append(
            gpt.do_chat_w_sys_prompt(
                main_prompt=main_prompt.format(
                    company_description=company_description,
                    hypothesis=hypothesis,
                    categories=category_str,
                    topic=news_dev.val,
                ),
                sys_prompt=NO_PROMPT,
            )
        )

    earnings_tasks = []
    for earnings_point in earnings_points_with_text:
        company_description = gbi_id_to_description[earnings_point.stock_id.gbi_id]  # type: ignore
        earnings_tasks.append(
            gpt.do_chat_w_sys_prompt(
                main_prompt=main_prompt.format(
                    company_description=company_description,
                    hypothesis=hypothesis,
                    categories=category_str,
                    topic=earnings_point.val,
                ),
                sys_prompt=NO_PROMPT,
            )
        )

    filings_tasks = []
    for filings_section in sec_filings_sections_with_text:
        company_description = gbi_id_to_description[filings_section.stock_id.gbi_id]  # type: ignore
        text = await filings_section.to_gpt_input()
        filings_tasks.append(
            gpt.do_chat_w_sys_prompt(
                main_prompt=main_prompt.format(
                    company_description=company_description,
                    hypothesis=hypothesis,
                    categories=category_str,
                    topic=text,
                ),
                sys_prompt=NO_PROMPT,
            )
        )

    all_tasks = news_tasks + earnings_tasks + filings_tasks
    all_objects = news_devs_with_text + earnings_points_with_text + sec_filings_sections_with_text
    results: List[str] = await gather_with_concurrency(tasks=all_tasks, n=min(50, len(all_tasks)))

    # Divide results
    category_to_mixed_topics: Dict[int, MixedTopics] = defaultdict(MixedTopics)
    for text_obj, result in zip(all_objects, results):
        try:
            cleaned_result: Dict = json.loads(clean_to_json_if_needed(result))
        except Exception as e:
            # we should not fail the whole process if one of the results is not parseable
            logger.warning(f"Failed to parse result {result} for text object {text_obj}: {e}")
            continue

        if not isinstance(cleaned_result, dict):
            continue

        relevant_category_idxs: List[int] = cleaned_result.get("relevant_categories", [])
        if not relevant_category_idxs:
            continue

        for category_idx in relevant_category_idxs:
            category_to_mixed_topics[category_idx].insert_topic(text_obj)

    category_topic_log = ""
    for category_idx, topics in category_to_mixed_topics.items():
        category = categories[category_idx]
        category_topic_log += f"Category <{category.name}>: {topics.total} topics; "
    await tool_log(log=category_topic_log, context=context)

    return category_to_mixed_topics


async def rank_and_summarize_for_each_category(
    target_stock: Optional[StockID],
    stocks: List[StockID],
    hypothesis: str,
    categories: List[Category],
    category_idx_to_mixed_topics: Dict[int, MixedTopics],
    gbi_id_to_description: Dict[int, str],
    context: PlanRunContext,
    gpt_service_stub: Optional[GPTServiceStub] = None,
) -> Dict[int, Dict]:
    logger = get_prefect_logger(__name__)

    # FIXME: Parallelize

    # Rank + Summarize for each category
    rank_by_category_sys_prompt_str = """
        You are a financial analyst who is creating a ranking list for the provided companies for a hypothesis
        on a specific financial success criteria. You must stay very strictly in the context of the main criteria
        to test the hypothesis and rank the companies. You should not consider any other criteria and must not
        mention other criteria.
        You will be provided with the hypothesis, the criteria, the descriptions of these companies,
        a list of companies' news topics and earnings topics relevant to the criteria.
        You will be also provided with a list of other criteria, only for the purpose of context. You should
        never use them to rank the companies. And you must not mention them in the explanation.
        I'll say it again, focus on the main criteria!
        For example, if the criteria is 'Revenue', then you should look through the topics related to
        revenue, such as 'revenue growth', 'revenue diversification', 'revenue forecast', and analyze
        how well these companies perform comprehensively in the criteria.
        Your arguments should be supported by the evidence from the topics and try to concisely mention
        numbers, specs, or comparisons in your summary or explanation. For example, '27% faster', '100X more' are
        good proof to support your arguments.
    """
    rank_by_category_sys_prompt = Prompt(
        template=rank_by_category_sys_prompt_str, name="RANK_BY_CATEGORY_SYS_PROMPT"
    )

    rank_by_category_main_prompt_str = """
        Analyze the following information about the companies and rank them based on how well they perform \
        compared to each other.
        The output should be in the format of a deserializeable JSON (key-value pairs).
        The first key should be 'ranking' and the value should be a list of objects, where each object has \
        the following 4 keys and DO NOT include extra keys that are not mentioned:
            - 'symbol': a string of the stock symbol
            - 'score': an integer in the range of 0 to 10 which represents how well the company performs \
                in the criteria. 0 means the company performs the worst, 10 means the company performs the best. \
                The score should be comparable to other companies in the ranking. You should take all companies \
                into account.
            - 'explanation': a string of medium sized paragraph that consists of 6 to 8 sentences that explains \
                why the company is ranked here within the context of the main criteria, followed by the detailed \
                evidence that supports your arguments. \
                For example, a good explanation should be like 'Apple dominates the phone chip in terms of Innovation \
                because the latest A15 chip is 20% faster than the previous A14 chip, and 2X faster than all \
                the other competitors' phone chips', which makes a good point of Apple leads the phone chip in the \
                criteria of innovation' and supported by the facts that it's making better chip than its own previous \
                version but even way better than the competitors. If you see topics like this, you should try to \
                include it in your explanation. \
                You must consider all the provided information. You should not only focus on this company's topics, \
                but also look at other companies' topics and analyze them together. For example, if a topic like \
                'Samsung's latest chip is 10% faster than the predecessor, chasing Apple's A15 chip'. This topic \
                should be considered when ranking both Apple and Samsung, and of course beneficial to Samsung's score \
                but not to Apple.
                I'll say it again, your conclusion should be derived from the topics with very concrete examples and \
                focus solely on the main criteria. Topics that contain general information should be ignored. \
                Topics that mention comparisons with predecessors, or with other competitors's similar products \
                or services are preferred to include in the explanation. If there are exact numbers, or specs \
                that can demonstrate the improvement or decline in the criteria, you MUST mention them \
                in the explanation. Try to mention 4 to 5 topics that are most relevant to the ranking. \
                Your explanation should match the score you give to the company but not explicitly mention the score. \
                The top ranked companies MUST specify why and where they are better than the others, and the bottom \
                ranked companies MUST specify why and where they fall behind than the higher ranked ones. \
            - 'citations': a list of integers that represents the indices of the topics that you used to \
                rank the company. Return them from the most relevant to the least relevant. No more than 5. \
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
        indices of the topics that you used to rank the companies. The indices should be 0-based. \
        You should review all the provided topics carefully and cite the MOST RELEVANT ones. \
        The order of the citations should be from the most relevant to the least relevant. \
        You should include at least 1 citation for low ranked companies and at least 2 citations for top \
        ranked companies. NO MORE than 3 citations for any company, so the total number of citations \
        should be NO MORE THAN {total_citations}.
        Here is the hypothesis: {hypothesis}\n
        Here is the main criteria you should use to evaluate the hypothesis:{category}\n
        Here are the other criteria which are only used to provide context and should not be considered
        to rank the companies. You must not mention them in the explanation:{other_categories}\n
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

    total_citations = 3 * len(stocks)

    company_description_list = []
    for stock in stocks:
        description = gbi_id_to_description[stock.gbi_id]  # type: ignore
        company_description_list.append(f"Symbol: {stock.symbol}. Description: {description}")
    company_description_str = "\n".join(company_description_list)

    gpt_context = create_gpt_context(
        GptJobType.AGENT_TOOLS, context.agent_id, GptJobIdType.AGENT_ID
    )
    gpt = GPT(context=gpt_context, model=GPT4_O, gpt_service_stub=gpt_service_stub)
    category_idx_to_result: Dict[int, Dict] = {}
    for category_idx, mixed_topics in category_idx_to_mixed_topics.items():
        logger.info(f"Ranking and summarizing for category <{categories[category_idx].name}>")

        topics_str = await mixed_topics.to_gpt_input()
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
                total_citations=total_citations,
            ),
            sys_prompt=rank_by_category_sys_prompt.format(),
        )

        result = json.loads(clean_to_json_if_needed(resp))
        category_idx_to_result[category_idx] = result

    await tool_log(
        log="Ranked all the relevant stocks for each category and created summaries",
        context=context,
    )

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

        output: HypothesisAnalysisByCategory = await analyze_hypothesis_with_categories(
            AnalyzeHypothesisWithCategoriesInput(
                hypothesis=hypothesis,
                categories=categories,  # type: ignore
                stocks=stocks,  # type: ignore
                all_text_data=all_texts,
                target_stock=stock_id,  # type: ignore
            ),
            context,
        )
        summary = await generate_summary_for_hypothesis_with_categories(
            GenerateSummaryForHypothesisWithCategoriesInput(
                hypothesis=hypothesis, hypothesis_analysis_by_category=output
            ),
            context,
        )
        print(summary)

    import asyncio

    from agent_service.utils.logs import init_stdout_logging

    init_stdout_logging()
    asyncio.run(main(hypothesis="Is NVDA the leader in the AI chips space?", main_stock="NVDA"))
