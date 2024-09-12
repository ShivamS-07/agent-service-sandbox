import json
import re
from collections import defaultdict
from typing import Any, Awaitable, Dict, List, Optional, Tuple, Type, cast

from gpt_service_proto_v1.service_grpc import GPTServiceStub

from agent_service.GPT.constants import GPT4_O, GPT4_O_MINI, NO_PROMPT
from agent_service.GPT.requests import GPT, _get_gpt_service_stub
from agent_service.io_type_utils import (
    ComplexIOBase,
    HistoryEntry,
    IOType,
    Score,
    dump_io_type,
    io_type,
    load_io_type,
)
from agent_service.io_types.stock import StockID
from agent_service.io_types.text import (
    CustomDocumentSummaryText,
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
    TextGroup,
)
from agent_service.planner.errors import EmptyInputError
from agent_service.tool import (
    TOOL_DEBUG_INFO,
    ToolArgs,
    ToolCategory,
    ToolRegistry,
    tool,
)
from agent_service.tools.category import (
    Category,
    CriteriaForCompetitiveAnalysis,
    get_criteria_for_competitive_analysis,
)
from agent_service.tools.LLM_analysis.prompts import CITATION_PROMPT, CITATION_REMINDER
from agent_service.tools.LLM_analysis.utils import (
    extract_citations_from_gpt_output,
    get_all_text_citations,
    get_second_order_citations,
)
from agent_service.tools.tool_log import tool_log
from agent_service.types import PlanRunContext
from agent_service.utils.async_utils import (
    gather_dict_as_completed,
    gather_with_concurrency,
    identity,
)
from agent_service.utils.check_cancelled import check_cancelled_decorator
from agent_service.utils.gpt_logging import GptJobIdType, GptJobType, create_gpt_context
from agent_service.utils.postgres import get_psql
from agent_service.utils.prefect import get_prefect_logger
from agent_service.utils.prompt_utils import Prompt
from agent_service.utils.string_utils import clean_to_json_if_needed
from agent_service.utils.tool_diff import get_prev_run_info

RELEVANT_CATEGORIES = "relevant_categories"

RANKING_HEADER = "Ranking List:\n"
SUMMARY_HEADER = "Summary:\n"

# Templates
COMPANY_DESCRIPTION_TEMPLATE = (
    "Company Name: {company_name}. Symbol: {symbol}. Description: {description}"
)
CATEGORY_RANKING_BULLET_POINT = "- {company_name} ({symbol}) Score {score}\n    - {explanation}\n"
FINAL_SUMMARY_WEIGHTED_SCORE = "- {symbol}: {score}"
FINAL_SUMMARY_CATEGORY_RANKING = "# Rankings for category <{category_name}>:\n{ranking}"

####################################################################################################
# Prompt
#########################################################################################################


RANK_BY_CATEGORY_EXPLANATION = """
The output should be in a long text format that includes the ranking list and a summary. \
The first section should start with a text 'Ranking List:'. In the new line list the companies \
in the markdown format. And the second section should start with a text 'Summary:', \
followed by a paragraph summarizing the ranking list. Each section should be separated by '\n\n'. \
For example, it should look like below:

Ranking List:
- <Company Name1> (<symbol1>) Score <score1>
    - <explanation1>
- <Company Name2> (<symbol2>) Score <score2>
    - <explanation2>
...
Summary:
<summary>

In detail, the ranking list should be in descending order by the companies' performance in the criteria.
'score' is a float in the range of 0 to 5 which represents how \
well the company performs in the criteria. 0 means the company performs the worst, 5 means the company \
performs the best. The score should be comparable to other companies in the ranking. You should take all \
companies into account.
'explanation' is a medium-sized paragraph that consists of 6 to 8 sentences that explains why the company \
is ranked here within the context of the main criteria, followed by the detailed evidence that supports \
your arguments. For example, a good explanation should be like 'Apple dominates the phone chip in terms \
of Innovation because the latest A15 chip is 20% faster than the previous A14 chip, and 2X faster than all \
the other competitors' phone chips', which makes the good point that Apple leads the phone chip in the criteria \
of innovation and supported by the facts that it's making better chip than its own previous version but even \
way better than the competitors. If you see source documents like this, you should try to include them in your
explanation. \
You must consider all the provided information. You should not only focus on source documents associated with the
current company, but also look at documents for other companies and analyze them together.
For example, if a source document says 'Samsung's latest chip is 10% faster than the predecessor, getting close to
Apple'sA15 chip'.This source document should be considered when ranking both Apple and Samsung, and of course
beneficial to Samsung's score but not to Apple. I'll say it again, your conclusion should be derived from the
source doument with very concrete examples and focused solely on the main criteria. Source documents that contain
general information should be ignored. Source documents that mention comparisons with predecessors, or with other
competitors's similar products or services are the most relevant, and must be included in the explanation. If there
are exact numbers, or specs that can demonstrate the improvement or decline in the criteria, you MUST mention them \
in the explanation. Try to mention 4 to 5 source documents that are most relevant to the ranking. \
Your explanation should match the score you give to the company but not explicitly mention the score. \
The top ranked companies MUST specify why and where they are better than the others, and the bottom \
ranked companies MUST specify why and where they fall behind than the higher ranked ones.
The 'ranking' list should contain all companies in the order of the ranking. If there are 3 or more \
companies in the ranking, the score of the bottom ranked company should be no higher than 1.5. \
You should also be conservative about the top company's score. If it is the undoubtedly best, the score \
should be 4.5 or above. However, if the top 1 doesn't show a significant leadership, its score should be \
no higher than 4. \
The difference of two companies' scores should reflect the difference of their performance in the criteria. \
0.5 point difference should represent a small difference in performance, 1.5 or more points difference \
should be a significant difference.
{summary_str}
"""


RANK_BY_CATEGORY_SYS_PROMPT_STR = """
You are a financial analyst who is creating a ranked list of companies for a competitive analysis
focused on a specific criterion. You must stay very strictly focused on the provided criterion. You should
not consider any other criteria and must not mention other criteria.
You will be provided with the prompt for the competitive analysis, the criterion, descriptions of these companies,
and a list of source documents that are relevant to the companies and (usually) the criterion. You should ignore
source documents that you don't think are actually relevant to the criterion.
You will be also provided with a list of other criteria, only for the purpose of context. You should
never use them to rank the companies, and you must not mention them in the explanation.
I'll say it again, focus on the target criterion!
For example, if the criterion is 'Revenue', then you should look through the source documents related to
revenue, such as 'revenue growth', 'revenue diversification', 'revenue forecast', and analyze
how well these companies perform comprehensively with regards to this criterion.
Your arguments should be supported by the evidence from the source documents. Try to concisely mention
numbers, specs, or comparisons in your summary or explanation.
For example, '27% faster', '100X more' examples of good evidence to support your arguments.
"""
RANK_BY_CATEGORY_SYS_PROMPT = Prompt(
    template=RANK_BY_CATEGORY_SYS_PROMPT_STR + CITATION_PROMPT, name="RANK_BY_CATEGORY_SYS_PROMPT"
)

UPDATE_RANK_BY_CATEGORY_SYS_PROMPT_STR = """
You are a financial analyst who is updating a ranked list of companies and an explanation for that
ranking. This work is part of an overall competitive analysis, but you are currently focused on a specific criterion.
You will be provided with the general prompt for the competitive analysis, the criterion, descriptions of
these companies, the previous ranking along with expanations, a list of source documents that you used when generating
the previous ranking, and new source documents that you must consider integrating into the updated ranking. You should
ignore source documents that you don't think are actually relevant to the current criterion. You will also be provided
with a list of other criteria, but only for the purpose of context. You should never use them to rank the companies,
and you must not mention them in the explanation. I'll say it again, focus on the target criterion!

Follow the specific instructions for writing such a ranking, provided further below. Since you are updating the
ranking, there are a few extra guidelines you must follow:
- You must ignore the specific ways that the previous ranking is not in line with the instructions. In particular,
in the old ranking, the summary is found above the detailed ranking, but you will write the detailed ranking
first. Most importantly, the old ranking does not include citations, but you must be very rigorous about citations.
Detailed instructions for how to do citations are also found below, make sure you must cite one of the source documents
for every single fact included in your final updated ranking. Follow the all instructions below carefully! Following the
instructions takes precedence over the previous ranking.
- You might find that for some facts included in the earlier ranking, there are no provided source documents related
to them. You must not include in your new ranking facts in the earlier ranking that you no longer have a citations for!
Just omit the discussion of that topic in the new ranking. Your old ranking should never be considered a source
document.
- If there are no new source documents with important information relevant to this particular company and the current
criterion, you should copy the text of your previous ranking for that company, though don't forget to add citations,
and to leave out any information from the original ranking where there is no source document to back it up.
- If there are new source documents and at least one of those documents provides useful information that informs the
ranking of the relevant company, then you should add that information to the explanation of the ranking (along with
citations!). However, be fairly conservative about changing the scores. The score should only be changed when the
information you are adding provides a significant new perspective on one or more of the relevant companies. It is fine
to add new information into the ranking explanation without changing the score.
- You must be EXTREMELY conservative about any major changes to the ranking of companies, only do it when there is major
new information available that has really shaken up the associated market.
- The summary should be updated based on relevant information that has been added or removed from the detailed rankings.
Do not, however, change the wording to make it sound like an update (say `X is the leader`, not `X remains the leader`)
unless a major change has occurred.

Here are the general instructions for writing these per-criterion competitive analysis rankings:
"""

UPDATE_RANK_BY_CATEGORY_SYS_PROMPT_STR += RANK_BY_CATEGORY_EXPLANATION

UPDATE_RANK_BY_CATEGORY_SYS_PROMPT_STR += "Here are the instructions for how to do citations:\n"

UPDATE_RANK_BY_CATEGORY_SYS_PROMPT_STR += CITATION_PROMPT

UPDATE_RANK_BY_CATEGORY_SYS_PROMPT = Prompt(
    template=UPDATE_RANK_BY_CATEGORY_SYS_PROMPT_STR, name="UPDATE RANK_BY_CATEGORY_SYS_PROMPT"
)

RANK_BY_CATEGORY_MAIN_PROMPT_STR = (
    "Analyze the following information about the companies and rank them based on how well they perform"
    "compared to each other."
)
RANK_BY_CATEGORY_MAIN_PROMPT_STR += RANK_BY_CATEGORY_EXPLANATION

RANK_BY_CATEGORY_MAIN_PROMPT_STR += """
Here is the original prompt: {hypothesis}\n
Here is the main criterion you should use to evaluate the hypothesis:{category}\n
Here are the other criteria which are only used to provide context and should not be considered
to rank the companies. You must not mention them in the explanation:{other_categories}\n
Here are the companies' descriptions, delimited by ---:
---
{company_descriptions}
---
Here are the source documents you should use to rank the companies, also delimited by ---:
---
{topics}
---
"""

RANK_BY_CATEGORY_MAIN_PROMPT_STR += CITATION_REMINDER

RANK_BY_CATEGORY_MAIN_PROMPT_STR += " Now write your ranking with summary and citations:\n"

RANK_BY_CATEGORY_MAIN_PROMPT = Prompt(
    template=RANK_BY_CATEGORY_MAIN_PROMPT_STR,
    name="RANK_BY_CATEGORY_MAIN_PROMPT",
)

UPDATE_RANK_BY_CATEGORY_MAIN_PROMPT_STR = """
Update this ranking with new information provided in the source document

Here is the original prompt: {hypothesis}

Here is the main criterion you should focus on:
{category}

Here are the other criteria which are only used to provide context and should not be considered
when ranking the companies here. You must not mention them in the explanation:
{other_categories}

Here are the companies' descriptions, delimited by ---:
---
{company_descriptions}
---

Here are the new source documents that you should consider including, also delimited by ---:
---
{new_topics}
---

Here are old source documents, used in the original ranking text, also delimited by ---:
---
{old_topics}
---

And here is the previous ranking text, which you should be largely copying except to add citations, and
when there is new information provided by the new source documents:
---
{old_ranking}
---
"""

UPDATE_RANK_BY_CATEGORY_MAIN_PROMPT_STR += CITATION_REMINDER

UPDATE_RANK_BY_CATEGORY_MAIN_PROMPT_STR += "Now rewrite the ranking, with summary and citations:\n"


UPDATE_RANK_BY_CATEGORY_MAIN_PROMPT = Prompt(
    template=UPDATE_RANK_BY_CATEGORY_MAIN_PROMPT_STR,
    name="UPDATE_RANK_BY_CATEGORY_MAIN_PROMPT",
)


SUMMARY_STR_WTARGET = (
    "The summmary should focus on the target stock ({stock}), justifying why it is positioned in that order. "
    "It should also try to mention other companies and concisely compare them with the target stock."
)

SUMMARY_STR_NOTARGET = (
    "The summmary should focus on the highest ranking stock, justifying why it is positioned in that order. "
    "It should also try to mention other companies and concisely compare them with the highest ranked stock."
)


####################################################################################################
# Classes
####################################################################################################
@io_type
class TopicGroup(TextGroup):

    def convert_to_gpt_input(self) -> str:
        # we don't care about the args here, just for mypy

        if not self.id_to_str:  # TopicGroups must have their id_to_str set separately
            raise RuntimeError("tried to Topic Group to string before deriving id_to_str")
        self._cut_input_topics(target_num=200)
        return super().convert_to_str(self.id_to_str, numbering=True, symbols=True)

    def _cut_input_topics(self, target_num: int = 200) -> None:
        # if there are too many topics passed to GPT, it may cause the output being chopped off
        # so we need to reduce the number of topics
        if target_num >= self.total:
            return

        # if there are too many news topics, we only reduce the number of news topics
        news_topics = self.get_certain_type_topics(StockNewsDevelopmentText)
        custom_doc_summaries = self.get_certain_type_topics(CustomDocumentSummaryText)
        earnings_points = self.get_certain_type_topics(StockEarningsSummaryPointText)
        filing_sections = self.get_certain_type_topics(StockSecFilingSectionText)

        num_news_topics = len(news_topics)
        if num_news_topics > self.total * 0.6:
            num_news_to_del = self.total - target_num
            news_ratio = 1 - num_news_to_del / len(news_topics)
            stock_to_news_topics: Dict[StockID, List[StockNewsDevelopmentText]] = defaultdict(list)
            for topic in news_topics:
                if topic.stock_id:
                    stock_to_news_topics[topic.stock_id].append(topic)  # type: ignore

            filtered_news_topics = []
            for topics in stock_to_news_topics.values():
                filtered_news_topics.extend(topics[: int(len(topics) * news_ratio)])

            self.val = filtered_news_topics + custom_doc_summaries + earnings_points + filing_sections  # type: ignore
            self.sync_map_with_val()

            return

        # otherwise we reduce the number of topics for each type and each company proportionally
        stock_to_topic_group: Dict[StockID, TopicGroup] = {}
        for topic in self.val:
            if topic.stock_id:
                if topic.stock_id not in stock_to_topic_group:
                    stock_to_topic_group[topic.stock_id] = TopicGroup(val=[])

                stock_to_topic_group[topic.stock_id].val.append(topic)

        # reset the lists
        ratio = target_num / self.total
        all_topics = []
        for topic_group in stock_to_topic_group.values():
            this_news_topics = topic_group.get_certain_type_topics(StockNewsDevelopmentText)
            if len(this_news_topics) > topic_group.total * 0.6:
                # if this company has too many news topics, we also only reduce news topics
                topic_group._cut_input_topics(target_num=int(topic_group.total * ratio))
                all_topics.extend(topic_group.val)
            else:
                # otherwise, proportionally keep at least 1 topic for each type
                this_earnings_points = topic_group.get_certain_type_topics(
                    StockEarningsSummaryPointText
                )
                this_filing_sections = topic_group.get_certain_type_topics(
                    StockSecFilingSectionText
                )
                this_custom_doc_summaries = topic_group.get_certain_type_topics(
                    CustomDocumentSummaryText
                )

                num_news_to_keep = max(1, int(len(this_news_topics) * ratio))
                num_earnings_to_keep = max(1, int(len(this_earnings_points) * ratio))
                num_filings_to_keep = max(1, int(len(this_filing_sections) * ratio))
                num_custom_docs_to_keep = max(1, int(len(this_custom_doc_summaries) * ratio))
                all_topics.extend(
                    this_news_topics[:num_news_to_keep]
                    + this_custom_doc_summaries[:num_custom_docs_to_keep]
                    + this_earnings_points[:num_earnings_to_keep]
                    + this_filing_sections[:num_filings_to_keep]
                )

        self.val = all_topics
        self.sync_map_with_val()

    @property
    def total(self) -> int:
        return len(self.val)

    def sync_map_with_val(self) -> None:
        """
        Sync `self.val` and `self.id_to_str`
        When we are modifying `self.val` (usually cutting the number of topics), we need to update
        `self.id_to_str` accordingly. Or drop the topics that are not in `self.id_to_str`.
        """

        if not self.id_to_str:
            return

        texts = []
        mapping = {}
        for topic in self.val:
            if topic.id in self.id_to_str:
                texts.append(topic)
                mapping[topic.id] = self.id_to_str[topic.id]

        self.val = texts
        self.id_to_str = mapping

    def get_all_stocks(self) -> List[StockID]:
        return list({topic.stock_id for topic in self.val if topic.stock_id})

    def get_certain_type_topics(self, topic_type: Type[Text]) -> List[Text]:
        return [t for t in self.val if isinstance(t, topic_type)]


@io_type
class RankedCompany(ComplexIOBase):
    # internal use only
    symbol: str
    company_name: str
    score: float
    explanation: str

    @classmethod
    def create_default_obj(
        cls, symbol: str, company_name: str, category_name: str
    ) -> "RankedCompany":
        return cls(
            symbol=symbol,
            company_name=company_name,
            score=0.0,
            explanation=(
                f"There is no information available regarding {symbol} in terms of {category_name}. "
                f"Therefore, it is impossible to rank the company based on the provided criteria."
            ),
        )

    def __str__(self) -> str:
        return CATEGORY_RANKING_BULLET_POINT.format(
            company_name=self.company_name,
            symbol=self.symbol,
            score=self.score,
            explanation=self.explanation,
        )


@io_type
class RankingList(ComplexIOBase):
    val: str  # well-formatted markdown (summary + ranking)
    ranking: List[RankedCompany]

    def __str__(self) -> str:
        return self.val


@io_type
class CompetitiveAnalysis(ComplexIOBase):
    actual_target_stock: StockID
    final_scores: Dict[str, float]
    ranked_categories: List[Category]
    category_idx_to_topic_group: Dict[int, TopicGroup]
    category_idx_to_result: Dict[int, RankingList]
    val: List[Text] = []

    async def split_into_components(self) -> List[IOType]:
        if not self.val:
            self.prepare_categorical_hypothesis_outputs()
        return self.val  # type: ignore

    def prepare_categorical_hypothesis_outputs(self) -> None:
        for category_idx, category in enumerate(self.ranked_categories):
            if category_idx not in self.category_idx_to_result:
                self.val.append(
                    Text(
                        val="Sorry, there is no relevant information found for this category",
                        history=[HistoryEntry(score=Score(val=0))],
                        title=f"Analysis - {category.name}",
                    )
                )
                continue

            result = self.category_idx_to_result[category_idx]

            # score
            score = 0.0
            for each_ranking in result.ranking:
                if each_ranking.symbol == self.actual_target_stock.symbol:
                    score = each_ranking.score
                    break

            citations = result.history[0].citations
            for c in citations:
                if isinstance(c, TextCitation) or hasattr(c, "source_text"):
                    c.source_text.reset_value()  # remove the content to save DB space

            self.val.append(
                Text(
                    val=result.val,
                    history=[
                        HistoryEntry(
                            score=Score.scale_input(score, lb=0, ub=5), citations=citations
                        )
                    ],
                    title=f"Analysis - {self.ranked_categories[category_idx].name}",
                )
            )


####################################################################################################
# Tools
####################################################################################################


# for backward compatibility
class AnalyzeHypothesisWithCategoriesInput(ToolArgs):
    hypothesis: str
    categories: List[Category]
    stocks: List[StockID]
    all_text_data: List[StockText]
    # if only 1 stock mentioned in `hypothesis`, it will be assigned. If no stock or more than 1
    # stock are mentioned, it will be None.
    target_stock: Optional[StockID] = None


@tool(description="", category=ToolCategory.COMPETITIVE_ANALYSIS, enabled=False)
async def analyze_hypothesis_with_categories(
    args: AnalyzeHypothesisWithCategoriesInput, context: PlanRunContext
) -> CompetitiveAnalysis:
    return await do_competitive_analysis(  # type: ignore
        args=DoCompetitiveAnalysisInput(
            prompt=args.hypothesis,
            criteria=args.categories,
            stocks=args.stocks,
            all_text_data=args.all_text_data,
            target_stock=args.target_stock,
        ),
        context=context,
    )


class DoCompetitiveAnalysisInput(ToolArgs):
    prompt: str
    criteria: List[Category]
    stocks: List[StockID]
    all_text_data: List[StockText]
    target_stock: Optional[StockID] = None


@tool(
    description=(
        "This is the main tool that carries out a competitive market analysis to evaluate the relative "
        "market position of a group of companies which sell comparable products or services.\n"
        "The `prompt` is a string which is represents the question or request from the client that "
        "forms the basis for the competitive analysis, e.g. `Is NVDA the leader in the AI chip space?` "
        "It always contains at least one of two components: the mention of a market (e.g. AI chips) or the "
        "mention of a company in that market (e.g. NVDA), and in many cases it will have both of these things.\n"
        "The `criteria` are the criteria for judging the companies during the analysis and are derived from the "
        "get_criteria_for_competitive_analysis tool. You must ALWAYS provide a list of criteria for any "
        "competitive analysis, this list cannot be empty!\n"
        "`stocks` should be a list of the stocks (companies) which will be evaluated as part of the analysis. "
        "There are three potential sources of such stocks. If the client specifically lists a group of stocks "
        "to be used in the analysis, you should take them from there. Otherwise, if the market is a particular "
        "class of product or service, you should derive the stocks using the filter_stocks_by_product_or_service"
        "tool. Finally, if the client requests a particular sector of stocks, you use the sector filtering tool.\n"
        "`all_text_data` is the output of the all_text_data retrieval tool called over the `stocks` to "
        "be included in this competitive analysis.\n"
        "If a single specific company is mentioned in the user input, then an identifier for that "
        "company MUST be passed as the target stock. "
        "For example, if the question is, `Is NVDA a leader in AI chips?, you MUST pass NVDA's identifier in "
        "as the target stock. "
        "Do not leave target_stock empty unless there are no specific companies mentioned!!!! "
        "If there is a target stock and you are filtering to get its peers as `stocks`, you should "
        "filter its peers from a popular stock index that contains the stock, e.g. if the stock was "
        "Carrefour you would choose the Stoxx 600 as your default stock universe."
        "This tool is not useful for applications other than market comparisons. It must only be used "
        "in situations which are focused on grading/ranking all companies which produce a product. "
        "An example of a situation you would NOT use this is due diligence checklists."
        "This function is intended to provide an thorough analysis of the relative market position of "
        "a small group of stocks in a single market. This function must NOT be used to filter stocks! "
        "(i.e. Do not use it for `Give/find me stocks...` type queries, use a stock filter tool). "
        "If the client is interested in ranking stocks by an easily quantifiable statistic, you should NOT "
        "use this tool, instead get the statistics and rank using the transform_table tool. Competitive analysis "
        "tools should be used only for a complex qualitative analysis. "
        "If the client is interested in comparing the contents of two specific documents or groups of documents, "
        "even if the comparison involves two different stocks, you should NOT use this tool, instead you "
        "should use the compare_texts tool. You should only use this competitive analysis tool when a "
        "comprehensive market analysis of two or more stocks is desired."
        "The output of this tool is scores and explanation of those across each criterion and each stock, "
        "indicating the relative ranking of these stocks in each evaluative category."
    ),
    category=ToolCategory.COMPETITIVE_ANALYSIS,
    tool_registry=ToolRegistry,
)
async def do_competitive_analysis(
    args: DoCompetitiveAnalysisInput, context: PlanRunContext
) -> CompetitiveAnalysis:
    logger = get_prefect_logger(__name__)

    debug_info: Dict[str, Any] = {}
    TOOL_DEBUG_INFO.set(debug_info)

    if not args.stocks:
        raise EmptyInputError("At least one stock must be provided to do competative analysis")
    elif not args.criteria:
        raise EmptyInputError("At least one criterion must be provided to do competative analysis")
    elif not args.all_text_data:
        raise EmptyInputError("At least one text data must be provided to do competative analysis")

    prev_run_info = None
    prev_internal_data = None
    try:  # since everything associated with diffing is optional, put in try/except
        if context.diff_info is not None:  # running updates
            prev_run_info = await get_prev_run_info(context, "do_competitive_analysis")
            if prev_run_info is not None:
                prev_args = DoCompetitiveAnalysisInput.model_validate_json(prev_run_info.inputs_str)
                # when doing automations, categories CANNOT change, so making sure of that
                args.criteria = prev_args.criteria
                prev_internal_data: Dict[str, str] = prev_run_info.debug  # type:ignore

    except Exception as e:
        logger.warning(f"Error getting info from previous run: {e}")

    # Step: Download content for text data
    logger.info("Downloading content for text data")
    gbi_id_to_short_description, topic_group = await download_content_for_text_data(
        context=context, all_text_data=args.all_text_data
    )

    debug_info["input_topics"] = dump_io_type(topic_group.val)

    gpt_service_stub = _get_gpt_service_stub()[0]

    if prev_internal_data and "revised hypothesis" in prev_internal_data:
        logger.info("Using existing revised hypothesis")
        revised_hypothesis = prev_internal_data["revised hypothesis"]
    else:
        # Step: Revise hypothesis to remove company specific information
        logger.info("Revising hypothesis to remove any company specific information")
        revised_hypothesis = await revise_hypothesis(
            context=context, hypothesis=args.prompt, gpt_service_stub=gpt_service_stub
        )
        logger.info(f"Revised hypothesis: {revised_hypothesis}")  # Not visible to users

    debug_info["revised hypothesis"] = revised_hypothesis

    orig_topic_group = None

    if prev_internal_data and "input_topics" in prev_internal_data:
        logger.info("Identifying new topics that need to be classified")
        old_input_topics = load_io_type(prev_internal_data["input_topics"])
        for topic in old_input_topics:
            topic.reset_id()  # otherwise ids are different for snippets
        old_input_topic_set = set(old_input_topics)
        new_topics = []
        for topic in topic_group.val:
            if topic not in old_input_topic_set:
                new_topics.append(topic)

        debug_info["new topics"] = dump_io_type(new_topics)  # what topics appear in this run

        orig_topic_group = topic_group
        topic_group = TopicGroup(val=new_topics, id_to_str=topic_group.id_to_str)

    # Step: Classify topics into categories
    logger.info("Classifying news, earnings, SEC topics into categories")
    categories = sorted(args.criteria, key=lambda x: x.weight, reverse=True)
    category_idx_to_topic_group = await filter_and_classify_topics_into_category(
        context=context,
        hypothesis=revised_hypothesis,
        categories=categories,
        gbi_id_to_description=gbi_id_to_short_description,
        topic_group=topic_group,
        gpt_service_stub=gpt_service_stub,
    )

    # Step: Rank and summarize for each category
    logger.info("Ranking and summarizing for each category")
    candidate_target_stock = args.target_stock

    stocks = args.stocks[:]
    if candidate_target_stock and candidate_target_stock not in stocks:
        stocks.append(candidate_target_stock)

    if prev_internal_data and orig_topic_group and "categories and ranks" in prev_internal_data:
        old_rankings = [
            tuple(pair) for pair in load_io_type(prev_internal_data["categories and ranks"])
        ]

        category_to_result = await update_rank_and_summarize_for_each_category(
            context=context,
            target_stock=candidate_target_stock,
            stocks=stocks,
            hypothesis=revised_hypothesis,
            categories=categories,
            new_category_idx_to_topic_group=category_idx_to_topic_group,
            old_rankings=old_rankings,
            orig_topic_group=orig_topic_group,
            gbi_id_to_description=gbi_id_to_short_description,
            gpt_service_stub=gpt_service_stub,
        )
    else:
        category_to_result = await rank_and_summarize_for_each_category(
            context=context,
            target_stock=candidate_target_stock,
            stocks=stocks,
            hypothesis=revised_hypothesis,
            categories=categories,
            category_idx_to_topic_group=category_idx_to_topic_group,
            gbi_id_to_description=gbi_id_to_short_description,
            gpt_service_stub=gpt_service_stub,
        )

    debug_info["categories and ranks"] = dump_io_type(
        [[category_idx, ranked_list] for category_idx, ranked_list in category_to_result.items()]
    )

    # Step: Calculate weighted average scores and determine the real target stock
    actual_target_stock, total_scores = calculate_weighted_average_scores(
        candidate_target_stock, stocks, categories, category_to_result
    )

    # Step: Prepare outputs
    output = CompetitiveAnalysis(
        actual_target_stock=actual_target_stock,
        final_scores=total_scores,
        ranked_categories=categories,
        category_idx_to_topic_group=category_idx_to_topic_group,
        category_idx_to_result=category_to_result,
    )

    return output


# backwards compatibility
class GenerateSummaryForHypothesisWithCategoriesInput(ToolArgs):
    hypothesis: str
    hypothesis_analysis_by_category: CompetitiveAnalysis


@tool(
    description="",
    category=ToolCategory.COMPETITIVE_ANALYSIS,
    tool_registry=ToolRegistry,
    enabled=False,
)
async def generate_summary_for_hypothesis_with_categories(
    args: GenerateSummaryForHypothesisWithCategoriesInput, context: PlanRunContext
) -> Text:
    return await generate_summary_for_competitive_analysis(  # type: ignore
        args=GenerateSummaryForCompetitiveAnalysisInput(
            prompt=args.hypothesis, competitive_analysis=args.hypothesis_analysis_by_category
        ),
        context=context,
    )


class GenerateSummaryForCompetitiveAnalysisInput(ToolArgs):
    prompt: str
    competitive_analysis: CompetitiveAnalysis


@tool(
    description=(
        "Given a competitive market analysis for group of stocks, this function generates a short "
        "text summary of the conclusions of the analysis. This prompt should be the same prompt passed "
        "to the do_competitive_analysis tool, and the competitive_analysis should be its output. In "
        "general, this tool will always be used directly after `do_competitive_analysis`. If there "
        "was a target stock in the analysis, this tool will also output an overall score for that stock."
    ),
    category=ToolCategory.COMPETITIVE_ANALYSIS,
    tool_registry=ToolRegistry,
)
async def generate_summary_for_competitive_analysis(
    args: GenerateSummaryForCompetitiveAnalysisInput, context: PlanRunContext
) -> Text:
    logger = get_prefect_logger(__name__)

    gpt_service_stub = _get_gpt_service_stub()[0]

    logger.info("Generating the final summary")
    final_summary = await overall_summary(
        target_stock=args.competitive_analysis.actual_target_stock,
        hypothesis=args.prompt,
        categories=args.competitive_analysis.ranked_categories,
        category_idx_to_result=args.competitive_analysis.category_idx_to_result,
        final_scores=args.competitive_analysis.final_scores,
        context=context,
        gpt_service_stub=gpt_service_stub,
    )

    try:
        final_summary_citations = await get_second_order_citations(
            final_summary, get_all_text_citations(args.competitive_analysis.val), context
        )
    except Exception as e:
        logger.exception(f"Failed to add second order citations: {e}")
        final_summary_citations = []

    await tool_log(log="Generated the final summary", context=context)

    target_stock_symbol = args.competitive_analysis.actual_target_stock.symbol
    final_scores = args.competitive_analysis.final_scores
    if target_stock_symbol not in final_scores:
        logger.warning(
            f"The actual target stock ({target_stock_symbol}) is not in the final scores "
            "large likely due to the lack of data. Set the final score to 0"
        )
        final_score = 0.0
    else:
        final_score = final_scores[target_stock_symbol]

    return Text(
        val=final_summary,
        history=[
            HistoryEntry(
                score=Score.scale_input(final_score, lb=0, ub=5),
                citations=final_summary_citations,  # type:ignore
            )
        ],
    )


####################################################################################################
# Utils
####################################################################################################
@check_cancelled_decorator
async def download_content_for_text_data(
    context: PlanRunContext,
    all_text_data: List[StockText],
) -> Tuple[Dict[int, str], TopicGroup]:
    """
    1. Download the actual text content for these text objects
    2. Assign the text content to the corresponding text objects's `val` field
    """

    logger = get_prefect_logger(__name__)

    logger.info("Separate text data by sources")
    company_descriptions = [t for t in all_text_data if isinstance(t, StockDescriptionText)]
    news_developments = [t for t in all_text_data if isinstance(t, StockNewsDevelopmentText)]
    earnings_summaries: List[StockEarningsText] = [
        t for t in all_text_data if isinstance(t, StockEarningsSummaryText)
    ]
    sec_filings = [t for t in all_text_data if isinstance(t, StockSecFilingText)]
    custom_doc_summaries = [t for t in all_text_data if isinstance(t, CustomDocumentSummaryText)]

    logger.info("Downloading content for company descriptions")
    gbi_ids = [t.stock_id.gbi_id for t in company_descriptions if t.stock_id]
    descriptions = get_psql().get_short_company_descriptions_for_gbi_ids(gbi_ids)
    gbi_id_to_short_description = {
        gbi_id: desc for gbi_id, (desc, _) in descriptions.items() if desc
    }

    # Currently ignoring earning texts containing transcripts instead of summaries
    logger.info("Converting earnings summaries to points")
    earnings_summary_points = await StockEarningsSummaryPointText.init_from_earnings_texts(
        earnings_summaries
    )

    # it does the downloading of the content inside `init_from_filings`
    logger.info("Downloading content for 10K/10Q and split into sections")
    sec_filing_sections_with_text = await StockSecFilingSectionText.init_from_filings(sec_filings)
    logger.info(
        f"Got {len(sec_filing_sections_with_text)} sections from {len(sec_filings)} filings"
    )

    # sort the list by the timestamp for later cutting
    text_list = (
        news_developments
        + custom_doc_summaries
        + earnings_summary_points
        + sec_filing_sections_with_text
    )
    text_list.sort(key=lambda x: x.timestamp, reverse=True)  # type: ignore
    topic_group = TopicGroup(val=text_list)  # type: ignore
    await Text.get_all_strs(topic_group)  # fill in `id_to_str` dict
    topic_group.sync_map_with_val()

    return gbi_id_to_short_description, topic_group


@check_cancelled_decorator
async def revise_hypothesis(
    context: PlanRunContext, hypothesis: str, gpt_service_stub: Optional[GPTServiceStub] = None
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


@check_cancelled_decorator
async def filter_and_classify_topics_into_category(
    context: PlanRunContext,
    hypothesis: str,
    categories: List[Category],
    gbi_id_to_description: Dict[int, str],
    topic_group: TopicGroup,
    gpt_service_stub: Optional[GPTServiceStub] = None,
) -> Dict[int, TopicGroup]:
    logger = get_prefect_logger(__name__)

    if topic_group.total == 0:
        await tool_log(log="No topics to process", context=context)
        return {}

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
related to stock performance. Do not choose more than 2 categories. The indices should be 0-based \
and the maximum index of the categories is {max_category_idx}. I'll say it again, DO NOT exceed the bounds! \
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
    gpt = GPT(context=gpt_context, model=GPT4_O_MINI, gpt_service_stub=gpt_service_stub)

    # Create GPT tasks
    # Don't want categories' weights to confuse GPT
    category_str = Category.multi_to_gpt_input(categories, include_weight=False)

    task_to_text_obj: Dict[Awaitable[str], Text] = {}
    for text_obj in topic_group.val:
        if not text_obj.stock_id:
            continue

        text_str = topic_group.get_str_for_text(text_obj.id)
        if not text_str:
            continue

        company_description = gbi_id_to_description.get(
            text_obj.stock_id.gbi_id, "No company description"
        )
        task = gpt.do_chat_w_sys_prompt(
            main_prompt=main_prompt.format(
                company_description=company_description,
                hypothesis=hypothesis,
                categories=category_str,
                topic=text_str,
                max_category_idx=len(categories) - 1,
            ),
            sys_prompt=NO_PROMPT,
        )
        task_to_text_obj[task] = text_obj

    category_idx_to_topic_group: Dict[int, TopicGroup] = {}
    async for text_obj, result in gather_dict_as_completed(task_to_text_obj, n=300):
        if text_obj is None or result is None:
            continue

        try:
            cleaned_result = json.loads(clean_to_json_if_needed(result))
            if not isinstance(cleaned_result, dict):
                continue
        except Exception as e:
            # we should not fail the whole process if one of the results is not parseable
            logger.warning(f"Failed to parse result {result} for text object {text_obj}: {e}")
            continue

        # no single gpt calls should fail the whole process
        try:
            relevant_category_idxs: List[int] = cleaned_result.get(RELEVANT_CATEGORIES, [])
            if not relevant_category_idxs:
                continue

            if isinstance(relevant_category_idxs, str):
                # '[0,1,2]' -> [0, 1, 2]
                relevant_category_idxs = json.loads(relevant_category_idxs)

            for category_idx in relevant_category_idxs:
                if category_idx < 0 or category_idx >= len(categories):
                    logger.warning(
                        f"Category index {category_idx} is out of range for {text_obj} {result}"
                    )
                    continue

                if category_idx not in category_idx_to_topic_group:
                    category_idx_to_topic_group[category_idx] = TopicGroup(val=[])

                category_idx_to_topic_group[category_idx].val.append(text_obj)
        except Exception as e:
            logger.exception(f"Failed to process result {result} for text object {text_obj}: {e}")

    category_topic_log = ""
    for category_idx, smaller_topic_group in category_idx_to_topic_group.items():
        # prepare tool log
        category = categories[category_idx]
        num_topics = len(smaller_topic_group.val)
        category_topic_log += f"Category <{category.name}>: {num_topics} topics; "

        # fill in `id_to_str` from the bigger text group
        smaller_topic_group.id_to_str = topic_group.id_to_str
        smaller_topic_group.sync_map_with_val()

    await tool_log(log=category_topic_log, context=context)

    return category_idx_to_topic_group


@check_cancelled_decorator
async def rank_and_summarize_for_each_category(
    context: PlanRunContext,
    target_stock: Optional[StockID],
    stocks: List[StockID],
    hypothesis: str,
    categories: List[Category],
    category_idx_to_topic_group: Dict[int, TopicGroup],
    gbi_id_to_description: Dict[int, str],
    gpt_service_stub: Optional[GPTServiceStub] = None,
) -> Dict[int, RankingList]:
    if target_stock is not None:

        summary_str = SUMMARY_STR_WTARGET.format(stock=target_stock.symbol)
    else:
        summary_str = SUMMARY_STR_NOTARGET

    gpt_context = create_gpt_context(
        GptJobType.AGENT_TOOLS, context.agent_id, GptJobIdType.AGENT_ID
    )
    gpt = GPT(context=gpt_context, model=GPT4_O, gpt_service_stub=gpt_service_stub)

    sorted_category_idxs = sorted(category_idx_to_topic_group.keys())
    category_tasks = []
    for category_idx in sorted_category_idxs:
        topic_group = category_idx_to_topic_group[category_idx]

        category_tasks.append(
            _rank_by_gpt_and_post_process(
                gpt,
                hypothesis,
                stocks,
                summary_str,
                category_idx,
                categories,
                topic_group,
                gbi_id_to_description,
                context,
            )
        )

    results = await gather_with_concurrency(category_tasks, n=len(category_tasks))

    await tool_log(
        log="Ranked all the relevant stocks for each category and created summaries",
        context=context,
    )

    category_idx_to_result: Dict[int, RankingList] = {
        idx: result for idx, result in zip(sorted_category_idxs, results) if result is not None
    }
    return category_idx_to_result


@check_cancelled_decorator
async def update_rank_and_summarize_for_each_category(
    context: PlanRunContext,
    target_stock: Optional[StockID],
    stocks: List[StockID],
    hypothesis: str,
    categories: List[Category],
    new_category_idx_to_topic_group: Dict[int, TopicGroup],
    old_rankings: List[Tuple[int, RankingList]],
    orig_topic_group: TopicGroup,
    gbi_id_to_description: Dict[int, str],
    gpt_service_stub: Optional[GPTServiceStub] = None,
) -> Dict[int, RankingList]:
    if target_stock is not None:

        summary_str = SUMMARY_STR_WTARGET.format(stock=target_stock.symbol)
    else:
        summary_str = SUMMARY_STR_NOTARGET

    gpt_context = create_gpt_context(
        GptJobType.AGENT_TOOLS, context.agent_id, GptJobIdType.AGENT_ID
    )
    gpt = GPT(context=gpt_context, model=GPT4_O, gpt_service_stub=gpt_service_stub)

    category_tasks = []
    for category_idx, old_ranking in old_rankings:
        if category_idx not in new_category_idx_to_topic_group:
            category_tasks.append(identity(old_ranking))

        else:
            new_topic_group = new_category_idx_to_topic_group[category_idx]
            category_tasks.append(
                _update_rank_by_gpt_and_post_process(
                    gpt,
                    hypothesis,
                    stocks,
                    summary_str,
                    category_idx,
                    categories,
                    new_topic_group,
                    orig_topic_group,
                    old_ranking,
                    gbi_id_to_description,
                    context,
                )
            )

    results = await gather_with_concurrency(category_tasks, n=len(category_tasks))

    await tool_log(
        log="Updated ranking for all the relevant stocks for each category and created summaries",
        context=context,
    )

    category_idx_to_result: Dict[int, RankingList] = {
        idx: result for (idx, _), result in zip(old_rankings, results) if result is not None
    }
    return category_idx_to_result


async def _rank_by_gpt_and_post_process(
    gpt: GPT,
    hypothesis: str,
    all_stocks: List[StockID],
    summary_str: str,
    category_idx: int,
    categories: List[Category],
    topic_group: TopicGroup,
    gbi_id_to_description: Dict[int, str],
    context: PlanRunContext,
) -> Optional[RankingList]:
    logger = get_prefect_logger(__name__)

    category_name = categories[category_idx].name

    try:
        ####################
        # Prepare GPT inputs
        ####################
        logger.info(f"Preparing GPT inputs to rank category <{category_name}>")

        # topics
        topics_str = topic_group.convert_to_gpt_input()

        # company descriptions
        involved_stocks = topic_group.get_all_stocks()
        company_description_str = "\n".join(
            COMPANY_DESCRIPTION_TEMPLATE.format(
                company_name=stock.company_name,
                symbol=stock.symbol,
                description=gbi_id_to_description[stock.gbi_id],
            )
            for stock in involved_stocks
        )

        # categories
        category_str = await categories[category_idx].to_gpt_input()
        other_categories = [categories[i] for i in range(len(categories)) if i != category_idx]
        other_categories_str = Category.multi_to_gpt_input(other_categories)

        ####################
        # Call GPT
        ####################
        logger.info(f"Calling GPT to rank category <{category_name}>")
        result = await gpt.do_chat_w_sys_prompt(
            main_prompt=RANK_BY_CATEGORY_MAIN_PROMPT.format(
                hypothesis=hypothesis,
                category=category_str,
                other_categories=other_categories_str,
                company_descriptions=company_description_str,
                topics=topics_str,
                summary_str=summary_str,
            ),
            sys_prompt=RANK_BY_CATEGORY_SYS_PROMPT.format(),
        )

        ##############################
        # Extract citations from text
        #############################
        logger.info(f"Extracting inline citations for category <{category_name}>")
        ranking_list_w_summary = await _rank_output_postprocess(
            result, category_name, topic_group, all_stocks, context
        )

        await tool_log(
            log=f"Finished ranking and summarizing for category <{category_name}>",
            context=context,
        )

        return ranking_list_w_summary
    except Exception as e:
        logger.exception(f"Failed to rank category <{category_name}>: {e}")
        return None


async def _update_rank_by_gpt_and_post_process(
    gpt: GPT,
    hypothesis: str,
    all_stocks: List[StockID],
    summary_str: str,
    category_idx: int,
    categories: List[Category],
    new_topics_group: TopicGroup,
    orig_topic_group: TopicGroup,
    old_ranking: RankingList,
    gbi_id_to_description: Dict[int, str],
    context: PlanRunContext,
) -> Optional[RankingList]:
    logger = get_prefect_logger(__name__)

    category_name = categories[category_idx].name

    try:
        ####################
        # Prepare GPT inputs
        ####################
        logger.info(f"Preparing GPT inputs to rerank category <{category_name}>")

        # new_topics
        new_topics_str = new_topics_group.convert_to_gpt_input()

        # old topics

        input_topic_set = set(orig_topic_group.val)

        # Get citation source texts from old ranking to use as references for new one
        old_texts = []
        for citation in old_ranking.history[0].citations:
            text_citation = cast(TextCitation, citation)  # type: ignore
            old_text = text_citation.source_text
            old_text.reset_id()  # we need this to match these across runs
            if old_text in input_topic_set:
                old_texts.append(old_text)

        old_topics_group = TopicGroup(
            val=old_texts, offset=new_topics_group.total, id_to_str=orig_topic_group.id_to_str
        )
        old_topics_str = old_topics_group.convert_to_gpt_input()

        # company descriptions
        involved_stocks = list(
            set(new_topics_group.get_all_stocks()) & set(old_topics_group.get_all_stocks())
        )
        company_description_str = "\n".join(
            COMPANY_DESCRIPTION_TEMPLATE.format(
                company_name=stock.company_name,
                symbol=stock.symbol,
                description=gbi_id_to_description[stock.gbi_id],
            )
            for stock in involved_stocks
        )

        # categories
        category_str = await categories[category_idx].to_gpt_input()
        other_categories = [categories[i] for i in range(len(categories)) if i != category_idx]
        other_categories_str = Category.multi_to_gpt_input(other_categories)

        ####################
        # Call GPT
        ####################
        logger.info(f"Calling GPT to rank category <{category_name}>")

        result = await gpt.do_chat_w_sys_prompt(
            main_prompt=UPDATE_RANK_BY_CATEGORY_MAIN_PROMPT.format(
                hypothesis=hypothesis,
                category=category_str,
                other_categories=other_categories_str,
                company_descriptions=company_description_str,
                new_topics=new_topics_str,
                old_topics=old_topics_str,
                old_ranking=old_ranking.val,
            ),
            sys_prompt=UPDATE_RANK_BY_CATEGORY_SYS_PROMPT.format(summary_str=summary_str),
        )

        ##############################
        # Extract citations from text
        #############################
        combined_topic_group: TopicGroup = TextGroup.join(new_topics_group, old_topics_group)  # type: ignore
        logger.info(f"Extracting inline citations for category <{category_name}>")
        ranking_list_w_summary = await _rank_output_postprocess(
            result, category_name, combined_topic_group, all_stocks, context
        )

        await tool_log(
            f"Finished ranking and summarizing for category <{categories[category_idx].name}>",
            context=context,
        )

        return ranking_list_w_summary
    except Exception as e:
        logger.exception(f"Failed to rerank category <{category_name}>: {e}")
        return None


async def _rank_output_postprocess(
    gpt_output: str,
    category_name: str,
    topic_group: TopicGroup,
    all_stocks: List[StockID],
    context: PlanRunContext,
) -> Optional[RankingList]:
    logger = get_prefect_logger(__name__)

    # Extract ranking, summary and citations
    ranking_idx = gpt_output.find(RANKING_HEADER)
    summary_idx = gpt_output.find(SUMMARY_HEADER)

    ranking_section = gpt_output[ranking_idx + len(RANKING_HEADER) : summary_idx].strip()

    citation_idx = gpt_output.rfind("\n")
    citation_section = gpt_output[citation_idx + 1 :].strip()
    summary_section = gpt_output[summary_idx + len(SUMMARY_HEADER) : citation_idx].strip()

    # Use regex to match and create the objects for ranking

    """
    - Advanced Micro Devices, Inc. (AMD) Score 3.5
        - Advanced Micro Devices, Inc. (AMD) has made significant strides...
    ...
    """

    pattern = r"- (.+?) \((.+?)\) Score ([0-9]+(?:\.[0-9]+)?)\n\s{1,4}-(.*?)(?=\n\s{0,4}- |\n$|$)"
    matches = re.findall(pattern, ranking_section, re.DOTALL)
    if not matches:
        logger.warning(f"Unable to match ranking list to the pattern: {ranking_section}")
        return None

    ranking_objs = []
    symbol_to_stock = {stock.symbol: stock for stock in all_stocks}
    for each_match in matches:
        ranking_objs.append(
            RankedCompany(
                company_name=each_match[0],
                symbol=each_match[1],
                score=float(each_match[2]),
                explanation=each_match[3].strip(),
            )
        )

        symbol_to_stock.pop(each_match[1])  # delete existing stocks, and keep only missing ones

    # Reorder the text
    reordered_text = f"{summary_section}\n\n{ranking_section}\n\n{citation_section}"

    text, citations = await extract_citations_from_gpt_output(reordered_text, topic_group, context)  # type: ignore # noqa

    # Add missing stocks in the end of the text
    final_text_list = [text]
    for stock in symbol_to_stock.values():
        default_obj = RankedCompany.create_default_obj(
            stock.symbol, stock.company_name, category_name  # type: ignore
        )
        ranking_objs.append(default_obj)
        final_text_list.append(str(default_obj))  # formatted text

    final_text = "\n".join(final_text_list)

    ranking_list_w_summary = RankingList(
        val=final_text, ranking=ranking_objs, history=[HistoryEntry(citations=citations)]  # type: ignore
    )
    return ranking_list_w_summary


def calculate_weighted_average_scores(
    candidate_target_stock: Optional[StockID],
    stocks: List[StockID],
    categories: List[Category],
    category_idx_to_result: Dict[int, RankingList],
) -> Tuple[StockID, Dict[str, float]]:
    scores_mapping = {}
    for category_idx, result in category_idx_to_result.items():
        weight = categories[category_idx].weight
        for ranking in result.ranking:
            if ranking.symbol not in scores_mapping:
                scores_mapping[ranking.symbol] = [ranking.score * weight, weight]
            else:
                scores_mapping[ranking.symbol][0] += ranking.score * weight
                scores_mapping[ranking.symbol][1] += weight

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
    category_idx_to_result: Dict[int, RankingList],
    final_scores: Dict[str, float],
    context: PlanRunContext,
    gpt_service_stub: Optional[GPTServiceStub] = None,
) -> str:
    # Prompt
    prompt_str = """
    You'll be given a hypothesis and your job is to interpret the hypothesis and answer it \
    based on the information provided.
    The hypothesis is broken down into several categories, each with explanations, justifications, and weights.
    For each category, you will also be provided with a ranking list of the companies and the explanations \
    why they are ranked in that order from the perspective of that category. You will also be provided with \
    a weighted-average score for each company based on the rankings in each category. The score is in range of 5.
    Return a string that consists of 3 to 5 sentences that focuses on the target company to answer the hypothesis.
    Again, the summary should be consistent with the weighted-average scores, but you should never mention the scores, \
    nor repeat the hypothesis in the summary.
    If the target company is not in any of the ranking lists, you should say that there is not enough information \
    to discuss the target company, and then talk about other companies in the ranking lists.
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
        FINAL_SUMMARY_WEIGHTED_SCORE.format(symbol=symbol, score=round(score, 2))
        for symbol, score in sorted(final_scores.items(), key=lambda x: -x[1])
    )

    rankings_list = []
    for category_idx, result in category_idx_to_result.items():
        rankings_list.append(
            FINAL_SUMMARY_CATEGORY_RANKING.format(
                category_name=categories[category_idx].name,
                ranking=str(result),
            )
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

    async def main(prompt: str, product_str: str, main_stock: str) -> None:
        import datetime

        from agent_service.tools.other_text import (
            GetAllTextDataForStocksInput,
            get_default_text_data_for_stocks,
        )
        from agent_service.tools.product_filter import (
            FilterStocksByProductOrServiceInput,
            filter_stocks_by_product_or_service,
        )
        from agent_service.tools.stocks import (
            GetStockUniverseInput,
            StockIdentifierLookupInput,
            get_stock_universe,
            stock_identifier_lookup,
        )

        context = PlanRunContext.get_dummy(user_id="6953b640-16f9-4757-914e-02de6b79fab4")

        # Get stock ID and peers' IDs
        stock_id = await stock_identifier_lookup(
            StockIdentifierLookupInput(stock_name=main_stock), context
        )

        # Get stocks from universe
        stocks: List[StockID] = await get_stock_universe(GetStockUniverseInput(universe_name="S&P 500"), context)  # type: ignore # noqa
        if stock_id not in stocks:
            stocks.append(stock_id)  # type: ignore

        filtered_stocks: List[StockID] = await filter_stocks_by_product_or_service(  # type: ignore
            FilterStocksByProductOrServiceInput(
                stock_ids=stocks, texts=[], product_str=product_str, must_include_stocks=[stock_id]  # type: ignore
            ),
            context,
        )

        all_texts: List[StockText] = await get_default_text_data_for_stocks(  # type: ignore
            GetAllTextDataForStocksInput(
                stock_ids=filtered_stocks, start_date=datetime.date(2024, 6, 1)  # type: ignore # noqa
            ),
            context,
        )

        categories = await get_criteria_for_competitive_analysis(
            CriteriaForCompetitiveAnalysis(market=product_str), context
        )

        output: CompetitiveAnalysis = await do_competitive_analysis(
            DoCompetitiveAnalysisInput(
                prompt=prompt,
                criteria=categories,  # type: ignore
                stocks=filtered_stocks,
                all_text_data=all_texts,
                target_stock=stock_id,  # type: ignore
            ),
            context,
        )
        summary = await generate_summary_for_competitive_analysis(
            GenerateSummaryForCompetitiveAnalysisInput(prompt=prompt, competitive_analysis=output),
            context,
        )
        print(summary)

    import asyncio

    from agent_service.utils.logs import init_stdout_logging

    init_stdout_logging()
    asyncio.run(
        main(
            prompt="Is AMD the leader in AI chips space?", product_str="AI chips", main_stock="AMD"
        )
    )
