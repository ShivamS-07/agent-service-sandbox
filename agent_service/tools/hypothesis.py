# this file has test_** functions that are not actually tests for pytest purposes

import json
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from gpt_service_proto_v1.service_grpc import GPTServiceStub

from agent_service.GPT.constants import GPT4_O, NO_PROMPT
from agent_service.GPT.requests import GPT, _get_gpt_service_stub
from agent_service.io_type_utils import HistoryEntry, Score
from agent_service.io_types.stock import StockID
from agent_service.io_types.text import (
    CustomDocumentSummaryText,
    StockEarningsSummaryText,
    StockEarningsText,
    StockHypothesisCustomDocumentText,
    StockHypothesisEarningsSummaryPointText,
    StockHypothesisNewsDevelopmentText,
    StockNewsDevelopmentText,
    Text,
    TextCitation,
    TextGroup,
)
from agent_service.planner.errors import EmptyInputError
from agent_service.tool import ToolArgs, ToolCategory, ToolRegistry, tool
from agent_service.tools.tool_log import tool_log
from agent_service.types import PlanRunContext
from agent_service.utils.async_utils import gather_with_concurrency
from agent_service.utils.gpt_logging import GptJobIdType, GptJobType, create_gpt_context
from agent_service.utils.hypothesis.hypothesis_pipeline import HypothesisPipeline
from agent_service.utils.hypothesis.types import (
    CompanyNewsTopicInfo,
    HypothesisNewsTopicInfo,
)
from agent_service.utils.prefect import get_prefect_logger
from agent_service.utils.prompt_utils import Prompt
from agent_service.utils.string_utils import clean_to_json_if_needed

logger = get_prefect_logger(__name__)


async def hypothesis_helper(
    hypothesis: str,
    stock_to_developments: Dict[StockID, List[StockNewsDevelopmentText]],
    context: PlanRunContext,
    gpt_service_stub: Optional[GPTServiceStub] = None,
) -> List[Tuple[CompanyNewsTopicInfo, HypothesisNewsTopicInfo]]:
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

    news_topic_pairs: List[Tuple[CompanyNewsTopicInfo, HypothesisNewsTopicInfo]] = []

    async def each_task(stock_id: StockID) -> None:
        news_developments = stock_to_developments[stock_id]
        pipeline = stock_id_to_pipeline[stock_id]

        hypothesis_news_topics, news_topics = await pipeline.get_stock_hypothesis_news_topics(
            topic_ids=[development.id for development in news_developments]
        )

        # safer to pair them here
        news_topic_pairs.extend(zip(news_topics, hypothesis_news_topics))

        await tool_log(
            log=f"Found {len(news_topics)} relevant news topics out of {len(news_developments)} for {stock_id.symbol}",
            context=context,
        )

    tasks = [each_task(stock_id) for stock_id in stock_to_developments.keys()]
    await gather_with_concurrency(tasks=tasks, n=len(tasks))

    return news_topic_pairs


class TestNewsHypothesisInput(ToolArgs):
    """
    `news_development_list` is a list of news topics (in nlp svc context) that are potentially
    relevant to different stocks. So in the tool we need to segerate them by stocks and use a for
    loop to process each stock
    """

    hypothesis: str
    news_development_list: List[StockNewsDevelopmentText]


@tool(
    description="Given a string of a user's hypothesis, and a list of news developments,"
    " test the hypothesis based on the input news development, and keep the relevant ones with the"
    " explanation and score that explain why the development is relevant to the hypothesis, and how"
    " much it supports or contradicts the hypothesis."
    " You should ONLY use this tool when the user is making a hypothesis and trying to find proof"
    " among news developments."
    " For example, if a user asks `Is NVDA the leader in AI chips space?`, you should convert"
    " it to a statement like `NVDA is the leader in AI chips space` and test it against the news.",
    category=ToolCategory.COMPETITIVE_ANALYSIS,
    tool_registry=ToolRegistry,
    enabled=False,
)
async def test_hypothesis_for_news_developments(
    args: TestNewsHypothesisInput, context: PlanRunContext
) -> List[StockHypothesisNewsDevelopmentText]:
    id_to_development = {development.id: development for development in args.news_development_list}

    # segerate news topics by stocks
    stock_to_developments: Dict[StockID, List[StockNewsDevelopmentText]] = defaultdict(list)
    for development in args.news_development_list:
        if development.stock_id is None:
            continue

        stock_to_developments[development.stock_id].append(development)

    logger.info(f"Processing hypothesis {args.hypothesis} for {len(stock_to_developments)} stocks")
    news_topic_pairs = await hypothesis_helper(
        args.hypothesis, stock_to_developments, context, gpt_service_stub=_get_gpt_service_stub()[0]
    )

    hypothesis_news_developments: List[StockHypothesisNewsDevelopmentText] = []
    for news_topic, hypothesis_news_topic in news_topic_pairs:
        development = id_to_development[news_topic.topic_id]

        support_score: float = hypothesis_news_topic.get_latest_support()  # type: ignore
        scaled_score = Score.scale_input(val=support_score, lb=-1, ub=1)
        reason: str = hypothesis_news_topic.get_latest_reason()  # type: ignore
        hypothesis_news_developments.append(
            StockHypothesisNewsDevelopmentText(
                id=news_topic.topic_id,
                support_score=scaled_score,
                reason=reason,
                history=[HistoryEntry(explanation=reason, score=scaled_score)],
                stock_id=development.stock_id,
            )
        )

    return hypothesis_news_developments


class SummarizeNewsHypothesisInput(ToolArgs):
    """
    Summarize hypothesis with the filtered news developments which are the
    outputs from tool `test_hypothesis_for_news_developments`
    """

    hypothesis: str
    news_developments: List[StockHypothesisNewsDevelopmentText] = []


@tool(
    description="Given a string of a user's hypothesis, and a list of relevant news developments,"
    " calculate the match score of how much this hypothesis is matched with these news topics and"
    " also generate a summary to explain."
    " This tool MUST be used when tool `test_hypothesis_for_news_developments` is used.",
    category=ToolCategory.COMPETITIVE_ANALYSIS,
    tool_registry=ToolRegistry,
    enabled=False,
)
async def summarize_hypothesis_from_news_developments(
    args: SummarizeNewsHypothesisInput, context: PlanRunContext
) -> Text:
    if not args.news_developments:
        raise EmptyInputError("Could not find any relevant news developments")

    # gbi_id doesn't matter, as long as it's valid
    pipeline = HypothesisPipeline(
        gbi_id=714, hypothesis_text=args.hypothesis, gpt_service_stub=_get_gpt_service_stub()[0]
    )
    await pipeline.llm.get_hypothesis_breakdown()  # a bit wasteful as we done before but it's cheap

    match_score, summary, ref_news_developments, _, _ = (
        await pipeline.calculate_match_score_and_generate_summary(
            args.news_developments,
            earnings_summary_points=[],
            custom_documents=[],
        )
    )

    score = Score.scale_input(match_score, lb=-1, ub=1)
    citations = [TextCitation(source_text=topic) for topic in ref_news_developments]
    return Text(val=summary, history=[HistoryEntry(score=score, citations=citations)])  # type: ignore  # noqa


class TestAndSummarizeNewsHypothesisInput(ToolArgs):
    hypothesis: str
    news_developments: List[StockNewsDevelopmentText]


@tool(
    description=(
        "Given a list of one or more relevant news developments, this function generates a score"
        "indicating the extent to which the provided hypothesis is supported by the "
        "news developments, and a short summary which explains the score with reference to "
        "the information in the news developments."
        "This hypothesis must be focused on a specific stock or a small group of stocks, this function "
        "must NOT be used to filter stocks more generally! (i.e. Do not use it for "
        " `Give/find me stocks...` type queries, use the filter by profile tool)"
    ),
    category=ToolCategory.COMPETITIVE_ANALYSIS,
    tool_registry=ToolRegistry,
    enabled=False,
)
async def test_and_summarize_hypothesis_with_news_developments(
    args: TestAndSummarizeNewsHypothesisInput, context: PlanRunContext
) -> Text:
    logger.info("Testing hypothesis for news...")
    hypothesis_news_developments: List[StockHypothesisNewsDevelopmentText] = (
        await test_hypothesis_for_news_developments(  # type: ignore
            TestNewsHypothesisInput(
                hypothesis=args.hypothesis, news_development_list=args.news_developments
            ),
            context,
        )
    )

    await tool_log(
        log=f"Identified {len(hypothesis_news_developments)} relevant news topics in total.",
        context=context,
    )

    logger.info("Summarizing news hypothesis...")
    text: Text = await summarize_hypothesis_from_news_developments(  # type: ignore
        SummarizeNewsHypothesisInput(
            hypothesis=args.hypothesis, news_developments=hypothesis_news_developments
        ),
        context,
    )

    return text


class TestEarningsHypothesisInput(ToolArgs):
    """
    `earnings_summary_list` is a list of earnings summaries (in nlp svc context) that are
    potentially relevant to different stocks. So in the tool we need to segerate them by stocks
    and use a for-loop to process each stock
    """

    hypothesis: str
    earnings_summary_list: List[StockEarningsText]


@tool(
    description="Given a string of a user's hypothesis, and a list of earnings summaries,"
    " test the hypothesis based on the input earnings summaries, and return a list of earnings"
    " summary remarks or questions bullet points that relevant to the hypothesis, with explanation"
    " and score that explain why the point is relevant to the hypothesis, and how much it supports"
    " or contradicts the hypothesis."
    " You should ONLY use this tool when the user is making a hypothesis and trying to find proof"
    " among earnings summaries."
    " For example, if a user asks `Is NVDA the leader in AI chips space?`, you should convert"
    " it to a statement like `NVDA is the leader in AI chips space` and test it against the earnings.",
    category=ToolCategory.COMPETITIVE_ANALYSIS,
    tool_registry=ToolRegistry,
    enabled=False,
)
async def test_hypothesis_for_earnings_summaries(
    args: TestEarningsHypothesisInput, context: PlanRunContext
) -> List[StockHypothesisEarningsSummaryPointText]:
    logger = get_prefect_logger(__name__)

    # segerate earnings summaries by stocks
    stock_to_summaries: Dict[StockID, List[StockEarningsText]] = defaultdict(list)
    for earnings_summary in args.earnings_summary_list:
        if earnings_summary.stock_id is None:
            continue

        # Only look at earnings with summary for now
        if isinstance(earnings_summary, StockEarningsSummaryText):
            stock_to_summaries[earnings_summary.stock_id].append(earnings_summary)

    logger.info(f"Processing hypothesis {args.hypothesis} for {len(stock_to_summaries)} stocks")

    pipeline: HypothesisPipeline = None  # type: ignore

    outputs: List[StockHypothesisEarningsSummaryPointText] = []
    gpt_service_stub = _get_gpt_service_stub()[0]
    for stock_id, earnings_summary_list in stock_to_summaries.items():
        if pipeline is None:
            pipeline = HypothesisPipeline(
                stock_id.gbi_id, args.hypothesis, gpt_service_stub=gpt_service_stub
            )
            await pipeline.initial_hypothesis_processing()
        else:
            new_hypothesis_obj = pipeline.create_hypothesis_info(stock_id.gbi_id, args.hypothesis)
            new_hypothesis_obj.embedding = pipeline.hypothesis.embedding
            new_hypothesis_obj.hypothesis_breakdown = pipeline.hypothesis.hypothesis_breakdown
            pipeline.hypothesis = new_hypothesis_obj

        hypothesis_earnings_topics, _ = await pipeline.get_stock_hypothesis_earnings_topics(
            summary_ids=[summary.id for summary in earnings_summary_list]  # type: ignore
        )

        for topic in hypothesis_earnings_topics:
            support_score: float = topic.get_latest_support()  # type: ignore
            scaled_score = Score.scale_input(val=support_score, lb=-1, ub=1)
            reason: str = topic.get_latest_reason()  # type: ignore
            outputs.append(
                StockHypothesisEarningsSummaryPointText(
                    id=hash((topic.topic_id, topic.summary_index, topic.summary_type.value)),
                    stock_id=stock_id,
                    summary_id=topic.topic_id,
                    summary_idx=topic.summary_index,
                    summary_type=topic.summary_type.value,
                    history=[HistoryEntry(explanation=reason, score=scaled_score)],
                    support_score=scaled_score,
                    reason=reason,
                )
            )

    return outputs


class SummarizeEarningsHypothesisInput(ToolArgs):
    """
    Summarize hypothesis with the filtered earnings summaries which are the
    outputs from tool `test_hypothesis_for_earnings_summaries`
    """

    hypothesis: str
    earnings_summary_points: List[StockHypothesisEarningsSummaryPointText] = []


@tool(
    description="Given a string of a user's hypothesis, and a list of relevant earnings summary"
    " points, calculate the match score of how much this hypothesis is matched with these earnings"
    " summary points and also generate a summary to explain."
    " This tool MUST be used when tool `test_hypothesis_for_earnings_summaries` is used.",
    category=ToolCategory.COMPETITIVE_ANALYSIS,
    tool_registry=ToolRegistry,
    enabled=False,
)
async def summarize_hypothesis_from_earnings_summaries(
    args: SummarizeEarningsHypothesisInput, context: PlanRunContext
) -> Text:
    if not args.earnings_summary_points:
        raise ValueError("Could not find any relevant earnings summary points")

    # gbi_id doesn't matter, as long as it's valid
    pipeline = HypothesisPipeline(
        gbi_id=714, hypothesis_text=args.hypothesis, gpt_service_stub=_get_gpt_service_stub()[0]
    )
    await pipeline.llm.get_hypothesis_breakdown()  # a bit wasteful as we done before but it's cheap

    match_score, summary, _, ref_earnings_points, _ = (
        await pipeline.calculate_match_score_and_generate_summary(
            news_developments=[],
            earnings_summary_points=args.earnings_summary_points,
            custom_documents=[],
        )
    )

    score = Score.scale_input(match_score, lb=-1, ub=1)
    citations = [TextCitation(source_text=topic) for topic in ref_earnings_points]
    return Text(val=summary, history=[HistoryEntry(score=score, citations=citations)])  # type: ignore  # noqa


EARNINGS_SUMMARY_MAIN_PROMPT_STR = (
    "You will read some relevant earnings call summaries from one or more companies and decide to "
    "what degree the following hypothesis is supported by the provided evidence. "
    "If the hypothesis is relevant to a specific company, and the documents are from several related companies, "
    "you should make sure that all documents from all companies are taken into account, and compare "
    "these companies as comprehensively as you can to get the final conclusion. "
    "You should output a json mapping with the three fields. "
    "The first json field should have the key `support_score` and the value should be an integer of range of 0-10. "
    "0 means that the provided evidence indicates with perfect certainty the 100% sure the hypothesis is false. "
    "5 means that the provided evidence does not provide conclusive evidence either way. "
    "10 means that the provided evidence indicates with perfect certainty the 100% sure the hypothesis is true. "
    "If, based on the evidence provided, you believe the hypothesis is not true, you must provide a score under 5."
    "You should be skeptical of the hypothesis, if there isn't strong evidence, you should prefer a score close to 5. "
    "It is very bad to say anything in the 8-10 range without very conclusive evidence. "
    "The second field should have the key `summary` and the value should be a string consisting of a paragragh of "
    "2 to 4 sentences. In this summary, you will discuss the evidence for the validity of the hypothesis. "
    "It should be compatible with your score, but do not explicitly mention the score."
    "You will justify your answer with explicit reference to facts in the earnings summaries"
    "The third field should have the key `citations` and correspond to a list of integers, where the integers "
    "refer to the text numbers of the provided texts. You should only cite those earnings summaries which you "
    "specifically pull facts from. If you are provided "
    "with only one earnings call summary, your output for citation should be [0]."
    "If you are provided multiple earnings calls, try to cite at least two."
    "Here is the hypothesis you are evaulating : {hypothesis}"
    "Here are the earnings summaries: {earnings}"
)

EARNINGS_SUMMARY_MAIN_PROMPT = Prompt(
    EARNINGS_SUMMARY_MAIN_PROMPT_STR, "EARNINGS_SUMMARY_MAIN_PROMPT"
)

EARNINGS_SUMMARY_SYS_PROMPT_STR = (
    "You are a financial analyst who will read some relevant earnings summaries from one "
    "or more companies and decide to what degree the following hypothesis is supported by the provided evidence. "
)
EARNINGS_SUMMARY_SYS_PROMPT = Prompt(EARNINGS_SUMMARY_SYS_PROMPT_STR, "EARNINGS_SUMMARY_SYS_PROMPT")


async def get_summary_and_score_for_earnings(
    hypothesis: str, earnings_summaries: List[StockEarningsText], agent_id: str
) -> Tuple[Score, str, List[TextCitation]]:

    earnings_text_group = TextGroup(val=earnings_summaries)  # type: ignore
    texts_str: str = await Text.get_all_strs(
        earnings_text_group, include_header=True, text_group_numbering=True  # type: ignore
    )

    gpt_context = create_gpt_context(GptJobType.AGENT_PLANNER, agent_id, GptJobIdType.AGENT_ID)
    gpt = GPT(model=GPT4_O, context=gpt_context)

    result = await gpt.do_chat_w_sys_prompt(
        EARNINGS_SUMMARY_MAIN_PROMPT.format(hypothesis=hypothesis, earnings=texts_str),
        EARNINGS_SUMMARY_SYS_PROMPT.format(),
    )
    result_json = json.loads(clean_to_json_if_needed(result))

    score = Score.scale_input(result_json["support_score"], lb=0, ub=10)
    summary: str = result_json["summary"]
    citations: List[TextCitation] = earnings_text_group.get_citations(result_json["citations"])  # type: ignore  # noqa

    return score, summary, citations


class TestAndSummarizeEarningsHypothesisInput(ToolArgs):

    hypothesis: str
    earnings_summaries: List[StockEarningsText]


@tool(
    description=(
        "Given a list of one or more relevant earnings summaries, this function generates a score"
        "indicating the extent to which the provided hypothesis is supported by the "
        "earnings calls, and a short summary which explains the score with reference to "
        "the information in the earnings call summaries"
        "This hypothesis must be focused on a specific stock or a small group of stocks, this function "
        "must NOT be used to filter stocks more generally! (i.e. Do not use it for "
        " `Give/find me stocks...` type queries, use the filter by profile tool)"
    ),
    category=ToolCategory.COMPETITIVE_ANALYSIS,
    tool_registry=ToolRegistry,
    enabled=False,
)
async def test_and_summarize_hypothesis_with_earnings_summaries(
    args: TestAndSummarizeEarningsHypothesisInput, context: PlanRunContext
) -> Text:
    earnings_summaries: List[StockEarningsText] = [
        earning_text
        for earning_text in args.earnings_summaries
        if isinstance(earning_text, StockEarningsSummaryText)
    ]
    if not earnings_summaries:
        raise ValueError("Could not find any relevant earnings summary points")

    support_score, summary, citations = await get_summary_and_score_for_earnings(
        args.hypothesis, earnings_summaries, agent_id=context.agent_id
    )

    return Text(val=summary, history=[HistoryEntry(score=support_score, citations=citations)])  # type: ignore  # noqa


class TestCustomDocsHypothesisInput(ToolArgs):
    """
    `custom_document_list` is a list of user uploaded custom document summaries
    that are potentially relevant to different stocks. So in the tool we need to segregate
    them by stocks and use a for loop to process each stock
    """

    hypothesis: str
    custom_document_list: List[CustomDocumentSummaryText]


@tool(
    description="Given a string of a user's hypothesis, and a list of custom documents,"
    " test the hypothesis based on the input custom documents, and keep the relevant ones with the"
    " explanation and score that explain why the custom document is relevant to the hypothesis, and how"
    " much it supports or contradicts the hypothesis."
    " You should ONLY use this tool when the user is making a hypothesis and trying to find proof"
    " among their uploaded custom documents."
    " For example, if a user asks `Is NVDA the leader in AI chips space?`, you should convert"
    " it to a statement like `NVDA is the leader in AI chips space` and test it against the"
    " documents that the user has uploaded.",
    category=ToolCategory.COMPETITIVE_ANALYSIS,
    tool_registry=ToolRegistry,
    enabled=False,
)
async def test_hypothesis_for_custom_documents(
    args: TestCustomDocsHypothesisInput, context: PlanRunContext
) -> List[StockHypothesisCustomDocumentText]:
    logger = get_prefect_logger(__name__)

    news_id_to_custom_doc = {
        custom_document.id: custom_document for custom_document in args.custom_document_list
    }

    # segerate news topics by stocks
    stock_to_custom_docs: Dict[StockID, List[CustomDocumentSummaryText]] = defaultdict(list)
    for custom_document in args.custom_document_list:
        if custom_document.stock_id is None:
            continue

        stock_to_custom_docs[custom_document.stock_id].append(custom_document)

    logger.info(f"Processing hypothesis {args.hypothesis} for {len(stock_to_custom_docs)} stocks")

    pipeline: HypothesisPipeline = None  # type: ignore

    hypothesis_custom_documents: List[StockHypothesisCustomDocumentText] = []
    gpt_service_stub = _get_gpt_service_stub()[0]
    for stock_id, custom_document_list in stock_to_custom_docs.items():
        if pipeline is None:
            pipeline = HypothesisPipeline(
                stock_id.gbi_id, args.hypothesis, gpt_service_stub=gpt_service_stub
            )
            await pipeline.initial_hypothesis_processing()  # only need to do it once
        else:
            new_hypothesis_obj = pipeline.create_hypothesis_info(stock_id.gbi_id, args.hypothesis)
            new_hypothesis_obj.embedding = pipeline.hypothesis.embedding
            new_hypothesis_obj.hypothesis_breakdown = pipeline.hypothesis.hypothesis_breakdown
            pipeline.hypothesis = new_hypothesis_obj

        hypothesis_news_topics, news_topics, topic_id_to_news_id = (
            await pipeline.get_stock_hypothesis_custom_document_topics(
                custom_document_news_ids=[
                    custom_document.id for custom_document in custom_document_list
                ]
            )
        )

        await tool_log(
            log=f"Found {len(news_topics)} relevant news topics for {stock_id.symbol}",
            context=context,
        )

        for news_topic, hypothesis_news_topic in zip(news_topics, hypothesis_news_topics):
            custom_document = news_id_to_custom_doc[topic_id_to_news_id[news_topic.topic_id]]

            # TODO: move the scaling logic into the class
            support_score: float = hypothesis_news_topic.get_latest_support()  # type: ignore
            scaled_score = Score.scale_input(val=support_score, lb=-1, ub=1)
            reason: str = hypothesis_news_topic.get_latest_reason()  # type: ignore
            hypothesis_custom_documents.append(
                StockHypothesisCustomDocumentText(
                    # here the ID is the custom doc ID == news ID
                    id=topic_id_to_news_id[news_topic.topic_id],
                    topic_id=news_topic.topic_id,
                    support_score=scaled_score,
                    reason=reason,
                    history=[HistoryEntry(explanation=reason, score=scaled_score)],
                    stock_id=custom_document.stock_id,
                    requesting_user=context.user_id,
                )
            )

    return hypothesis_custom_documents


class SummarizeCustomDocumentHypothesisInput(ToolArgs):
    """
    Summarize hypothesis with the filtered custom document news which are the
    outputs from tool `test_hypothesis_for_custom_documents`
    """

    hypothesis: str
    custom_documents: List[StockHypothesisCustomDocumentText] = []


@tool(
    description="Given a string of a user's hypothesis, and a list of relevant uploaded custom document"
    " news, calculate the match score of how much this hypothesis is matched with these"
    " news topics and also generate a summary to explain."
    " This tool MUST be used when tool `test_hypothesis_for_custom_documents` is used.",
    category=ToolCategory.COMPETITIVE_ANALYSIS,
    tool_registry=ToolRegistry,
    enabled=False,
)
async def summarize_hypothesis_from_custom_documents(
    args: SummarizeCustomDocumentHypothesisInput, context: PlanRunContext
) -> Text:
    if not args.custom_documents:
        raise ValueError("Could not find any relevant custom documents")

    # gbi_id doesn't matter, as long as it's valid
    pipeline = HypothesisPipeline(
        gbi_id=714, hypothesis_text=args.hypothesis, gpt_service_stub=_get_gpt_service_stub()[0]
    )
    await pipeline.llm.get_hypothesis_breakdown()  # a bit wasteful as we done before but it's cheap

    match_score, summary, _, _, ref_custom_documents = (
        await pipeline.calculate_match_score_and_generate_summary(
            news_developments=[],
            earnings_summary_points=[],
            custom_documents=args.custom_documents,
        )
    )

    score = Score.scale_input(match_score, lb=-1, ub=1)
    citations = [TextCitation(source_text=topic) for topic in ref_custom_documents]
    return Text(val=summary, history=[HistoryEntry(score=score, citations=citations)])  # type: ignore  # noqa


class TestAndSummarizeCustomDocsHypothesisInput(ToolArgs):
    hypothesis: str
    custom_documents: List[CustomDocumentSummaryText]


@tool(
    description=(
        "Given a list of one or more relevant custom documents, this function generates a score"
        "indicating the extent to which the provided hypothesis is supported by the "
        "custom documents, and a short summary which explains the score with reference to "
        "the information in the custom documents."
    ),
    category=ToolCategory.COMPETITIVE_ANALYSIS,
    tool_registry=ToolRegistry,
    enabled=False,
)
async def test_and_summarize_hypothesis_with_custom_documents(
    args: TestAndSummarizeCustomDocsHypothesisInput, context: PlanRunContext
) -> Text:
    logger.info("Testing hypothesis for custom docs...")
    hypothesis_custom_docs: List[StockHypothesisCustomDocumentText] = await test_hypothesis_for_custom_documents(  # type: ignore # noqa
        TestCustomDocsHypothesisInput(
            hypothesis=args.hypothesis, custom_document_list=args.custom_documents
        ),
        context,
    )

    await tool_log(
        log=f"Identified {len(hypothesis_custom_docs)} relevant custom document news topics in total.",
        context=context,
    )

    logger.info("Summarizing custom docs hypothesis...")
    text: Text = await summarize_hypothesis_from_custom_documents(  # type: ignore
        SummarizeCustomDocumentHypothesisInput(
            hypothesis=args.hypothesis, custom_documents=hypothesis_custom_docs
        ),
        context,
    )

    return text


class SummarizeHypothesisFromVariousSourcesInput(ToolArgs):
    hypothesis: str
    hypothesis_summaries: List[Text]


@tool(
    description="Given a string of a user's hypothesis, a list of summary texts for the hypothesis"
    " derived from different sources (news, earnings, custom documents, etc.), combine them into a"
    " single summary that explains the extent to which the hypothesis is supported overall."
    " This tool is normally used when the user is making a hypothesis and trying to find proof among"
    " various sources. You MUST use it when more than one of the following tools are used:"
    " `test_and_summarize_hypothesis_with_news_developments`,"
    " `test_and_summarize_hypothesis_with_earnings_summaries` and"
    " `test_and_summarize_hypothesis_with_custom_documents`.",
    category=ToolCategory.COMPETITIVE_ANALYSIS,
    tool_registry=ToolRegistry,
    enabled=False,
)
async def summarize_hypothesis_from_various_sources(
    args: SummarizeHypothesisFromVariousSourcesInput, context: PlanRunContext
) -> Text:
    if not args.hypothesis_summaries:
        raise ValueError("None valid hypothesis summaries provided!")

    avg_score = sum([s.history[0].score.val for s in args.hypothesis_summaries]) / len(args.hypothesis_summaries)  # type: ignore  # noqa

    citations = []
    for summary in args.hypothesis_summaries:
        citations.extend(summary.history[0].citations)

    summary = await _combine_summaries(
        agent_id=context.agent_id, summaries=[s.val for s in args.hypothesis_summaries]
    )

    return Text(val=summary, history=[HistoryEntry(score=Score(val=avg_score), citations=citations)])  # type: ignore  # noqa


async def _combine_summaries(agent_id: str, summaries: List[str]) -> str:
    main_prompt = (
        "You should read the following summaries and combine them into a single summary that captures "
        " the essence of all the summaries and make sure to preserve all the evidence from all the summaries. "
        "I will say it again, DO NOT miss any points or evidence.\n{summaries_str}"
    )
    summaries_str = "\n".join([f"Summary {i + 1}: {s}" for i, s in enumerate(summaries)])

    gpt_context = create_gpt_context(GptJobType.AGENT_PLANNER, agent_id, GptJobIdType.AGENT_ID)
    gpt = GPT(model=GPT4_O, context=gpt_context)

    return await gpt.do_chat_w_sys_prompt(
        Prompt(template=main_prompt, name="COMBINE_SUMMARIES_PROMPT").format(
            summaries_str=summaries_str
        ),
        NO_PROMPT,
    )
