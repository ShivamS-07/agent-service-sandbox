from collections import defaultdict
from typing import Dict, List

from agent_service.io_type_utils import HistoryEntry, Score
from agent_service.io_types.stock import StockID
from agent_service.io_types.text import (
    CustomDocumentSummaryText,
    StockEarningsSummaryText,
    StockHypothesisCustomDocumentText,
    StockHypothesisEarningsSummaryPointText,
    StockHypothesisNewsDevelopmentText,
    StockNewsDevelopmentText,
    Text,
    TextCitation,
)
from agent_service.tool import ToolArgs, ToolCategory, ToolRegistry, tool
from agent_service.types import PlanRunContext
from agent_service.utils.hypothesis.hypothesis_pipeline import HypothesisPipeline
from agent_service.utils.prefect import get_prefect_logger


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
    category=ToolCategory.HYPOTHESIS,
    tool_registry=ToolRegistry,
)
async def test_hypothesis_for_news_developments(
    args: TestNewsHypothesisInput, context: PlanRunContext
) -> List[StockHypothesisNewsDevelopmentText]:
    logger = get_prefect_logger(__name__)

    id_to_development = {development.id: development for development in args.news_development_list}

    # segerate news topics by stocks
    stock_to_developments: Dict[StockID, List[StockNewsDevelopmentText]] = defaultdict(list)
    for development in args.news_development_list:
        if development.stock_id is None:
            continue

        stock_to_developments[development.stock_id].append(development)

    logger.info(f"Processing hypothesis {args.hypothesis} for {len(stock_to_developments)} stocks")

    pipeline: HypothesisPipeline = None  # type: ignore

    hypothesis_news_developments: List[StockHypothesisNewsDevelopmentText] = []
    for stock_id, news_development_list in stock_to_developments.items():
        if pipeline is None:
            pipeline = HypothesisPipeline(stock_id.gbi_id, args.hypothesis)
            await pipeline.initial_hypothesis_processing()  # only need to do it once
        else:
            new_hypothesis_obj = pipeline.create_hypothesis_info(stock_id.gbi_id, args.hypothesis)
            new_hypothesis_obj.embedding = pipeline.hypothesis.embedding
            new_hypothesis_obj.hypothesis_breakdown = pipeline.hypothesis.hypothesis_breakdown
            pipeline.hypothesis = new_hypothesis_obj

        hypothesis_news_topics, news_topics = await pipeline.get_stock_hypothesis_news_topics(
            topic_ids=[development.id for development in news_development_list]
        )
        for news_topic, hypothesis_news_topic in zip(news_topics, hypothesis_news_topics):
            development = id_to_development[news_topic.topic_id]

            # TODO: move the scaling logic into the class
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
    category=ToolCategory.HYPOTHESIS,
    tool_registry=ToolRegistry,
)
async def summarize_hypothesis_from_news_developments(
    args: SummarizeNewsHypothesisInput, context: PlanRunContext
) -> Text:
    if not args.news_developments:
        raise ValueError("Could not find any relevant news developments")

    # gbi_id doesn't matter, as long as it's valid
    pipeline = HypothesisPipeline(gbi_id=714, hypothesis_text=args.hypothesis)
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


class TestEarningsHypothesisInput(ToolArgs):
    """
    `earnings_summary_list` is a list of earnings summaries (in nlp svc context) that are
    potentially relevant to different stocks. So in the tool we need to segerate them by stocks
    and use a for-loop to process each stock
    """

    hypothesis: str
    earnings_summary_list: List[StockEarningsSummaryText]


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
    category=ToolCategory.HYPOTHESIS,
    tool_registry=ToolRegistry,
)
async def test_hypothesis_for_earnings_summaries(
    args: TestEarningsHypothesisInput, context: PlanRunContext
) -> List[StockHypothesisEarningsSummaryPointText]:
    logger = get_prefect_logger(__name__)

    # segerate earnings summaries by stocks
    stock_to_summaries: Dict[StockID, List[StockEarningsSummaryText]] = defaultdict(list)
    for earnings_summary in args.earnings_summary_list:
        if earnings_summary.stock_id is None:
            continue

        stock_to_summaries[earnings_summary.stock_id].append(earnings_summary)

    logger.info(f"Processing hypothesis {args.hypothesis} for {len(stock_to_summaries)} stocks")

    pipeline: HypothesisPipeline = None  # type: ignore

    outputs: List[StockHypothesisEarningsSummaryPointText] = []
    for stock_id, earnings_summary_list in stock_to_summaries.items():
        if pipeline is None:
            pipeline = HypothesisPipeline(stock_id.gbi_id, args.hypothesis)
            await pipeline.initial_hypothesis_processing()
        else:
            new_hypothesis_obj = pipeline.create_hypothesis_info(stock_id.gbi_id, args.hypothesis)
            new_hypothesis_obj.embedding = pipeline.hypothesis.embedding
            new_hypothesis_obj.hypothesis_breakdown = pipeline.hypothesis.hypothesis_breakdown
            pipeline.hypothesis = new_hypothesis_obj

        hypothesis_earnings_topics, _ = await pipeline.get_stock_hypothesis_earnings_topics(
            summary_ids=[summary.id for summary in earnings_summary_list]
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
    category=ToolCategory.HYPOTHESIS,
    tool_registry=ToolRegistry,
)
async def summarize_hypothesis_from_earnings_summaries(
    args: SummarizeEarningsHypothesisInput, context: PlanRunContext
) -> Text:
    if not args.earnings_summary_points:
        raise ValueError("Could not find any relevant earnings summary points")

    # gbi_id doesn't matter, as long as it's valid
    pipeline = HypothesisPipeline(gbi_id=714, hypothesis_text=args.hypothesis)
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
    category=ToolCategory.HYPOTHESIS,
    tool_registry=ToolRegistry,
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
    for stock_id, custom_document_list in stock_to_custom_docs.items():
        if pipeline is None:
            pipeline = HypothesisPipeline(stock_id.gbi_id, args.hypothesis)
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
    category=ToolCategory.HYPOTHESIS,
    tool_registry=ToolRegistry,
)
async def summarize_hypothesis_from_custom_documents(
    args: SummarizeCustomDocumentHypothesisInput, context: PlanRunContext
) -> Text:
    if not args.custom_documents:
        raise ValueError("Could not find any relevant custom documents")

    # gbi_id doesn't matter, as long as it's valid
    pipeline = HypothesisPipeline(gbi_id=714, hypothesis_text=args.hypothesis)
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


class SummarizeHypothesisFromVariousSourcesInput(ToolArgs):
    """
    Summarize hypothesis with the filtered news developments and earnings summaries which are the
    outputs from tool `test_hypothesis_for_news_developments` and `test_hypothesis_for_earnings_summaries`
    """

    hypothesis: str
    news_developments: List[StockHypothesisNewsDevelopmentText] = []
    earnings_summary_points: List[StockHypothesisEarningsSummaryPointText] = []
    custom_documents: List[StockHypothesisCustomDocumentText] = []


@tool(
    description="Given a string of a user's hypothesis, a list of relevant news developments,"
    " a list of relevant earnings summary points, and a list of relevant custom document"
    " news developments, calculate the match score of how much this"
    " hypothesis is matched with these topics and also generate a summary to explain."
    " This tool is normally used when the user is making a hypothesis and trying to find proof among"
    " various sources, including news developments and earnings summaries. You should use it when"
    " more than one of the following tools: `summarize_hypothesis_from_news_developments`,"
    " `summarize_hypothesis_from_earnings_summaries` and `summarize_hypothesis_from_custom_documents`"
    " are used.",
    category=ToolCategory.HYPOTHESIS,
    tool_registry=ToolRegistry,
)
async def summarize_hypothesis_from_various_sources(
    args: SummarizeHypothesisFromVariousSourcesInput, context: PlanRunContext
) -> Text:
    if not any((args.news_developments, args.earnings_summary_points, args.custom_documents)):
        raise ValueError("Could not find any relevant news developments or earnings summaries")

    # gbi_id doesn't matter, as long as it's valid
    pipeline = HypothesisPipeline(gbi_id=714, hypothesis_text=args.hypothesis)
    await pipeline.llm.get_hypothesis_breakdown()  # a bit wasteful as we done before but it's cheap

    match_score, summary, ref_news_developments, ref_earnings_points, ref_custom_documents = (
        await pipeline.calculate_match_score_and_generate_summary(
            args.news_developments,
            args.earnings_summary_points,
            args.custom_documents,
        )
    )

    score = Score.scale_input(match_score, lb=-1, ub=1)
    citations = [
        TextCitation(source_text=topic)
        for topic in (ref_news_developments + ref_earnings_points + ref_custom_documents)
    ]
    return Text(val=summary, history=[HistoryEntry(score=score, citations=citations)])  # type: ignore  # noqa
