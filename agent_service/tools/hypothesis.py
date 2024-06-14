from collections import defaultdict
from typing import Dict, List

from agent_service.io_type_utils import HistoryEntry, Score
from agent_service.io_types.stock import StockID
from agent_service.io_types.text import StockNewsDevelopmentText
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
    " test the hypothesis based on the input news development, and keep the ones that are relevant."
    " In the output, each news development will have a `history` field with the explanation and"
    " score that explain why the development is relevant to the hypothesis, and how much it supports"
    " or contradicts the hypothesis. You should ONLY use this tool when the user is making a"
    " hypothesis and trying to find proof among news developments."
    " For example, if a user asks `Is NVDA the leader in AI chips space?`, you should convert"
    " it to a statement like `NVDA is the leader in AI chips space` and test it against the news.",
    category=ToolCategory.HYPOTHESIS,
    tool_registry=ToolRegistry,
    enabled=False,  # FIXME: Set to True when it's ready
)
async def test_hypothesis_for_news_developments(
    args: TestNewsHypothesisInput, context: PlanRunContext
) -> List[StockNewsDevelopmentText]:
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

    filtered_news_developments: List[StockNewsDevelopmentText] = []
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
            development.history.append(
                HistoryEntry(
                    explanation=hypothesis_news_topic.get_latest_reason(),
                    score=Score.scale_input(val=support_score, lb=-1, ub=1),
                )
            )
            filtered_news_developments.append(development)

    return filtered_news_developments
