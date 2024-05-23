import datetime
from collections import defaultdict
from typing import Dict, List, Optional

from agent_service.external.grpc_utils import timestamp_to_datetime
from agent_service.external.nlp_svc_client import get_multi_companies_news_topics
from agent_service.io_types.text import (
    StockNewsDevelopmentArticlesText,
    StockNewsDevelopmentText,
)
from agent_service.tool import ToolArgs, ToolCategory, ToolRegistry, tool
from agent_service.types import PlanRunContext
from agent_service.utils.date_utils import get_now_utc
from agent_service.utils.postgres import get_psql


class GetNewsDevelopmentsAboutCompaniesInput(ToolArgs):
    stock_ids: List[int]
    start_date: Optional[datetime.date] = None
    end_date: Optional[datetime.date] = None


@tool(
    description=(
        "This function calls an internal API which provides all the news developments "
        "with articles between the start date and the end date that are relevant to the"
        " provided list of stocks,the output is a list of list of news development identifiers, "
        "each internal list corresponds to the news for one input stock. If end_date is left out, "
        "the current date is used. If start_date is left out, 1 week ago is used. The"
        "length of the returned list of lists is the same as stock_ids."
    ),
    category=ToolCategory.NEWS,
    tool_registry=ToolRegistry,
)
async def get_news_developments_about_companies(
    args: GetNewsDevelopmentsAboutCompaniesInput, context: PlanRunContext
) -> List[List[StockNewsDevelopmentText]]:
    response = await get_multi_companies_news_topics(
        user_id=context.user_id, gbi_ids=args.stock_ids
    )
    # Response now has a list of topics. Build an association dict to ensure correct ordering.
    stock_to_topics_map: Dict[int, List] = defaultdict(list)
    for topic in response.topics:
        stock_to_topics_map[topic.gbi_id].append(topic)

    start_date = args.start_date
    end_date = args.end_date
    if not start_date:
        start_date = (get_now_utc() - datetime.timedelta(days=7)).date()
    if not end_date:
        # Add an extra day to be sure we don't miss anything with timezone weirdness
        end_date = get_now_utc().date() + datetime.timedelta(days=1)
    outputs: List[List[StockNewsDevelopmentText]] = []
    for gbi_id in args.stock_ids:
        topics = stock_to_topics_map[gbi_id]
        topic_list = []
        for topic in topics:
            topic_date = timestamp_to_datetime(topic.last_article_date).date()
            if topic_date < start_date or topic_date > end_date:
                # Filter topics not in the time window
                continue
            # Only return ID's
            topic_list.append(StockNewsDevelopmentText(id=topic.topic_id.id))
        outputs.append(topic_list)

    return outputs


class GetNewsArticlesForStockDevelopmentsInput(ToolArgs):
    developments_list: List[StockNewsDevelopmentText]


@tool(
    description=(
        "This function takes a list of news developments and returns a list of all the news"
        " development articles for those news developments."
    ),
    category=ToolCategory.NEWS,
    tool_registry=ToolRegistry,
)
async def get_news_articles_for_stock_developments(
    args: GetNewsArticlesForStockDevelopmentsInput, context: PlanRunContext
) -> List[StockNewsDevelopmentArticlesText]:
    sql = """
        SELECT news_id::VARCHAR
        FROM nlp_service.stock_news
        WHERE topic_id = ANY(%(topic_ids)s)
    """
    rows = get_psql().generic_read(
        sql, {"topic_ids": [topic.id for topic in args.developments_list]}
    )
    return [StockNewsDevelopmentArticlesText(id=row["news_id"]) for row in rows]