import datetime
from collections import defaultdict
from typing import Dict, List, Optional

from agent_service.external.grpc_utils import timestamp_to_datetime
from agent_service.external.nlp_svc_client import get_multi_companies_news_topics
from agent_service.io_types.text import (
    StockAlignedTextGroups,
    StockNewsDevelopmentArticlesText,
    StockNewsDevelopmentText,
    TextGroup,
)
from agent_service.tool import ToolArgs, ToolCategory, ToolRegistry, tool
from agent_service.types import PlanRunContext
from agent_service.utils.date_utils import get_now_utc
from agent_service.utils.postgres import get_psql


async def _get_news_developments_helper(
    gbi_ids: List[int],
    user_id: str,
    start_date: Optional[datetime.date] = None,
    end_date: Optional[datetime.date] = None,
) -> Dict[int, List[StockNewsDevelopmentText]]:
    response = await get_multi_companies_news_topics(user_id=user_id, gbi_ids=gbi_ids)
    # Response now has a list of topics. Build an association dict to ensure correct ordering.
    stock_to_topics_map: Dict[int, List] = defaultdict(list)
    for topic in response.topics:
        stock_to_topics_map[topic.gbi_id].append(topic)

    if not start_date:
        start_date = (get_now_utc() - datetime.timedelta(days=7)).date()
    if not end_date:
        # Add an extra day to be sure we don't miss anything with timezone weirdness
        end_date = get_now_utc().date() + datetime.timedelta(days=1)

    output_dict: Dict[int, List[StockNewsDevelopmentText]] = {}
    for gbi_id in gbi_ids:
        topics = stock_to_topics_map[gbi_id]
        topic_list = []
        for topic in topics:
            topic_date = timestamp_to_datetime(topic.last_article_date).date()
            if topic_date < start_date or topic_date > end_date:
                # Filter topics not in the time window
                continue
            # Only return ID's
            topic_list.append(StockNewsDevelopmentText(id=topic.topic_id.id))
        output_dict[gbi_id] = topic_list

    return output_dict


class GetNewsDevelopmentsAboutCompaniesInput(ToolArgs):
    stock_ids: List[int]
    start_date: Optional[datetime.date] = None
    end_date: Optional[datetime.date] = None


@tool(
    description=(
        "This function calls an internal API which provides all the news developments "
        "with articles between the start date and the end date that are relevant to the"
        " provided list of stocks, the output is a list of news developments. "
        "Unlike get_stock_aligned_news_developments, all developments are returned in a single list"
        "there is no segregation by company, so it is appropriately used when you are filtering "
        "and or summarizing all news about one or more stocks into a single summary. "
        "This function is not appropriate for use in filtering of the input stocks, "
        "or other applications where you need to do per stock analysis or per stock generation of "
        "text since all the news is included together and it is not possible to do anything else "
        "at an individual stock level"
        "An example of the kind of query you would use this function for: "
        " `Summarize all the news about GPUs for Nvida and Intel over the last 3 weeks. "
        "If you want to filter news articles by topic, you should choose this function. "
        "If end_date is left out, "
        "the current date is used. If start_date is left out, 1 week ago is used."
    ),
    category=ToolCategory.NEWS,
    tool_registry=ToolRegistry,
)
async def get_all_news_developments_about_companies(
    args: GetNewsDevelopmentsAboutCompaniesInput, context: PlanRunContext
) -> List[StockNewsDevelopmentText]:
    topic_lookup = await _get_news_developments_helper(
        args.stock_ids, context.user_id, args.start_date, args.end_date
    )
    output: List[StockNewsDevelopmentText] = []
    for topic_list in topic_lookup.values():
        output.extend(topic_list)
    return output


@tool(
    description=(
        "This function calls an internal API which provides all the news developments "
        "with articles between the start date and the end date, arranged "
        " according to stock, the output is a list of StockAlignedTextGroups with a "
        "mapping from stocks to lists of news articles associated with the stock "
        "This function should be used when you plan to pass this data to an LLM-based aligned function"
        " to filter stocks. Use get_all_news_developments_about_companies if you simply want"
        " to summarize all the news. An example of the kind of query you would use this for: "
        "`I want a list of airline stocks that have faced major customer service issues in the last month. "
        "Again, if you want to filter stocks by topic, you should choose this function."
        "If end_date is left out, the current date is used. If start_date is left out, 1 week ago is used"
    ),
    category=ToolCategory.NEWS,
    tool_registry=ToolRegistry,
)
async def get_stock_aligned_news_developments(
    args: GetNewsDevelopmentsAboutCompaniesInput, context: PlanRunContext
) -> StockAlignedTextGroups:
    topic_lookup = await _get_news_developments_helper(
        args.stock_ids, context.user_id, args.start_date, args.end_date
    )
    output: Dict[int, TextGroup] = {}
    for stock_id, topic_list in topic_lookup.items():
        output[stock_id] = TextGroup(val=topic_list)  # type: ignore
    return StockAlignedTextGroups(val=output)


class GetNewsArticlesForStockDevelopmentsInput(ToolArgs):
    developments_list: List[StockNewsDevelopmentText]


@tool(
    description=(
        "This function takes a list of news developments and returns a list of all the news"
        " development articles for those news developments. "
        "This function should be used if a client specifically mentions that they "
        "want to see individual news articles, rather than summarized developments. "
        "Do not convert the developments to articles unless it is very clear that the "
        "clients wants that level of detail."
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
