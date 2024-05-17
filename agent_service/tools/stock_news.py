import datetime
from collections import defaultdict
from typing import Dict, List, Optional

from agent_service.external.grpc_utils import timestamp_to_datetime
from agent_service.external.nlp_svc_client import get_multi_companies_news_topics
from agent_service.io_types import NewsDevelopmentText
from agent_service.tool import ToolArgs, ToolCategory, ToolRegistry, tool
from agent_service.types import PlanRunContext
from agent_service.utils.date_utils import get_now_utc


class GetNewsDevelopmentsAboutCompaniesInput(ToolArgs):
    stock_ids: List[int]
    start_date: Optional[datetime.date] = None
    end_date: Optional[datetime.date] = None


@tool(
    description=(
        "This function calls an internal API which provides all the news developments "
        "with articles between the start date and the end date that are relevant to the"
        " provided list of stocks,the output is a list of list of news development identifiers, "
        "each internal list corresponds to an input company. If end_date is left out, "
        "the current date is used. If start_date is left out, 1 week ago is used."
    ),
    category=ToolCategory.NEWS,
    tool_registry=ToolRegistry,
)
async def get_news_developments_about_companies(
    args: GetNewsDevelopmentsAboutCompaniesInput, context: PlanRunContext
) -> List[List[NewsDevelopmentText]]:
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
    outputs: List[List[NewsDevelopmentText]] = []
    for gbi_id in args.stock_ids:
        topics = stock_to_topics_map[gbi_id]
        topic_list = []
        for topic in topics:
            topic_date = timestamp_to_datetime(topic.last_article_date).date()
            if topic_date < start_date or topic_date > end_date:
                # Filter topics not in the time window
                continue
            # Only return ID's
            topic_list.append(NewsDevelopmentText(id=topic.topic_id.id))
        outputs.append(topic_list)

    return outputs
