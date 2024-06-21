import datetime
from typing import List, Optional

from agent_service.external.custom_data_svc_client import (
    get_custom_docs_by_security,
    get_custom_docs_by_topic,
)
from agent_service.io_types.dates import DateRange
from agent_service.io_types.stock import StockID
from agent_service.io_types.text import CustomDocumentSummaryText
from agent_service.tool import ToolArgs, ToolCategory, ToolRegistry, tool
from agent_service.types import PlanRunContext


class GetCustomDocsInput(ToolArgs):
    stock_ids: List[StockID]
    limit: Optional[int] = None
    start_date: Optional[datetime.date] = None
    end_date: Optional[datetime.date] = None
    date_range: Optional[DateRange] = None


@tool(
    description=(
        "This function takes a list of stock IDs and returns a list of N most recent"
        " custom document summaries associated with the stocks."
        " If someone wants summarized information"
        " specific to custom documents or uploaded documents they have uploaded about a stock,"
        " this is the best tool to use."
        " The limit N can be specified using the"
        " `limit` parameter and represents the concept of the N documents most relevant"
        " to the topic."
        " This function must not be used if you intend to filter by stocks, the news articles do not"
        " contain information about which stocks they are relevant to."
    ),
    category=ToolCategory.TEXT,
    tool_registry=ToolRegistry,
)
async def get_user_custom_documents(
    args: GetCustomDocsInput, context: PlanRunContext
) -> List[CustomDocumentSummaryText]:
    # if a date range obj was provided, fill in any missing dates
    if args.date_range:
        if not args.start_date:
            args.start_date = args.date_range.start_date
        if not args.end_date:
            args.end_date = args.date_range.end_date

    custom_doc_summaries = await get_custom_docs_by_security(
        context.user_id,
        gbi_ids=[s.gbi_id for s in args.stock_ids],
        publish_date_start=args.start_date,
        publish_date_end=args.end_date,
    )

    output: List[CustomDocumentSummaryText] = [
        CustomDocumentSummaryText(requesting_user=context.user_id, id=document.article_id)
        for document in custom_doc_summaries.documents
    ]
    if len(output) == 0:
        raise Exception(
            "No user uploaded documents found for these stocks over the specified time period"
        )
    return output


class GetCustomDocsByTopicInput(ToolArgs):
    topic: str
    limit: Optional[int] = None
    start_date: Optional[datetime.date] = None
    end_date: Optional[datetime.date] = None
    date_range: Optional[DateRange] = None


@tool(
    description=(
        "This function takes a topic and returns a list of top N custom document"
        " summaries associated with the topic. The limit N can be specified using the"
        " `limit` parameter and represents the concept of the N documents most relevant"
        " to the topic."
        " If someone wants summarized information specific to custom"
        " documents or uploaded documents they have uploaded about a topic, or related"
        " to a phrase or keyword, this is the best tool to use."
    ),
    category=ToolCategory.TEXT,
    tool_registry=ToolRegistry,
)
async def get_user_custom_documents_by_topic(
    args: GetCustomDocsByTopicInput, context: PlanRunContext
) -> List[CustomDocumentSummaryText]:
    # if a date range obj was provided, fill in any missing dates
    if args.date_range:
        if not args.start_date:
            args.start_date = args.date_range.start_date
        if not args.end_date:
            args.end_date = args.date_range.end_date

    custom_doc_summaries = await get_custom_docs_by_topic(
        context.user_id,
        topic=args.topic,
        limit=args.limit,
        publish_date_start=args.start_date,
        publish_date_end=args.end_date,
    )

    output: List[CustomDocumentSummaryText] = [
        CustomDocumentSummaryText(requesting_user=context.user_id, id=document.summary.article_id)
        for document in custom_doc_summaries.documents
    ]
    if len(output) == 0:
        raise Exception(
            "No user uploaded documents found for these stocks over the specified time period"
        )
    return output
