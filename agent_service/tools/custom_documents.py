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
from agent_service.tools.stocks import get_stock_ids_from_company_ids
from agent_service.tools.tool_log import tool_log
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
        " If someone explicitly wants information"
        " specific to custom documents or uploaded documents they have uploaded about a stock,"
        " this is the best tool to use. Do not use this tool if the user has not"
        " explicitly asked for custom documents or uploaded documents."
        " The limit N can be specified using the"
        " `limit` parameter and represents the concept of the most recent N documents"
        " uploaded about a stock."
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

    # arbitrary limit of 50 company documents if no limit is specified
    if args.limit is None or args.limit == 0:
        args.limit = 50

    stock_ids = [s.gbi_id for s in args.stock_ids]
    custom_doc_summaries = await get_custom_docs_by_security(
        context.user_id,
        gbi_ids=stock_ids,
        publish_date_start=args.start_date,
        publish_date_end=args.end_date,
        limit=args.limit,
    )
    await tool_log(
        log=f"Got {len(custom_doc_summaries.documents)}"
        + f"documents for {len(stock_ids)} companies.",
        context=context,
    )
    cids = [document.spiq_company_id for document in custom_doc_summaries.documents]
    cid_to_stock = await get_stock_ids_from_company_ids(
        context, cids, prefer_gbi_ids=[s.gbi_id for s in args.stock_ids]
    )
    output: List[CustomDocumentSummaryText] = [
        CustomDocumentSummaryText(
            requesting_user=context.user_id,
            stock_id=cid_to_stock.get(document.spiq_company_id),
            id=document.article_id,
        )
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
        " summaries associated with the topic. Do not use this tool if the user has not"
        " explicitly asked for custom documents or uploaded documents."
        " The limit N can be specified using the"
        " `limit` parameter and represents the concept of the N documents most relevant"
        " to the topic."
        " If someone explicitly wants summarized information specific to custom"
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

    # arbitrary limit of 10 topic-vector-similarity fdocs if no limit is specified
    # TODO: pass this through a secondary LLM filter to be more confident in result.
    if args.limit is None or args.limit == 0:
        args.limit = 10

    custom_doc_summaries = await get_custom_docs_by_topic(
        context.user_id,
        topic=args.topic,
        publish_date_start=args.start_date,
        publish_date_end=args.end_date,
        limit=args.limit,
    )
    await tool_log(
        log=f"Got {len(custom_doc_summaries.documents)} documents for topic {args.topic}.",
        context=context,
    )

    cids = [document.summary.spiq_company_id for document in custom_doc_summaries.documents]
    cid_to_stock = await get_stock_ids_from_company_ids(context, cids)
    output: List[CustomDocumentSummaryText] = [
        CustomDocumentSummaryText(
            requesting_user=context.user_id,
            stock_id=cid_to_stock.get(document.summary.spiq_company_id),
            id=document.summary.article_id,
        )
        for document in custom_doc_summaries.documents
    ]
    if len(output) == 0:
        raise Exception(
            f"No user uploaded documents found for {args.topic=} over the specified time period"
        )
    return output
