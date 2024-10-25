from collections import defaultdict
from typing import List, Optional

from agent_service.external.custom_data_svc_client import (
    get_custom_docs_by_file_names,
    get_custom_docs_by_security,
    get_custom_docs_by_topic,
)
from agent_service.external.grpc_utils import timestamp_to_datetime
from agent_service.io_types.dates import DateRange
from agent_service.io_types.stock import StockID
from agent_service.io_types.text import CustomDocumentSummaryText
from agent_service.planner.errors import EmptyOutputError
from agent_service.tool import ToolArgs, ToolCategory, ToolRegistry, tool
from agent_service.tools.stocks import get_stock_ids_from_company_ids
from agent_service.tools.tool_log import tool_log
from agent_service.types import PlanRunContext


class GetCustomDocsInput(ToolArgs):
    stock_ids: List[StockID]
    date_range: Optional[DateRange] = None


@tool(
    description=(
        "This function takes a list of stock IDs and returns custom document summaries"
        " associated with the stocks. If someone explicitly wants information"
        " specific to custom documents or uploaded documents they have uploaded about a stock,"
        " this is the best tool to use. Do not use this tool if the user has not"
        " explicitly asked for custom documents or uploaded documents."
        " `date_range` can be optionally specified to filter the documents published"
        " during a certain time period. This should not be used to filter to documents"
        " ABOUT events that occurred during said period as many documents are forwards or"
        " backwards-looking."
        " You should not pass a date_range containing dates after todays date into this function."
        " documents can only be found for dates in the past up to the present, including todays date."
        " I repeat you will be FIRED if you try to find documents from the future!!! YOU MUST NEVER DO THAT!!!"
    ),
    category=ToolCategory.TEXT,
    tool_registry=ToolRegistry,
)
async def get_user_custom_documents(
    args: GetCustomDocsInput, context: PlanRunContext
) -> List[CustomDocumentSummaryText]:
    # if a date range obj was provided, fill in any missing dates
    if args.date_range:
        start_date = args.date_range.start_date
        end_date = args.date_range.end_date
    else:
        start_date = None
        end_date = None

    stock_ids = [s.gbi_id for s in args.stock_ids]
    custom_doc_summaries = await get_custom_docs_by_security(
        context.user_id,
        gbi_ids=stock_ids,
        publish_date_start=start_date,
        publish_date_end=end_date,
        # return all custom docs without limit.
        # NOTE: the limit argument is not provided as an argument as it has lead
        # to very inconsistent behavior. I dont personally see a huge use case for this yet
        limit=None,
    )
    await tool_log(
        log=f"Got {len(custom_doc_summaries.documents)} "
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
            timestamp=timestamp_to_datetime(document.publication_time),
        )
        for document in custom_doc_summaries.documents
    ]
    if len(output) == 0:
        if start_date is not None and end_date is not None:
            start, end = start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")
            await tool_log(
                f"No user uploaded documents found for these stocks between {start=}, {end=}",
                context=context,
            )
        else:
            await tool_log(
                "No user uploaded documents found for these stocks",
                context=context,
            )
    return output


class GetCustomDocsByTopicInput(ToolArgs):
    topic: str
    top_n: Optional[int] = 100
    date_range: Optional[DateRange] = None


@tool(
    description=(
        "This function takes a topic and returns N most relevant"
        " summaries associated with a given non-stock topic."
        " Do not use this tool if the user has not"
        " specifically asked for custom documents or uploaded documents."
        " If someone explicitly wants summarized information specific to custom"
        " documents or uploaded documents they have uploaded about a topic, or related"
        " to a phrase or keyword, this is the best tool to use."
        " Do not use the output of this function for in a stock filter function, the documents"
        " do not contain information about associated stocks."
        " `top_n` can be optionally specified to limit the number of documents returned to"
        " only the N most relevant documents."
        " `date_range` can be specified to filter the documents PUBLISHED"
        " during a certain time period. This should not be used to filter to documents"
        " ABOUT events that occurred during said period as many documents are forwards or"
        " backwards-looking."
        " You should not pass a date_range containing dates after todays date into this function."
        " documents can only be found for dates in the past up to the present, including todays date."
        " I repeat you will be FIRED if you try to find documents from the future!!! YOU MUST NEVER DO THAT!!!"
    ),
    category=ToolCategory.TEXT,
    tool_registry=ToolRegistry,
)
async def get_user_custom_documents_by_topic(
    args: GetCustomDocsByTopicInput, context: PlanRunContext
) -> List[CustomDocumentSummaryText]:
    # if a date range obj was provided, fill in any missing dates
    if args.date_range:
        start_date = args.date_range.start_date
        end_date = args.date_range.end_date
    else:
        start_date = None
        end_date = None

    # arbitrary top_n of 100 topic-vector-similarity docs if no top_n is specified
    # TODO: pass this through a secondary LLM filter to be more confident in result.
    if args.top_n is None or args.top_n == 0:
        args.top_n = 100

    custom_doc_summaries = await get_custom_docs_by_topic(
        context.user_id,
        topic=args.topic,
        publish_date_start=start_date,
        publish_date_end=end_date,
        limit=args.top_n,
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
            timestamp=timestamp_to_datetime(document.summary.publication_time),
        )
        for document in custom_doc_summaries.documents
    ]
    if len(output) == 0:
        raise EmptyOutputError(
            f"No user uploaded documents found for {args.topic=} over the specified time period"
        )
    return output


class GetCustomDocsByFileInput(ToolArgs):
    file_names: List[str]


@tool(
    description=(
        "This function takes in a list of file names and returns all uploaded document"
        " summaries associated with the exact file name. A file name can be any quoted string"
        " that is specifically used in the context of identifying a file that the user"
        " is interested in. File names usually end in an extension like '.pdf' or '.docx'."
        " Make sure to never truncate or otherwise modify the file name - we will only match "
        " exact file names from the user."
        " If someone explicitly specifies a document name in the context of custom"
        " documents or uploaded documents, this is the best tool to use."
        " Do not use this tool if the user has not"
        " specifically asked for custom documents or uploaded documents."
        " Do not use this tool under any circumstances with an empty list of file names."
        " Do not use the output of this function for in a stock filter function, the documents"
        " do not contain information about associated stocks."
    ),
    category=ToolCategory.TEXT,
    tool_registry=ToolRegistry,
)
async def get_user_custom_documents_by_filename(
    args: GetCustomDocsByFileInput, context: PlanRunContext
) -> List[CustomDocumentSummaryText]:
    """
    NOTE: for the initial version, we are looking for something quick and dirty.
    i.e. no integration with the portfolio picker as of now - just give it our best shot
    to parse out the file name as a string and do some basic matching.
    """
    file_names = args.file_names
    if len(file_names) == 0:
        raise ValueError("No file names provided to query by custom doc name tool.")

    custom_doc_summaries = await get_custom_docs_by_file_names(
        context.user_id,
        file_names=file_names,
    )

    # fill worklog by document name
    docs_for_file_name = defaultdict(list)
    for doc in custom_doc_summaries.documents:
        docs_for_file_name[doc.file_name].append(doc)
    for file in file_names:
        num_chunks = len(docs_for_file_name.get(file, []))
        await tool_log(
            log=f'Got {num_chunks} document chunks for file: "{file}"',
            context=context,
        )

    # get stocks and return the output format
    cids = [document.spiq_company_id for document in custom_doc_summaries.documents]
    cid_to_stock = await get_stock_ids_from_company_ids(context, cids)
    output: List[CustomDocumentSummaryText] = [
        CustomDocumentSummaryText(
            requesting_user=context.user_id,
            stock_id=cid_to_stock.get(document.spiq_company_id),
            id=document.article_id,
            timestamp=timestamp_to_datetime(document.publication_time),
        )
        for document in custom_doc_summaries.documents
    ]
    if len(output) == 0:
        raise EmptyOutputError(f"No user uploaded documents found for {file_names}.")
    return output
