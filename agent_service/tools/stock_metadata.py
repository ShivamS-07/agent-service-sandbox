from typing import List

import pandas as pd

from agent_service.io_type_utils import TableColumnType
from agent_service.io_types.stock import StockID
from agent_service.io_types.table import StockTable, TableColumnMetadata
from agent_service.io_types.text import StockDescriptionText, StockText
from agent_service.tool import ToolArgs, ToolCategory, default_tool_registry, tool
from agent_service.tools.tool_log import tool_log
from agent_service.tools.web_search.general_websearch import (
    GeneralStockWebSearchInput,
    general_stock_web_search,
)
from agent_service.types import PlanRunContext
from agent_service.utils.async_db import get_async_db
from agent_service.utils.prefect import get_prefect_logger


class GetStockDescriptionInput(ToolArgs):
    stock_ids: List[StockID]


@tool(
    description=(
        "Given a list of stock ID's, return a list of descriptions for the stocks. "
        "A stock description generally contains basic, general information about the company's "
        "operations, including major products, services, and holdings, the regions they operate in, etc. "
        "You must never, ever call prepare_output on company descriptions directly, you must always pass"
        "them to other tools for processing, by default if the client asks for a description of a company "
        "pass the output of this tool on the summarize_texts tool, which will provide proper "
        "summarization and formatting."
    ),
    category=ToolCategory.TEXT_RETRIEVAL,
    tool_registry=default_tool_registry(),
)
async def get_company_descriptions(
    args: GetStockDescriptionInput, context: PlanRunContext
) -> List[StockText]:
    logger = get_prefect_logger(__name__)
    db = get_async_db()
    gbi_ids = {stock.gbi_id: stock for stock in args.stock_ids}

    # If the user doesn't want web results, we should fill any missing
    # descriptions with placeholder text since we won't be able to do web
    # search.
    desc_map = await db.get_company_descriptions(
        gbi_ids=list(gbi_ids.keys()),
        use_placeholder_text=not context.user_settings.include_web_results,
    )
    missing_desc_gbi_ids = set(gbi_ids.keys()) - set(desc_map.keys())
    results = [
        StockDescriptionText(id=stock_id.gbi_id, stock_id=stock_id)
        for stock_id in args.stock_ids
        if stock_id.gbi_id not in missing_desc_gbi_ids
    ]
    if missing_desc_gbi_ids and context.user_settings.include_web_results:
        # If the user is ok with web results, and we have some stocks with
        # missing descriptions, search the web for some info. Just do a quick
        # scrape.
        logger.info(
            f"Falling back to web search, no company descriptions for: {missing_desc_gbi_ids}"
        )
        web_results = await general_stock_web_search(
            args=GeneralStockWebSearchInput(
                stock_ids=[
                    gbi_ids[gbi] for gbi in missing_desc_gbi_ids if gbi_ids[gbi].company_name
                ],
                topic="Company Description",
                num_news_urls=0,
                num_google_urls=2,
            ),
            context=context,
        )
        results.extend(web_results)  # type: ignore

    await tool_log(log=f"Found {len(results)} company descriptions", context=context)
    return results  # type: ignore


class GetCompanyNamesInput(ToolArgs):
    stock_ids: List[StockID]


@tool(
    description=(
        "This tool returns a table of company names given a list of StockIDs. "
        "Each company name is stored in string format. This tool is useful if a user wants "
        "to include 'company name' as a field when constructing a table, given a list of StockIDs."
    ),
    category=ToolCategory.STOCK,
    tool_registry=default_tool_registry(),
)
async def get_company_names(args: GetCompanyNamesInput, context: PlanRunContext) -> StockTable:
    await tool_log(log=f"Found {len(args.stock_ids)} company names", context=context)
    df = pd.DataFrame({"ids": args.stock_ids, "names": [x.company_name for x in args.stock_ids]})
    return StockTable.from_df_and_cols(
        data=df,
        columns=[
            TableColumnMetadata(label="ids", col_type=TableColumnType.STOCK),
            TableColumnMetadata(label="names", col_type=TableColumnType.STRING),
        ],
    )
