import asyncio
import json
import logging
from typing import Any, Dict, List

import pandas as pd

from agent_service.GPT.constants import GPT4_O
from agent_service.GPT.requests import GPT
from agent_service.io_type_utils import HistoryEntry, TableColumnType
from agent_service.io_types.stock import StockID
from agent_service.io_types.table import Table, TableColumnMetadata
from agent_service.io_types.text import Text, TextGroup
from agent_service.tool import ToolArgs, ToolCategory, ToolRegistry, tool
from agent_service.tools.LLM_analysis.utils import extract_citations_from_gpt_output
from agent_service.tools.product_comparison.prompts import (
    GET_PRODUCT_COMPARE_SYS_PROMPT,
    GET_PRODUCT_SUMMARY_MAIN_PROMPT,
    WEB_SCRAPE_IMPORTANT_SPECS_MAIN_PROMPT_OBJ,
    WEB_SCRAPE_IMPORTANT_SPECS_SYS_PROMPT_OBJ,
)
from agent_service.tools.product_comparison.websearch import WebScraper
from agent_service.types import PlanRunContext
from agent_service.utils.feature_flags import get_ld_flag, get_user_context
from agent_service.utils.postgres import SyncBoostedPG
from agent_service.utils.string_utils import repair_json_if_needed

logger = logging.getLogger(__name__)


class ProductCompareInput(ToolArgs):
    stock_ids: List[StockID]
    category: str
    main_stock: StockID


# Potentially can have a tool which returns an intermediate data object like ProductSpecs instead of just a table
class ProductSpecs:
    stock_id: StockID
    product_name: str
    specs: str


async def get_important_specs(
    company_names: str, product_name: str, llm: GPT, num_specs: int = 6
) -> List[str]:
    result = await llm.do_chat_w_sys_prompt(
        main_prompt=WEB_SCRAPE_IMPORTANT_SPECS_MAIN_PROMPT_OBJ.format(
            product=product_name,
            companies=company_names,
            num_specs=num_specs,
        ),
        sys_prompt=WEB_SCRAPE_IMPORTANT_SPECS_SYS_PROMPT_OBJ.format(),
    )

    return json.loads(repair_json_if_needed(result))


def enabler_function(user_id: str) -> bool:
    ld_user = get_user_context(user_id)
    result = get_ld_flag("product-comparison-tool", default=False, user_context=ld_user)
    logger.info(f"product comparison tool found?: {result}")
    return result


@tool(
    description="Given a list of stocks, a product name/category and a main stock, "
    "return back a table of comparisons between said product between the listed companies. "
    "This resultant table should ALWAYS be rendered as a table, never create a graph out of it."
    "Important! This tool is for comparing product specifications. This is different from competitive analysis"
    "This should be called when user wants to compare product specifications between different companies",
    category=ToolCategory.KPI,
    tool_registry=ToolRegistry,
    enabled_checker_func=enabler_function,
)
async def get_product_comparison_table(args: ProductCompareInput, context: PlanRunContext) -> Table:
    llm = GPT(context=None, model=GPT4_O)
    scraper = WebScraper(context=context)
    important_specs = await get_important_specs(
        "\n".join([stock.company_name for stock in args.stock_ids]), args.category, llm
    )
    data_list = await get_llm_product_comparisons(
        args.stock_ids, args.category, args.main_stock, important_specs, scraper
    )

    columns: List[TableColumnMetadata] = []
    if len(data_list) > 0:
        column_titles = data_list[0].keys()
        for title in column_titles:
            if title == "stock_id":
                columns.append(
                    TableColumnMetadata(label="stock_id", col_type=TableColumnType.STOCK)
                )

            else:
                columns.append(TableColumnMetadata(label=title, col_type=TableColumnType.STRING))

    df = pd.DataFrame(data_list)

    product_table = Table.from_df_and_cols(data=df, columns=columns)

    # If the product table has no columns, we can't validate it, so we return the empty product table
    if not product_table.columns:
        return product_table

    # scrape URLs for filling in the table of information
    final_table = await scraper.validate_product_info(  # type: ignore
        product_table,
        important_specs,
    )

    return final_table


async def get_llm_product_comparisons(
    stock_ids: List[StockID],
    category: str,
    main_stock: StockID,
    important_specs: List[str],
    scraper: WebScraper,
) -> List[Dict[str, Any]]:
    product_frame = []

    for stock_id in stock_ids:
        product_name = await scraper.get_product_name(
            stock_id.company_name, category, main_stock.company_name
        )
        if product_name:
            for product in product_name:
                # add the first product found within the array, then move onto the next company
                if product:
                    product_frame.append(
                        {
                            "stock_id": stock_id,
                            "product_name": product,
                            "release_date": "n/a",
                        }
                        | {spec: "n/a" for spec in important_specs}
                    )
                    break

    return product_frame


class ProductSummaryInput(ToolArgs):
    product_comparison_table: Table
    category: str


@tool(
    description="DO NOT RUN THIS TOOL DIRECTLY AFTER get_product_comparison_table without product_output in between. "
    "Given a table of different products from different companies, return a text "
    "string describing each product, how they compare, and the company's status in the product's field",
    category=ToolCategory.KPI,
    tool_registry=ToolRegistry,
    enabled=False,
)
async def get_product_compare_summary(args: ProductSummaryInput, context: PlanRunContext) -> Text:
    llm = GPT(context=None, model=GPT4_O)
    result = await get_llm_product_summary(
        args.product_comparison_table, args.category, llm, context
    )
    return result


async def get_llm_product_summary(
    product_comparison_table: Table, category: str, llm: GPT, context: PlanRunContext
) -> Text:
    table_df = product_comparison_table.to_df()
    table_str_list = [
        ", ".join(f"{col}: {val}" for col, val in zip(table_df.columns, row))
        for row in table_df.values
    ]
    table_contents = "\n".join(table_str_list)

    all_text_group = TextGroup(
        val=[
            citation.source_text  # type: ignore
            for history_entry in product_comparison_table.history
            for citation in history_entry.citations
        ]
    )

    result = await llm.do_chat_w_sys_prompt(
        main_prompt=GET_PRODUCT_SUMMARY_MAIN_PROMPT.format(
            table_contents=table_contents, product=category, citations=all_text_group
        ),
        sys_prompt=GET_PRODUCT_COMPARE_SYS_PROMPT.format(),
    )

    summary_text, citations = await extract_citations_from_gpt_output(
        result, all_text_group, context
    )

    summary = Text(val=summary_text or result)
    summary = summary.inject_history_entry(
        HistoryEntry(title="Product Comparison Summary", citations=citations)  # type: ignore
    )

    return summary


async def main() -> None:
    plan_context = PlanRunContext.get_dummy()

    aapl_stock_id = StockID(gbi_id=714, symbol="AAPL", isin="", company_name="Apple")
    # amd_stock_id = StockID(gbi_id=124, symbol="AMD", isin="", company_name="AMD")
    # nvda_stock_id = StockID(gbi_id=7555, symbol="NVDA", isin="", company_name="Nvidia")
    goog_stock_id = StockID(gbi_id=30336, symbol="GOOG", isin="", company_name="Google")
    # ma_stock_id = StockID(gbi_id=15857, symbol="MA", isin="", company_name="Mastercard")
    # intel_stock_id = StockID(gbi_id=5766, symbol="INTL", isin="", company_name="Intel")

    stock_ids = [
        goog_stock_id,
        aapl_stock_id,
    ]

    search_category = "Mobile Phone"
    main_stock = aapl_stock_id

    table_result: Table = await get_product_comparison_table(  # type: ignore
        ProductCompareInput(stock_ids=stock_ids, category=search_category, main_stock=main_stock),
        plan_context,
    )

    df = table_result.to_df()
    for i in range(len(df)):
        print(df.iloc[i])

    rich_output = await table_result.to_rich_output(pg=SyncBoostedPG())
    print(rich_output)

    """
    # product comparison summary
    summary_result: Text = await get_product_compare_summary(  # type: ignore
        ProductSummaryInput(product_comparison_table=table_result, category=search_category),
        plan_context,
    )

    print(summary_result)
    """


if __name__ == "__main__":
    asyncio.run(main())
