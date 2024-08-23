import asyncio
import json
from typing import Any, Dict, List, Text

import pandas as pd

from agent_service.GPT.constants import GPT4_O
from agent_service.GPT.requests import GPT
from agent_service.io_type_utils import TableColumnType
from agent_service.io_types.stock import StockID
from agent_service.io_types.table import Table, TableColumnMetadata
from agent_service.tool import ToolArgs, ToolCategory, ToolRegistry, tool
from agent_service.tools.product_comparison.prompts import (
    GET_PRODUCT_COMPARE_MAIN_PROMPT,
    GET_PRODUCT_COMPARE_SYS_PROMPT,
    GET_PRODUCT_SUMMARY_MAIN_PROMPT,
)
from agent_service.tools.product_comparison.websearch import WebScraper
from agent_service.types import PlanRunContext
from agent_service.utils.postgres import SyncBoostedPG
from agent_service.utils.string_utils import repair_json_if_needed


class ProductCompareInput(ToolArgs):
    stock_ids: List[StockID]
    category: str
    main_stock: StockID


# Potentially can have a tool which returns an intermediate data object like ProductSpecs instead of just a table
class ProductSpecs:
    stock_id: StockID
    product_name: str
    specs: str


@tool(
    description="Given a list of stocks, a product name/category and a main stock, "
    "return back a table of comparisons between said product between the listed companies. "
    "This resultant table should ALWAYS be rendered as a table, never create a graph out of it."
    "Important! This tool is for comparing product specifications. This is different from competitive analysis"
    "This should be called when user wants to compare product specifications between different companies",
    category=ToolCategory.KPI,
    tool_registry=ToolRegistry,
    enabled=False,
)
async def get_product_comparison_table(args: ProductCompareInput, context: PlanRunContext) -> Table:
    llm = GPT(context=None, model=GPT4_O)
    data_list = await get_llm_product_comparisons(
        args.stock_ids, args.category, args.main_stock, llm
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

    # scrape URLs
    scraper = WebScraper(context=context)
    final_table = await scraper.validate_product_info(  # type: ignore
        product_table,
    )

    return final_table


async def get_llm_product_comparisons(
    stock_ids: List[StockID], category: str, main_stock: StockID, llm: GPT
) -> List[Dict[str, Any]]:
    stock_id_str_list = [
        f"gbi_id: {stock_id.gbi_id},"
        f"Symbol: {'unavailable' if (stock_id.symbol is None) else stock_id.symbol}, ISIN: {stock_id.isin}"
        for stock_id in stock_ids
    ]
    stock_id_list = "\n".join(stock_id_str_list)

    result = await llm.do_chat_w_sys_prompt(
        main_prompt=GET_PRODUCT_COMPARE_MAIN_PROMPT.format(
            product=category,
            companies=stock_id_list,
            main_stock=main_stock.symbol,
        ),
        sys_prompt=GET_PRODUCT_COMPARE_SYS_PROMPT.format(),
    )

    result = json.loads(repair_json_if_needed(result))

    for data in result:
        data["stock_id"] = (await StockID.from_gbi_id_list([data["stock_id"]]))[0]

    return result


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
    result = await get_llm_product_summary(args.product_comparison_table, args.category, llm)
    return result


async def get_llm_product_summary(product_comparison_table: Table, category: str, llm: GPT) -> Text:
    table_df = product_comparison_table.to_df()
    table_str_list = [
        ", ".join(f"{col}: {val}" for col, val in zip(table_df.columns, row))
        for row in table_df.values
    ]
    table_contents = "\n".join(table_str_list)

    result = await llm.do_chat_w_sys_prompt(
        main_prompt=GET_PRODUCT_SUMMARY_MAIN_PROMPT.format(
            table_contents=table_contents,
            product=category,
        ),
        sys_prompt=GET_PRODUCT_COMPARE_SYS_PROMPT.format(),
    )

    return result


async def main() -> None:
    plan_context = PlanRunContext.get_dummy()

    aapl_stock_id = StockID(gbi_id=714, symbol="AAPL", isin="")
    amd_stock_id = StockID(gbi_id=124, symbol="AMD", isin="")
    nvda_stock_id = StockID(gbi_id=7555, symbol="NVDA", isin="")
    goog_stock_id = StockID(gbi_id=30336, symbol="GOOG", isin="")
    ma_stock_id = StockID(gbi_id=15857, symbol="MA", isin="")
    intel_stock_id = StockID(gbi_id=5766, symbol="INTL", isin="")

    stock_ids = [
        intel_stock_id,
        aapl_stock_id,
        amd_stock_id,
        goog_stock_id,
        nvda_stock_id,
        ma_stock_id,
    ]

    search_category = "Mobile Phone"
    main_stock = StockID(gbi_id=714, symbol="AAPL", isin="")

    table_result: Table = await get_product_comparison_table(  # type: ignore
        ProductCompareInput(stock_ids=stock_ids, category=search_category, main_stock=main_stock),
        plan_context,
    )

    df = table_result.to_df()
    for i in range(len(df)):
        print(df.iloc[i])

    rich_output = await table_result.to_rich_output(pg=SyncBoostedPG())
    print(rich_output)

    # product comparison summary
    """
    summary_result: Text = await get_product_compare_summary(  # type: ignore
        ProductSummaryInput(table=table_result, category=search_category),
        plan_context,
    )

    print(summary_result)
    """


if __name__ == "__main__":
    asyncio.run(main())
