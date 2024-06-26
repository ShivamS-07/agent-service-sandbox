import asyncio
import json
from typing import List

import pandas as pd

from agent_service.GPT.constants import GPT4_O
from agent_service.GPT.requests import GPT
from agent_service.io_type_utils import TableColumnType
from agent_service.io_types.stock import StockID
from agent_service.io_types.table import Table, TableColumnMetadata
from agent_service.tool import ToolArgs, ToolCategory, ToolRegistry, tool
from agent_service.types import PlanRunContext
from agent_service.utils.prompt_utils import Prompt
from agent_service.utils.string_utils import repair_json_if_needed


class ProductCompareInput(ToolArgs):
    stock_ids: List[StockID]
    category: str
    main_stock: str


# Potentially can have a tool which returns an intermediate data object like ProductSpecs instead of just a table
class ProductSpecs:
    stock_id: StockID
    product_name: str
    specs: str


GET_PRODUCT_COMPARE_MAIN_PROMPT_STR = (
    "What are the most impactful latest {product} products from each of the following "
    "companies {companies}? The {main_stock} product should be first. Look at the stock "
    "symbol and/or ISIN to identify the stock. "
    "Return a list where each entry is "
    "a pythonic dictionary representing the latest {product} from each of the "
    "companies, leave the company out if they are not related to the product. "
    "The dictionary should have the following entries. "
    "One entry for the company that the stock belongs to whose key is `stock_name`, "
    "one entry for the product name whose key is `product_name`, "
    "one entry for the product release date whose key is `release_date` and "
    "the remaining entries for each of the specifications, the key being the "
    "specification and the entry being a description. "
    "For any specifications, make sure similar units are being compared "
    "against each other. Be sure to keep specifications which are "
    "relevant to {product}. Use double quotes"
    "The specification values should not be comparative "
    "to previous products. If the specification key is missing any details, "
    "fill in the details in the value field. "
    "You can place the units after the numbers in each "
    "returned result. Be sure to include any detailed specifications "
    "important to {product}. No other details, justification or formatting such as "
    "``` or python should be returned. I'm sure there are newer chips, so try again. "
    "Finally, make SURE that all specification keys are the same in all dicts in the list."
)


GET_PRODUCT_COMPARE_MAIN_PROMPT = Prompt(
    name="GET_PRODUCT_COMPARE_MAIN_PROMPT", template=GET_PRODUCT_COMPARE_MAIN_PROMPT_STR
)

GET_PRODUCT_COMPARE_SYS_PROMPT_STR = (
    "You are a financial analyst highly skilled at searching company products and making comparisons. You "
    "are to return the latest products with specifications from the selected "
    "companies for comparison purposes. Your result should make it easy for all "
    "users to compare the different products by making the keys consistent. It is VERY important that the products "
    "are the latest so your clients get accurate and up to date info"
)

GET_PRODUCT_COMPARE_SYS_PROMPT = Prompt(
    name="GET_PRODUCT_COMPARE_SYS_PROMPT", template=GET_PRODUCT_COMPARE_SYS_PROMPT_STR
)


@tool(
    description="Given a list of StockID objects and a product name/category,"
    "return back a table of comparisons between said product between the listed companies."
    "This resultant table should ALWAYS be rendered as a table, never create a graph out of it",
    category=ToolCategory.KPI,
    tool_registry=ToolRegistry,
)
async def get_product_comparison_table(args: ProductCompareInput, context: PlanRunContext) -> Table:
    llm = GPT(context=None, model=GPT4_O)
    result = await get_llm_product_comparisons(args.stock_ids, args.category, args.main_stock, llm)

    data_list = json.loads(repair_json_if_needed(result))
    columns: List[TableColumnMetadata] = []

    if len(data_list) > 0:
        column_titles = data_list[0].keys()
        for title in column_titles:
            columns.append(TableColumnMetadata(label=title, col_type=TableColumnType.STRING))

    df = pd.DataFrame(data_list)
    return Table.from_df_and_cols(data=df, columns=columns)


async def get_llm_product_comparisons(
    stock_ids: List[StockID], category: str, main_stock: str, llm: GPT
) -> str:
    formatted_stock_ids = [
        (
            "Symbol: " + ("unavailable" if (stock_id.symbol is None) else stock_id.symbol),
            "ISIN: " + stock_id.isin,
        )
        for stock_id in stock_ids
    ]

    result = await llm.do_chat_w_sys_prompt(
        main_prompt=GET_PRODUCT_COMPARE_MAIN_PROMPT.format(
            product=category,
            companies=str(formatted_stock_ids),
            main_stock=main_stock,
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

    table_result: Table = await get_product_comparison_table(  # type: ignore
        ProductCompareInput(stock_ids=stock_ids, category="AI chips", main_stock="NVDA"),
        plan_context,
    )
    print(table_result)

    df = table_result.to_df()
    print(df.head())
    for column in table_result.columns:
        print(column.metadata.label)


if __name__ == "__main__":
    asyncio.run(main())
