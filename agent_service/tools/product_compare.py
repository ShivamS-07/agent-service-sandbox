import asyncio
import json
import re
from typing import Any, Dict, List, Text

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
    "companies? {companies} The {main_stock} product should be first. Look at the stock "
    "symbol and/or ISIN to identify the stock. "
    "Return a list where each entry is "
    "a pythonic dictionary representing the latest {product} from each of the "
    "companies, leave the company out if they are not related to the product. "
    "The dictionary should have the following entries. "
    "One entry for the company that the stock belongs to whose key is `stock_name`, "
    "one entry for the product name whose key is `product_name`, "
    "one entry for the product release date whose key is `release_date` and "
    "the remaining entries for each of the specifications, the key being the "
    "specification and the entry being a description, missing/unavailable/undisclosed values should be n/a. "
    "For any specifications, make sure similar units are being compared "
    "against each other. Be sure to keep specifications which are "
    "relevant to {product}. Use double quotes"
    "The specification values should not be comparative "
    "to previous products. If the specification key is missing any details, "
    "fill in the details in the value field. "
    "You can place the units after the numbers in each "
    "returned result. Be sure to include any detailed specifications "
    "important to {product}. No other details, justification or formatting such as "
    "``` or python should be returned. I'm sure there are newer results, so try again. "
    "Finally, make SURE that all specification keys are the same in all dicts in the list."
)

GET_PRODUCT_SUMMARY_MAIN_PROMPT_STR = (
    "Look at each product within the ones listed below. {table_contents} Give a short "
    "description on each {product} product which compares it to the others as well "
    "as prior products. As well, output a bit regarding the company's trajectory "
    "with {product} products, and its status within the {product} field. "
    "You can give an overall summary afterwards. "
    "I'm only interested in the comparisons for the output, no need to restate "
    "numerical specifications or measurements from the table. "
    "No formatting symbols should be returned, only text."
)

GET_PRODUCT_COMPARE_SYS_PROMPT_STR = (
    "You are a financial analyst highly skilled at searching company products and making comparisons. You "
    "are to return the latest products with specifications from the selected "
    "companies for comparison purposes. Your result should make it easy for all "
    "users to compare the different products by making the keys consistent. It is VERY important that the products "
    "are the latest so your clients get accurate and up to date info."
)

GET_PRODUCT_COMPARE_SYS_PROMPT = Prompt(
    name="GET_PRODUCT_COMPARE_SYS_PROMPT", template=GET_PRODUCT_COMPARE_SYS_PROMPT_STR
)

GET_PRODUCT_COMPARE_MAIN_PROMPT = Prompt(
    name="GET_PRODUCT_COMPARE_MAIN_PROMPT", template=GET_PRODUCT_COMPARE_MAIN_PROMPT_STR
)


def column_treatment(title: str, data_list: List[Dict[str, Any]]) -> TableColumnMetadata:
    """
    This function assumes the following to function:
    - the data values START with an integer/float
    - The data values do not start with . or ,
    - numbers are not directly followed by . or , and instead followed by a space of a letter

    This function will MUTATE the data_list by removing units off of the strings if it stores the unit in the column
    """
    values = [data[title] for data in data_list]
    values_set = set(values)

    # remove the numbers from the front of each value to get the suffix
    value_suffixes_set = set(
        [re.sub(r"^[\d.,]+", "", value).strip() for value in values if value != "n/a"]
    )

    # if the suffix is shared amongst all values, and it is a value which is new (a prefix number was removed)
    if len(value_suffixes_set) == 1 and len(values_set | value_suffixes_set) > len(values_set):
        suffix = list(value_suffixes_set)[0]
        value_prefixes = [
            match.group().strip().replace(",", "") if match else "n/a"
            for value in values
            for match in [re.match(r"^[\d.,]+", value)]
        ]

        is_float = False
        for value_prefix in value_prefixes:
            is_float = is_float or "." in value_prefix

        if is_float:
            for i, data in enumerate(data_list):
                if re.match(r"^\d", value_prefixes[i]):
                    data[title] = float(value_prefixes[i])

            if suffix == "":
                return TableColumnMetadata(label=title, col_type=TableColumnType.FLOAT)
            else:
                return TableColumnMetadata(
                    label=title, unit=suffix, col_type=TableColumnType.FLOAT_WITH_UNIT
                )

        for i, data in enumerate(data_list):
            if re.match(r"^\d", value_prefixes[i]):
                data[title] = int(value_prefixes[i])

        if suffix == "":
            return TableColumnMetadata(label=title, col_type=TableColumnType.INTEGER)
        else:
            return TableColumnMetadata(
                label=title, unit=suffix, col_type=TableColumnType.INTEGER_WITH_UNIT
            )
    else:
        return TableColumnMetadata(label=title, col_type=TableColumnType.STRING)


@tool(
    description="Given a list of stocks, a product name/category and a main stock, "
    "return back a table of comparisons between said product between the listed companies. "
    "This resultant table should ALWAYS be rendered as a table, never create a graph out of it.",
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
            columns.append(column_treatment(title, data_list))

    df = pd.DataFrame(data_list)
    return Table.from_df_and_cols(data=df, columns=columns)


async def get_llm_product_comparisons(
    stock_ids: List[StockID], category: str, main_stock: str, llm: GPT
) -> Text:
    stock_id_str_list = [
        f"Symbol: {'unavailable' if (stock_id.symbol is None) else stock_id.symbol}, ISIN: {stock_id.isin}"
        for stock_id in stock_ids
    ]
    stock_id_list = "\n".join(stock_id_str_list)

    result = await llm.do_chat_w_sys_prompt(
        main_prompt=GET_PRODUCT_COMPARE_MAIN_PROMPT.format(
            product=category,
            companies=stock_id_list,
            main_stock=main_stock,
        ),
        sys_prompt=GET_PRODUCT_COMPARE_SYS_PROMPT.format(),
    )

    return result


GET_PRODUCT_SUMMARY_MAIN_PROMPT = Prompt(
    name="GET_PRODUCT_SUMMARY_MAIN_PROMPT", template=GET_PRODUCT_SUMMARY_MAIN_PROMPT_STR
)


class ProductSummaryInput(ToolArgs):
    table: Table
    category: str


@tool(
    description="DO NOT RUN THIS TOOL DIRECTLY AFTER get_product_comparison_table without product_output in between. "
    "Given a table of different products from different companies, return a text "
    "string describing each product, how they compare, and the company's status in the product's field",
    category=ToolCategory.KPI,
    tool_registry=ToolRegistry,
)
async def get_product_compare_summary(args: ProductSummaryInput, context: PlanRunContext) -> Text:
    llm = GPT(context=None, model=GPT4_O)
    result = await get_llm_product_summary(args.table, args.category, llm)
    return result


async def get_llm_product_summary(table: Table, category: str, llm: GPT) -> Text:
    table_df = table.to_df()
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

    table_result: Table = await get_product_comparison_table(  # type: ignore
        ProductCompareInput(stock_ids=stock_ids, category="AI chip", main_stock="NVDA"),
        plan_context,
    )

    summary_result: Text = await get_product_compare_summary(  # type: ignore
        ProductSummaryInput(table=table_result, category="AI Chips"),
        plan_context,
    )

    df = table_result.to_df()

    print(df.head())
    print(df.iloc[0])
    print(summary_result)


if __name__ == "__main__":
    asyncio.run(main())
