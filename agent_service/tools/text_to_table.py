import asyncio
import datetime
import logging
import re
from io import StringIO
from typing import Any, List, Optional, Tuple

import pandas as pd

from agent_service.GPT.requests import GPT
from agent_service.GPT.tokens import GPTTokenizer
from agent_service.io_type_utils import PrimitiveType, TableColumnType
from agent_service.io_types.stock import StockID
from agent_service.io_types.table import StockTable, Table, TableColumnMetadata
from agent_service.io_types.text import Text, TextCitation, TextGroup
from agent_service.planner.errors import EmptyInputError
from agent_service.tool import ToolArgs, ToolCategory, tool
from agent_service.tools.LLM_analysis.constants import DEFAULT_LLM
from agent_service.tools.LLM_analysis.utils import (
    extract_citations_from_gpt_output,
    get_initial_breakdown,
    initial_filter_texts,
)
from agent_service.tools.stocks import (
    StockIdentifierLookupInput,
    stock_identifier_lookup_helper,
)
from agent_service.tools.table_utils.prompts import (
    TEXT_TO_TABLE_MAIN_PROMPT,
    TEXT_TO_TABLE_SYS_PROMPT,
)
from agent_service.types import PlanRunContext
from agent_service.utils.async_utils import gather_with_concurrency
from agent_service.utils.feature_flags import get_ld_flag
from agent_service.utils.gpt_logging import GptJobIdType, GptJobType, create_gpt_context
from agent_service.utils.prefect import get_prefect_logger
from agent_service.utils.string_utils import strip_code_backticks
from agent_service.utils.text_utils import partition_to_smaller_text_sizes

logger = logging.getLogger(__name__)

COL_HEADER_REGEX = re.compile(r"\(([^)]+)\)")


class TextToTableArgs(ToolArgs):
    texts: List[Text]
    table_description: str


def _extract_and_clean_header(input_str: str) -> Tuple[str, TableColumnType]:
    # Match the text in parentheses and extract it
    m = re.search(COL_HEADER_REGEX, input_str)
    if m:
        extracted_type = m.group(1)
        # Remove the parentheses and content from the original string
        cleaned_str = re.sub(r"\s*\([^)]*\)", "", input_str)
        if any((extracted_type == val for val in TableColumnType)):
            return cleaned_str, TableColumnType(extracted_type)
        else:
            return cleaned_str, TableColumnType.STRING
    return input_str, TableColumnType.STRING


async def _extract_citation_from_value(
    val: str,
    text_group: TextGroup,
    context: PlanRunContext,
    citation_dict: dict[str, List[dict[str, Any]]],
) -> Tuple[str, List[TextCitation]]:
    val, citations = await extract_citations_from_gpt_output(
        val, text_group=text_group, context=context, premade_anchor_dict=citation_dict
    )
    return val, (citations or [])


async def _lookup_helper_wrapper(
    stock: str, context: PlanRunContext
) -> Tuple[str, Optional[StockID]]:
    # Make sure to strip out citations first
    if "[" in stock:
        stock = stock.split("[")[0].strip()
    try:
        return (
            stock,
            await stock_identifier_lookup_helper(
                args=StockIdentifierLookupInput(stock_name=stock), context=context
            ),
        )
    except Exception:
        logger.exception(f"Failed to resolve stock reference: {stock}")
        return (stock, None)


async def _lookup_stocks(stocks: List[str], context: PlanRunContext) -> dict[str, StockID]:
    tasks = [_lookup_helper_wrapper(stock, context) for stock in stocks]
    results = await gather_with_concurrency(tasks, n=50)
    return {stock: stock_id for stock, stock_id in results if stock_id}


async def _handle_table_col(
    header: str,
    values: pd.Series,
    citation_dict: Optional[dict[str, List[dict[str, Any]]]],
    context: PlanRunContext,
    text_group: TextGroup,
) -> Tuple[TableColumnMetadata, List[Any]]:
    """
    For a single table column, do stock name resolution and citation
    resolution. Returns the newly created table col metadata object as well as
    the modified values.
    """
    col_name, col_type = _extract_and_clean_header(header)
    new_vals = []
    cell_citations: dict[int, List[TextCitation]] = {}
    stock_metadata: dict[str, StockID] = {}
    if col_type == TableColumnType.STOCK:
        stock_metadata = await _lookup_stocks(stocks=values.tolist(), context=context)
        if len(stock_metadata) < len(values):
            # We failed to map one or more stocks, for now just use strings instead
            logger.warning("Failed to match a stock, falling back to a string column")
            col_type = TableColumnType.STRING
            stock_metadata = {}

    for i, val in enumerate(values):
        if citation_dict:
            val, citations = await _extract_citation_from_value(
                val=val, citation_dict=citation_dict, context=context, text_group=text_group
            )
            cell_citations[i] = citations

        if stock_metadata and val in stock_metadata:
            val = stock_metadata[val]

        new_vals.append(val)

    col_meta = TableColumnMetadata(label=col_name, col_type=col_type)
    col_meta.cell_citations = cell_citations  # type: ignore
    return col_meta, new_vals


def enabler_function(user_id: Optional[str]) -> bool:
    return get_ld_flag("enable-text-to-table-tool", default=False, user_context=user_id)


@tool(
    description="""
This function takes a list of texts and a table description and produces a Table
object based on the data in the input texts. Make sure the description is
detailed and contains exactly what kind of table the user wants. The texts will
be fed into an LLM to produce a Table output. You should use this tool if the
user explicitly asks for something to be displayed as a table, or if they ask
for some graphing or transformation that requires a table but you only have text
data. You can also use this tool if you need a list of StockID from the texts,
just pair it with the tool to extract stocks from a table.
""",
    category=ToolCategory.TABLE,
    enabled_checker_func=enabler_function,
)
async def text_to_table(args: TextToTableArgs, context: PlanRunContext) -> Table:
    logger = get_prefect_logger(__name__)
    gpt_context = create_gpt_context(
        GptJobType.AGENT_TOOLS, context.agent_id, GptJobIdType.AGENT_ID
    )
    llm = GPT(context=gpt_context, model=DEFAULT_LLM)

    if len(args.texts) == 0:
        raise EmptyInputError("Cannot create a table from an empty list of text")

    args.texts = await partition_to_smaller_text_sizes(args.texts, context)

    texts = initial_filter_texts(args.texts)
    if len(args.texts) != len(texts):
        logger.warning(f"Too many texts, filtered {len(args.texts)} split texts to {len(texts)}")

    text_group = TextGroup(val=texts)
    texts_str: str = await Text.get_all_strs(  # type: ignore
        text_group,
        include_header=True,
        text_group_numbering=True,
        include_symbols=True,
    )
    if context.chat:
        chat_str = context.chat.get_gpt_input()
    else:
        chat_str = ""

    table_gen_main_prompt = TEXT_TO_TABLE_MAIN_PROMPT
    table_gen_sys_prompt = TEXT_TO_TABLE_SYS_PROMPT

    texts_str = GPTTokenizer(DEFAULT_LLM).do_truncation_if_needed(
        texts_str,
        [
            table_gen_main_prompt.template,
            table_gen_sys_prompt.template,
            chat_str,
            args.table_description,
        ],
    )

    main_prompt = table_gen_main_prompt.format(
        table_description=args.table_description,
        texts=texts_str,
        chat_context=chat_str,
        today=(
            context.as_of_date.date().isoformat()
            if context.as_of_date
            else datetime.date.today().isoformat()
        ),
    )
    sys_prompt = table_gen_sys_prompt.format()

    result = await llm.do_chat_w_sys_prompt(
        main_prompt,
        sys_prompt,
    )

    # Split the text into the actual csv and the citations
    text, citation_dict = get_initial_breakdown(result)

    text = strip_code_backticks(text)

    df = pd.read_csv(StringIO(text))
    metadatas = []
    new_df_data: dict[PrimitiveType, list[Any]] = {}
    tasks = []
    # For each column, resolve the stock names (if a stock column) as well as
    # the citations
    for col, vals in df.items():
        tasks.append(
            _handle_table_col(
                header=str(col),
                values=vals,
                citation_dict=citation_dict,
                context=context,
                text_group=text_group,
            )
        )

    for result in await asyncio.gather(*tasks):
        metadata, new_vals = result
        metadatas.append(metadata)
        new_df_data[metadata.label] = new_vals

    new_df = pd.DataFrame(data=new_df_data)

    table = Table.from_df_and_cols(columns=metadatas, data=new_df)
    if table.get_stock_column():
        return StockTable(
            history=table.history, columns=table.columns, prefer_graph_type=table.prefer_graph_type
        )
    return table


if __name__ == "__main__":
    from agent_service.io_type_utils import load_io_type
    from agent_service.utils.logs import init_stdout_logging
    from agent_service.utils.postgres import get_psql

    db = get_psql()

    sql = """
    SELECT
        output
    FROM agent.task_run_info
    WHERE
        task_id = %(task_id)s
        AND plan_run_id = %(plan_run_id)s
    """

    row = db.generic_read(
        sql,
        {
            "task_id": "73a36d63-bfc1-4436-bbe3-249e1a85c699",
            "plan_run_id": "43a62e12-916f-4c64-9ef4-b5771b5c17bb",
        },
    )[0]
    texts = load_io_type(row["output"])

    init_stdout_logging()
    out = asyncio.run(
        text_to_table(
            args=TextToTableArgs(
                texts=texts,  # type: ignore
                table_description=(
                    "grab all the stocks from Pershing Squares latest "
                    "13f and display it in a table please"
                ),
            ),
            context=PlanRunContext.get_dummy(),
        )
    )
    breakpoint()
    print(out)
