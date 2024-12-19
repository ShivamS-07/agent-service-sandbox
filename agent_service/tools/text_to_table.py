import asyncio
import datetime
import logging
import re
from collections import defaultdict
from copy import deepcopy
from io import StringIO
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from agent_service.GPT.constants import GPT4_O_MINI, NO_PROMPT
from agent_service.GPT.requests import GPT
from agent_service.GPT.tokens import GPTTokenizer
from agent_service.io_type_utils import PrimitiveType, TableColumnType
from agent_service.io_types.stock import StockID
from agent_service.io_types.table import StockTable, Table, TableColumnMetadata
from agent_service.io_types.text import StockText, Text, TextCitation, TextGroup, TextIDType
from agent_service.tool import ToolArgs, ToolCategory, tool
from agent_service.tools.LLM_analysis.constants import DEFAULT_LLM
from agent_service.tools.LLM_analysis.utils import (
    extract_citations_from_gpt_output,
    get_initial_breakdown,
    initial_filter_texts,
)
from agent_service.tools.stocks import (
    StockIdentifierLookupInput,
    stock_identifier_lookup,
)
from agent_service.tools.table_utils.prompts import (
    TEXT_SNIPPET_TABLE_RELEVANCY_PROMPT,
    TEXT_TO_TABLE_MAIN_PROMPT,
    TEXT_TO_TABLE_SYS_PROMPT,
    TEXT_TO_TABLES_INPUT_SCHEMA_PROMPT,
    TEXT_TO_TABLES_NO_INPUT_SCHEMA_PROMPT,
)
from agent_service.tools.tables import JoinTableArgs, join_tables
from agent_service.types import AgentUserSettings, PlanRunContext
from agent_service.utils.async_postgres_base import DEFAULT_ASYNCDB_GATHER_CONCURRENCY
from agent_service.utils.async_utils import gather_with_concurrency
from agent_service.utils.constants import CURRENCY_SYMBOL_TO_ISO, ISO_CURRENCY_CODES
from agent_service.utils.feature_flags import get_ld_flag
from agent_service.utils.gpt_logging import GptJobIdType, GptJobType, create_gpt_context
from agent_service.utils.iterables import chunk
from agent_service.utils.prefect import get_prefect_logger
from agent_service.utils.string_utils import strip_code_backticks
from agent_service.utils.text_utils import partition_to_smaller_text_sizes

logger = logging.getLogger(__name__)

COL_HEADER_REGEX = re.compile(r"\(([^)]+)\)")
NUMBER_EXTRACT_REGEX = re.compile(r"^(.*?)(-?\d+(?:\.\d+)?)(.*)?$")

NO_DATA_STR = "no data"


class TextToTableArgs(ToolArgs):
    texts: List[Text]
    table_description: str
    table_schema: Optional[List[str]] = None


async def classify_stock_text_relevancy_for_table(
    text: str,
    table_description: str,
    llm: GPT,
) -> bool:
    chopped_text_str = GPTTokenizer(model=llm.model).do_truncation_if_needed(
        truncate_str=text,
        other_prompt_strs=[
            TEXT_SNIPPET_TABLE_RELEVANCY_PROMPT.template,
            table_description,
        ],
    )

    output = await llm.do_chat_w_sys_prompt(
        main_prompt=TEXT_SNIPPET_TABLE_RELEVANCY_PROMPT.format(
            table_description=table_description,
            text=chopped_text_str,
        ),
        sys_prompt=NO_PROMPT,
        max_tokens=2,
    )
    return "yes" in output.strip().lower()


async def classify_stock_text_relevancies_for_table(
    texts: List[Text],
    table_description: str,
    gpt_context: Dict[str, str],
    text_cache: Optional[Dict[TextIDType, str]] = None,
) -> List[Text]:
    filtered_texts: List[Text] = []
    text_strs = await Text.get_all_strs(
        texts, include_header=True, include_timestamps=False, text_cache=text_cache
    )

    llm = GPT(
        model=GPT4_O_MINI,
        context=gpt_context,
    )

    tasks = []
    for i, text in enumerate(texts):
        if isinstance(text, StockText):
            text_str = text_strs[i]
            if not text.stock_id:
                continue
            tasks.append(
                classify_stock_text_relevancy_for_table(
                    text=text_str, table_description=table_description, llm=llm
                )
            )

    results = await gather_with_concurrency(tasks, n=200)
    for i, relevancy_decision in enumerate(results):
        if relevancy_decision:
            filtered_texts.append(texts[i])
    return filtered_texts


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


scalar_units = {
    "hundred": 100,
    "thousand": 1000,
    "million": 1000000,
    "billion": 1000000000,
    "trillion": 1000000000000,
}


def extract_number_with_unit_from_text(
    val: str,
    return_int: bool = False,
) -> Tuple[float | int | None, Optional[str]]:
    val = val.strip()
    val = val.replace(",", "")
    m = re.search(NUMBER_EXTRACT_REGEX, val)
    if not m:
        return (None, None)
    prefix = m.group(1)  # e.g. "$"
    num_val = m.group(2)
    suffix = m.group(3)
    scalar = 1

    if prefix:
        prefix = prefix.strip()
    if suffix:
        suffix = suffix.strip()

    if suffix in scalar_units:
        scalar = scalar_units[suffix]
        suffix = None
    if suffix not in ISO_CURRENCY_CODES:
        suffix = None
    if not suffix and prefix in CURRENCY_SYMBOL_TO_ISO:
        suffix = CURRENCY_SYMBOL_TO_ISO[prefix]

    num: float | int | None = None
    try:
        if return_int:
            num = int(num_val) * scalar
        else:
            num = float(num_val) * scalar
    except Exception:
        pass

    return (num, suffix)


async def _lookup_helper_wrapper(
    stock: str, context: PlanRunContext
) -> Tuple[str, Optional[StockID]]:
    # Make sure to strip out citations first
    if "[" in stock:
        stock = stock.split("[")[0].strip()
    try:
        return (
            stock,
            await stock_identifier_lookup(  # type: ignore
                args=StockIdentifierLookupInput(stock_name=stock), context=context
            ),
        )
    except Exception:
        logger.exception(f"Failed to resolve stock reference: {stock}")
        return (stock, None)


async def _lookup_stocks(stocks: List[str], context: PlanRunContext) -> dict[str, StockID]:
    stocks = [stock for stock in stocks if stock.lower() != NO_DATA_STR]
    context = deepcopy(context)
    context.skip_task_logging = True
    tasks = [_lookup_helper_wrapper(stock, context) for stock in stocks]
    results = await gather_with_concurrency(tasks, n=DEFAULT_ASYNCDB_GATHER_CONCURRENCY)
    return {stock: stock_id for stock, stock_id in results if stock_id}


async def _handle_table_col(
    header: str,
    values: pd.Series,
    citation_dict: Optional[dict[str, List[dict[str, Any]]]],
    context: PlanRunContext,
    text_group: TextGroup,
) -> Tuple[TableColumnMetadata, dict[int, Any]]:
    """
    For a single table column, do stock name resolution and citation
    resolution. Returns the newly created table col metadata object as well as
    the modified values. Values are returned as a mapping from row index to
    value, since some values might be filtered out (e.g. stocks).
    """
    col_name, col_type = _extract_and_clean_header(header)
    new_vals: dict[int, Any | None] = {}
    cell_citations: dict[int, List[TextCitation]] = {}
    stock_metadata: dict[str, StockID] = {}
    unit_set = set()
    unmapped_stocks = set()
    if col_type == TableColumnType.STOCK:
        stock_metadata = await _lookup_stocks(stocks=list(set(values)), context=context)
    for i, val in enumerate(values):
        if val is None or pd.isna(val) or val.lower() == NO_DATA_STR:
            new_vals[i] = None
            continue
        if citation_dict:
            val = str(val)
            val, citations = await _extract_citation_from_value(
                val=val, citation_dict=citation_dict, context=context, text_group=text_group
            )
            cell_citations[i] = citations

        if col_type == TableColumnType.STOCK:
            if val not in stock_metadata:
                unmapped_stocks.add(val)
                # Skip stocks we couldn't map
                continue
            else:
                val = stock_metadata[val]

        if (
            isinstance(val, str)
            and col_type.is_float_type()
            or col_type in (TableColumnType.INTEGER, TableColumnType.INTEGER_WITH_UNIT)
        ):
            if not val:
                continue
            num_val, unit = extract_number_with_unit_from_text(
                val=val,
                return_int=not col_type.is_float_type(),  # type: ignore
            )
            if num_val:
                val = num_val
                if unit:
                    unit_set.add(unit)
            else:
                logger.warning("Failed to extract a number, falling back to a string column")
                col_type = TableColumnType.STRING

        new_vals[i] = val

    unit = None
    if len(unit_set) == 1:
        # Only add a unit to the column if all values agree
        unit = unit_set.pop()
    col_meta = TableColumnMetadata(label=col_name, col_type=col_type, unit=unit)
    col_meta.cell_citations = cell_citations  # type: ignore
    if unmapped_stocks:
        logger.error(
            (
                f"For {context.plan_run_id=}, {context.task_id=},"
                f" got unmapped stocks in text to table: {unmapped_stocks}"
            )
        )
    return col_meta, new_vals


def enabler_function(user_id: Optional[str], user_settings: Optional[AgentUserSettings]) -> bool:
    return get_ld_flag("enable-text-to-table-tool", default=False, user_context=user_id)


async def _text_to_table_helper(
    input_texts: List[Text | StockText],
    table_description: str,
    context: PlanRunContext,
    table_schema: Optional[List[str]] = None,
    text_cache: Optional[dict[TextIDType, str]] = None,
) -> Table:
    logger = get_prefect_logger(__name__)
    gpt_context = create_gpt_context(
        GptJobType.AGENT_TOOLS, context.agent_id, GptJobIdType.AGENT_ID
    )
    llm = GPT(context=gpt_context, model=DEFAULT_LLM)

    if len(input_texts) == 0:
        return Table(columns=[])

    initial_text_snippets = await partition_to_smaller_text_sizes(input_texts, context)

    logger.warning("Doing text filtering...")

    texts = await classify_stock_text_relevancies_for_table(
        initial_text_snippets, table_description, gpt_context=gpt_context, text_cache=text_cache
    )

    texts = initial_filter_texts(
        texts
    )  # just in case there's still a crazy number, shouldn't matter
    if len(initial_text_snippets) != len(texts):
        logger.warning(f"Filtered {len(initial_text_snippets)} snippets to {len(texts)}")

    text_group = TextGroup(val=texts)
    texts_str: str = await Text.get_all_strs(  # type: ignore
        text_group,
        include_header=True,
        text_group_numbering=True,
        include_symbols=True,
        text_cache=text_cache,
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
            table_description,
        ],
    )

    table_schema_str = (
        TEXT_TO_TABLES_NO_INPUT_SCHEMA_PROMPT
        if not table_schema
        else TEXT_TO_TABLES_INPUT_SCHEMA_PROMPT.format(header_schema=table_schema)
    )
    main_prompt = table_gen_main_prompt.format(
        table_description=table_description,
        texts=texts_str,
        chat_context=chat_str,
        table_schema=table_schema_str,
        today=(
            context.as_of_date.date().isoformat()
            if context.as_of_date
            else datetime.date.today().isoformat()
        ),
    )
    sys_prompt = table_gen_sys_prompt.format()

    logger.info(f"using GPT to convert {len(texts)} text(s) into table")
    result = await llm.do_chat_w_sys_prompt(
        main_prompt,
        sys_prompt,
    )

    logger.info("get citations")
    # Split the text into the actual csv and the citations
    text, citation_dict = get_initial_breakdown(result)

    text = strip_code_backticks(text)
    df = pd.read_csv(StringIO(text))
    if len(df) <= 2 and df.isnull().values.any():
        logger.error(f"Failed to create table: {text=}")
        return Table(columns=[])

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

    row_indexes_to_skip: set[int] = set()
    column_dicts = []
    for result_task in await asyncio.gather(*tasks):
        metadata: TableColumnMetadata
        new_vals_dict: dict[int, Any]
        metadata, new_vals_dict = result_task
        # Since the key values are simply 0 -> length of table, we want to
        # filter any rows where ANY column in the row is empty. This is mostly
        # used for filtering out rows for stocks that are not matched.
        row_indexes_to_skip.update((i for i in range(len(new_vals_dict)) if i not in new_vals_dict))
        metadatas.append(metadata)
        column_dicts.append(new_vals_dict)

    for metadata, col_dict in zip(metadatas, column_dicts):
        new_vals = [col_dict[i] for i in col_dict.keys() if i not in row_indexes_to_skip]
        new_df_data[metadata.label] = new_vals
    new_df = pd.DataFrame(data=new_df_data)
    new_df = new_df.dropna(how="all")  # Drop rows that are all "No Data"

    table = Table.from_df_and_cols(columns=metadatas, data=new_df)
    if table.get_stock_column():
        return StockTable(
            history=table.history, columns=table.columns, prefer_graph_type=table.prefer_graph_type
        )
    return table


@tool(
    description=f"""
This function takes a list of texts, a table description, and a table
schema (as a list of column names + types), and produces a Table object based on
the data
in the input texts. The schema should be in the format:
    ["col1 title (string)", "col2 title (integer)"] etc.

Possible column types are below, you should only use these when defining the schema:
    {TableColumnType.get_type_explanations()}
Never ever choose two column types, just do your best and choose the one the
fits the best. This format MUST be followed, with the column title followed by
the type in parentheses. You will be fired if you do not conform to these exact
specifications.

You must almost always provide a schema, especially if the client asks for specific
columns, though even if they do not, you should be able to decide on sensible
columns based on their request. The only situation you will NOT provide a schema is if
user just asks to reproduce a table directly from a source (document, website,
etc.), in that situation, you cannot provide a schema since you don't know what will
be in the table. So you will always provide a schema unless the client asks to "reproduce"
a SINGLE specific table from a source, in that case the source table must be specifically
mentioned in the client's request.

Make sure the description is detailed and contains exactly what kind of table
the user wants. The texts will be fed into an LLM to produce a Table output. You
should use this tool if the user explicitly asks for something to be displayed
as a table, or if they ask for some graphing or transformation that requires a
table but you only have text data. You can also use this tool if you need a list
of StockID from the texts, just pair it with the tool to extract stocks from a
table. You also should never use this tool on a summarized text. ONLY use it on
texts taken directly from their sources, with no intervening processing.
""",
    category=ToolCategory.TABLE,
    enabled_checker_func=enabler_function,
)
async def text_to_table(args: TextToTableArgs, context: PlanRunContext) -> Table:
    logger = get_prefect_logger(__name__)
    text_cache: dict[TextIDType, str] = {}
    text_group = TextGroup(val=args.texts)
    _ = await Text.get_all_strs(
        text_group,
        include_header=True,
        text_group_numbering=True,
        include_symbols=True,
        text_cache=text_cache,
    )
    has_security_column = args.table_schema and any(("(stock)" in col for col in args.table_schema))
    stock_to_texts = defaultdict(list)
    for text in args.texts:
        if isinstance(text, StockText):
            stock_to_texts[text.stock_id].append(text)

    if has_security_column and len(stock_to_texts) > 1:
        logger.info("Generating table per stock group...")
        tasks = []
        # TODO compute N smartly
        for stocks in chunk(stock_to_texts.keys(), n=3):
            stock_texts = []
            for stock in stocks:
                stock_texts.extend(stock_to_texts[stock])
            tasks.append(
                _text_to_table_helper(
                    input_texts=stock_texts,  # type: ignore
                    table_description=args.table_description,
                    context=context,
                    table_schema=args.table_schema,
                    text_cache=text_cache,
                )
            )
        tables: List[Table] = await gather_with_concurrency(tasks, n=10)
        all_stocks_mapped = all((table.get_stock_column() is not None for table in tables))
        if not all_stocks_mapped:
            # If there are some stocks that weren't mapped, convert ALL stocks to strings for now.
            # TODO determine if we need a better way to handle this.
            for table in tables:
                stock_col = table.get_stock_column()
                if not stock_col:
                    continue
                stock_col.metadata.col_type = TableColumnType.STRING
                stock_col.data = [
                    stock.symbol if isinstance(stock, StockID) else stock
                    for stock in stock_col.data
                ]
        tables = [table for table in tables if len(table.columns) > 0 and table.get_num_rows() > 0]
        return await join_tables(  # type: ignore
            args=JoinTableArgs(input_tables=tables, row_join=True), context=context
        )

    return await _text_to_table_helper(
        input_texts=args.texts,
        table_description=args.table_description,
        table_schema=args.table_schema,
        text_cache=text_cache,
        context=context,
    )


if __name__ == "__main__":
    # from agent_service.io_type_utils import load_io_type
    from agent_service.utils.logs import init_stdout_logging
    # from agent_service.utils.postgres import get_psql

    # db = get_psql()

    # sql = """
    # SELECT
    #     output
    # FROM agent.task_run_info
    # WHERE
    #     task_id = %(task_id)s
    #     AND plan_run_id = %(plan_run_id)s
    # """

    # row = db.generic_read(
    #     sql,
    #     {
    #         "task_id": "73a36d63-bfc1-4436-bbe3-249e1a85c699",
    #         "plan_run_id": "43a62e12-916f-4c64-9ef4-b5771b5c17bb",
    #     },
    # )[0]
    # texts = load_io_type(row["output"])
    # args = (
    #     TextToTableArgs(
    #         texts=texts,  # type: ignore
    #         table_description=(
    #             "grab all the stocks from Pershing Squares latest "
    #             "13f and display it in a table please"
    #         ),
    #     ),
    # )

    with open("/Users/zach/Documents/agent-service/test.json") as f:
        args = TextToTableArgs.model_validate_json(f.read())

    init_stdout_logging()
    out = asyncio.run(
        text_to_table(
            args=args,
            context=PlanRunContext.get_dummy(),
        )
    )
    print(out)
