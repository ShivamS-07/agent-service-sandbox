import copy
import datetime
import json
import os
import subprocess
import sys
import tempfile
from itertools import chain
from json.decoder import JSONDecodeError
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd
from pydantic import ValidationError

from agent_service.GPT.constants import NO_PROMPT, SONNET
from agent_service.GPT.requests import GPT
from agent_service.io_type_utils import HistoryEntry, dump_io_type, load_io_type
from agent_service.io_types.stock import StockID
from agent_service.io_types.table import (
    StockTable,
    StockTableColumn,
    Table,
    TableColumnMetadata,
    TableColumnType,
)
from agent_service.planner.errors import NonRetriableError
from agent_service.tool import TOOL_DEBUG_INFO, ToolArgs, ToolCategory, tool
from agent_service.tools.table_utils.prompts import (
    DATAFRAME_SCHEMA_GENERATOR_MAIN_PROMPT,
    DATAFRAME_SCHEMA_GENERATOR_SYS_PROMPT,
    DATAFRAME_TRANSFORMER_MAIN_PROMPT,
    DATAFRAME_TRANSFORMER_OLD_CODE_TEMPLATE,
    DATAFRAME_TRANSFORMER_SYS_PROMPT,
    TABLE_ADD_DIFF_MAIN_PROMPT,
    TABLE_REMOVE_DIFF_MAIN_PROMPT,
)
from agent_service.tools.tool_log import tool_log
from agent_service.types import PlanRunContext
from agent_service.utils.async_utils import gather_with_concurrency
from agent_service.utils.gpt_logging import GptJobIdType, GptJobType, create_gpt_context
from agent_service.utils.prefect import get_prefect_logger
from agent_service.utils.tool_diff import get_prev_run_info


def _dump_cols(cols: List[TableColumnMetadata]) -> str:
    return json.dumps([col.model_dump(mode="json") for col in cols])


def _strip_code_markers(gpt_output: str, lang: str) -> str:
    if gpt_output.startswith(f"```{lang}"):
        gpt_output = gpt_output[len(f"```{lang}") :]
    if gpt_output.endswith("```"):
        gpt_output = gpt_output[:-3]

    return gpt_output.strip()


async def gen_new_column_schema(
    gpt: GPT, transformation_description: str, current_table_cols: List[TableColumnMetadata]
) -> List[TableColumnMetadata]:
    logger = get_prefect_logger(__name__)
    prompt = DATAFRAME_SCHEMA_GENERATOR_MAIN_PROMPT.format(
        schema=TableColumnMetadata.schema_json(),
        transform=transformation_description,
        input_cols=_dump_cols(current_table_cols),
        col_type_explain=TableColumnType.get_type_explanations(),
        error="",
    )
    res = await gpt.do_chat_w_sys_prompt(
        main_prompt=prompt, sys_prompt=DATAFRAME_SCHEMA_GENERATOR_SYS_PROMPT
    )
    json_str = _strip_code_markers(res, lang="json")
    try:
        cols = json.loads(json_str.strip())
        if not cols:
            # Empty object = unchanged
            return current_table_cols
        return [TableColumnMetadata.model_validate(item) for item in cols]
    except (ValidationError, JSONDecodeError) as e:
        prompt = DATAFRAME_SCHEMA_GENERATOR_MAIN_PROMPT.format(
            schema=TableColumnMetadata.schema_json(),
            transform=transformation_description,
            input_cols=_dump_cols(current_table_cols),
            col_type_explain=TableColumnType.get_type_explanations(),
            error=(
                "The last time you ran this you got the following error, "
                f"please correct your mistake:\nLast Result:\n{res}\n\nError:\n{str(e)}"
            ),
        )
        logger.warning("Invalid response from GPT, trying again")
        res = await gpt.do_chat_w_sys_prompt(
            main_prompt=prompt, sys_prompt=DATAFRAME_SCHEMA_GENERATOR_SYS_PROMPT
        )
        json_str = _strip_code_markers(res, lang="json")
        cols = json.loads(json_str)
        return [TableColumnMetadata.model_validate(item) for item in cols]


class TransformTableArgs(ToolArgs):
    input_table: Table
    transformation_description: str


def _get_command(data_file: str, code_file: str) -> str:
    exec_code_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "table_utils/pandas_exec.py"
    )
    command = f"{sys.executable} {exec_code_path} -d {data_file} -c {code_file}"

    if sys.platform == "darwin":
        # This is mostly for testing purposes
        helper_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "table_utils/macos_sandbox_exec_config.sb"
        )
        command = f"sandbox-exec -f {helper_path} {command}"
    elif sys.platform == "win32":
        logger = get_prefect_logger(__name__)
        logger.warning("Running LLM generated code on Windows is not safe!")

    else:
        command = f"unshare -n -r {command}"
    return command


def _run_transform_code(df: pd.DataFrame, code: str) -> Tuple[Optional[pd.DataFrame], str]:
    if code.startswith("```python"):
        code = code[9:]
    if code.endswith("```"):
        code = code[:-3]
    delete = (
        sys.platform != "win32"
    )  # turning off delete is necessary for temp file to work in Windows
    with tempfile.NamedTemporaryFile(
        mode="w+", delete=delete
    ) as code_file, tempfile.NamedTemporaryFile(mode="w+", delete=delete) as data_file:
        code_file.write(code)
        code_file.flush()
        serialized = df.to_json()
        data_file.write(serialized)
        data_file.flush()
        command: str = _get_command(data_file=data_file.name, code_file=code_file.name)
        ret = subprocess.run(command, text=True, shell=True, capture_output=True)

    if ret.returncode == 0:
        json_str = ret.stdout
        return (pd.read_json(json_str), "")
    else:
        return (None, ret.stderr)


def _get_df_info(df: pd.DataFrame) -> str:
    return f"""
    Number of rows: {len(df)}
    Columns: {df.columns}
    """


async def table_filter_added_diff_info(
    added_stocks: Set[StockID],
    transformation: str,
    curr_stock_values: Dict[StockID, Dict[str, float]],
    prev_stock_values: Dict[StockID, Dict[str, float]],
    agent_id: str,
) -> Dict[StockID, str]:
    gpt_context = create_gpt_context(GptJobType.AGENT_TOOLS, agent_id, GptJobIdType.AGENT_ID)
    llm = GPT(context=gpt_context, model=SONNET)

    tasks = []
    for stock in added_stocks:
        tasks.append(
            llm.do_chat_w_sys_prompt(
                TABLE_ADD_DIFF_MAIN_PROMPT.format(
                    company_name=stock.company_name,
                    transformation=transformation,
                    curr_stats=curr_stock_values[stock],
                    prev_stats=prev_stock_values[stock],
                ),
                NO_PROMPT,
            )
        )

    results = await gather_with_concurrency(tasks)
    return {stock: explanation for stock, explanation in zip(added_stocks, results)}


async def table_filter_removed_diff_info(
    removed_stocks: Set[StockID],
    transformation: str,
    curr_stock_values: Dict[StockID, Dict[str, float]],
    prev_stock_values: Dict[StockID, Dict[str, float]],
    agent_id: str,
) -> Dict[StockID, str]:
    gpt_context = create_gpt_context(GptJobType.AGENT_TOOLS, agent_id, GptJobIdType.AGENT_ID)
    llm = GPT(context=gpt_context, model=SONNET)

    tasks = []
    for stock in removed_stocks:
        tasks.append(
            llm.do_chat_w_sys_prompt(
                TABLE_REMOVE_DIFF_MAIN_PROMPT.format(
                    company_name=stock.company_name,
                    transformation=transformation,
                    curr_stats=curr_stock_values[stock],
                    prev_stats=prev_stock_values[stock],
                ),
                NO_PROMPT,
            )
        )

    results = await gather_with_concurrency(tasks)
    return {stock: explanation for stock, explanation in zip(removed_stocks, results)}


@tool(
    description="""This is a function that allows you to do aribtrary transformations on Table objects.
Tables are simply wrappers around pandas dataframes. There are a few primary use cases for this function:
- Sorting tables
- Filtering or ranking tables based on numeric criteria
- Aggregating tables ACROSS row or stocks

For things like percent change of price, or other per-stock calculations, please
use the `get_statistic_data_for_companies` function instead. Do NOT use this.

The `transformation_description` argument is a free text description of a
transformation that will be applied to the table by an LLM, so feel free to be
detailed in your description of the desired transformation. Anything that could
be done in pandas is supported here. Simple table formatting should not use
this. It is better to be overly detailed than not detailed enough. Note again
that the input MUST be a table, not a list!
You must never pass a vague transformation_description into this tool, you must always be fully
explicit about the operations, particularly filtering. If the client only mentioned, for example,
a filtering to "high" or "good" values without providing a definition of what "high" or "good"
means, the transformation_description must not use words like "high" or "good", and instead
include an exact numerical cutoff appropriate to the particular statistic mentioned.
For example, a high market cap might be considered 10 billion, so if a user asked for high
market cap companies, the transformation_description should explicitly ask for a filtering of
stock to those with market cap greater than 10 billion.
If you are filtering stocks, the transformation description must begin with the word `Filter`
""",
    category=ToolCategory.TABLE,
)
async def transform_table(args: TransformTableArgs, context: PlanRunContext) -> Table:
    logger = get_prefect_logger(__name__)

    old_schema: Optional[List[TableColumnMetadata]] = None
    old_code = None
    prev_args = None
    prev_output_table = None
    try:  # since everything here is optional, put in try/except
        prev_run_info = await get_prev_run_info(context, "transform_table")
        if prev_run_info is not None:
            prev_args: TransformTableArgs = prev_run_info[0]  # type: ignore
            prev_output_table: Table = prev_run_info[1]  # type:ignore
            prev_other: Dict[str, str] = prev_run_info[2]  # type:ignore
            if prev_other:
                old_code = (
                    prev_other["code_second_attempt"]
                    if "code_second_attempt" in prev_other
                    else prev_other["code_first_attempt"]
                )
                old_schema = load_io_type(prev_other["table_schema"])  # type:ignore

    except Exception as e:
        logger.warning(f"Error doing getting info from previous run: {e}")

    debug_info: Dict[str, Any] = {}
    TOOL_DEBUG_INFO.set(debug_info)
    input_col_metadata = [col.metadata for col in args.input_table.columns]
    gpt_context = create_gpt_context(
        GptJobType.AGENT_TOOLS, context.agent_id, GptJobIdType.AGENT_ID
    )
    gpt = GPT(context=gpt_context)
    if old_schema:
        await tool_log(log="Using table schema from previous run", context=context)
        new_col_schema = old_schema

    else:
        await tool_log(log="Computing new table schema", context=context)
        new_col_schema = await gen_new_column_schema(
            gpt,
            transformation_description=args.transformation_description,
            current_table_cols=input_col_metadata,
        )

    debug_info["table_schema"] = dump_io_type(new_col_schema)
    await tool_log(log="Transforming table", context=context)
    data_df = args.input_table.to_df(stocks_as_hashables=True)
    if old_code:
        old_code_section = DATAFRAME_TRANSFORMER_OLD_CODE_TEMPLATE.format(old_code=old_code)
    else:
        old_code_section = ""

    code = await gpt.do_chat_w_sys_prompt(
        main_prompt=DATAFRAME_TRANSFORMER_MAIN_PROMPT.format(
            col_schema=_dump_cols(input_col_metadata),
            output_schema=_dump_cols(new_col_schema),
            info=_get_df_info(data_df),
            transform=args.transformation_description,
            col_type_explain=TableColumnType.get_type_explanations(),
            today=datetime.date.today(),
            error="",
            old_code=old_code_section,
        ),
        sys_prompt=DATAFRAME_TRANSFORMER_SYS_PROMPT,
    )
    debug_info["code_first_attempt"] = code
    logger.info(f"Running transform code:\n{code}")
    output_df, error = _run_transform_code(df=data_df, code=code)
    if output_df is None:
        logger.warning("Failed when transforming dataframe... trying again")
        logger.warning(f"Failing code:\n{code}")
        code = await gpt.do_chat_w_sys_prompt(
            main_prompt=DATAFRAME_TRANSFORMER_MAIN_PROMPT.format(
                col_schema=_dump_cols(input_col_metadata),
                output_schema=_dump_cols(new_col_schema),
                info=_get_df_info(data_df),
                transform=args.transformation_description,
                col_type_explain=TableColumnType.get_type_explanations(),
                today=datetime.date.today(),
                error=(
                    "Your last code failed with this error, please correct it:\n"
                    f"Last Code:\n\n{code}\n\n"
                    f"Error:\n{error}"
                ),
                old_code=old_code_section,
            ),
            sys_prompt=DATAFRAME_TRANSFORMER_SYS_PROMPT,
        )
        debug_info["code_second_attempt"] = code
        logger.info(f"Running transform code:\n{code}")
        output_df, error = _run_transform_code(df=data_df, code=code)
        if output_df is None:
            raise RuntimeError(f"Table transformation subprocess failed with:\n{error}")

    output_table = Table.from_df_and_cols(
        columns=new_col_schema, data=output_df, stocks_are_hashable_objs=True
    )

    await tool_log(
        log=f"Transformed table has {len(output_table.columns[0].data)} rows", context=context
    )

    if output_table.get_num_rows() == 0:
        raise NonRetriableError(message="Table transformation resulted in an empty table")

    if output_table.get_stock_column():
        output_table = StockTable(columns=output_table.columns)

    try:  # since everything here is optional, put in try/except
        if (
            context.diff_info is not None
            and prev_args
            and isinstance(args.input_table, StockTable)
            and args.transformation_description.lower().startswith("filter")
        ):
            curr_input_table: StockTable = args.input_table  # type: ignore
            prev_input_table: StockTable = prev_args.input_table  # type: ignore
            curr_output_table: StockTable = output_table  # type: ignore
            prev_output_table: StockTable = prev_output_table  # type: ignore
            curr_stock_values = curr_input_table.get_values_for_stocks()
            prev_stock_values = prev_input_table.get_values_for_stocks()
            shared_input_stocks = set(args.input_table.get_stocks()) & set(
                args.input_table.get_stocks()
            )
            curr_output_stocks = set(curr_output_table.get_stocks())
            prev_output_stocks = set(prev_output_table.get_stocks())
            added_stocks = (curr_output_stocks - prev_output_stocks) & shared_input_stocks
            removed_stocks = (prev_output_stocks - curr_output_stocks) & shared_input_stocks
            added_diff_info = await table_filter_added_diff_info(
                added_stocks,
                args.transformation_description,
                curr_stock_values,
                prev_stock_values,
                context.agent_id,
            )

            removed_diff_info = await table_filter_removed_diff_info(
                removed_stocks,
                args.transformation_description,
                curr_stock_values,
                prev_stock_values,
                context.agent_id,
            )

            context.diff_info[context.task_id] = {
                "added": added_diff_info,
                "removed": removed_diff_info,
            }

    except Exception as e:
        logger.warning(f"Error doing diff from previous run: {e}")

    return output_table


class JoinTableArgs(ToolArgs):
    input_tables: List[Table]


def _join_two_tables(first: Table, second: Table) -> Table:
    # Find columns to join by, ideally a date column, stock column, or both
    first_stock_col = None
    second_stock_col = None
    first_date_col = None
    second_date_col = None
    # Other cols = cols not used for the join key
    first_other_cols = []
    second_other_cols = []
    other_cols = []

    # First, find the date and stock columns for both tables if they are present.
    for col in first.columns:
        if not first_date_col and col.metadata.col_type.is_date_type():
            first_date_col = col
            continue
        if not first_stock_col and col.metadata.col_type == TableColumnType.STOCK:
            first_stock_col = col
            continue
        first_other_cols.append(col.metadata)
    for col in second.columns:
        if not second_date_col and col.metadata.col_type.is_date_type():
            second_date_col = col
            continue
        if not second_stock_col and col.metadata.col_type == TableColumnType.STOCK:
            second_stock_col = col
            continue
        second_other_cols.append(col.metadata)

    other_cols = first_other_cols + second_other_cols
    first_data = first.to_df()
    second_data = second.to_df()

    # Merge the stocks' histories together if necessary
    stock_hash_to_merged_stock_obj_map: Dict[StockID, StockID] = {}
    if first_stock_col and second_stock_col:
        for val in chain(first_stock_col.data, second_stock_col.data):
            if not val:
                continue
            if val not in stock_hash_to_merged_stock_obj_map:
                stock_hash_to_merged_stock_obj_map[val] = val  # type: ignore
            else:
                stock_hash_to_merged_stock_obj_map[val].union_history_with(val)  # type: ignore

    # Go case by case:
    #   1. Join on stocks AND dates
    #   2. Join on just stocks
    #   3. Join on just dates
    output_df = pd.concat((first_data, second_data))
    if first_stock_col and second_stock_col and first_date_col and second_date_col:
        key_cols = [first_date_col.metadata, first_stock_col.metadata]
    elif first_stock_col and second_stock_col:
        key_cols = [first_stock_col.metadata]
    elif first_date_col and second_date_col:
        key_cols = [first_date_col.metadata]
    else:
        # Can't join on anything! Just concat
        return Table(columns=first.columns + second.columns)

    # Collapse rows with the same keys
    output_df = output_df.groupby(by=[col.label for col in key_cols]).first().reset_index()  # type: ignore
    output_col_metas = key_cols + other_cols  # type: ignore
    output_table = Table.from_df_and_cols(columns=output_col_metas, data=output_df)

    # Make sure we use the objects' full histories
    stock_col = output_table.get_stock_column()
    if stock_col:
        stock_col.data = [
            stock_hash_to_merged_stock_obj_map.get(stock, stock) for stock in stock_col.data  # type: ignore
        ]
    return output_table


@tool(
    description="""Given a list of input tables, attempt to join the tables into
a single table. Ideally, the tables will share a column or two that can be used
to join (e.g. a stock column or a date column). This will create a single table
from the multiple inputs. If you want to transform multiple tables, they must be
merged with this first.
""",
    category=ToolCategory.TABLE,
)
async def join_tables(args: JoinTableArgs, context: PlanRunContext) -> Table:
    if len(args.input_tables) == 0:
        raise RuntimeError("Cannot join an empty list of tables!")
    if len(args.input_tables) == 1:
        raise RuntimeError("Cannot join a list of tables with one element!")

    joined_table = args.input_tables[0]
    for table in args.input_tables[1:]:
        joined_table = _join_two_tables(joined_table, table)

    if joined_table.get_stock_column():
        return StockTable(columns=joined_table.columns)
    return joined_table


class JoinStockListTableArgs(ToolArgs):
    input_table: StockTable
    stock_list: List[StockID]


@tool(
    description="""Given an input stock table and a list of stocks, join the
list of stocks to the table. All the metadata that is tracked in the stock ID's
will be merged together. You should call this function when you have a list of
stocks and a StockTable of data for the SAME set of stocks, and you want to
display everything combined in a single table. Ideally, the table should be
derived in some way from the stock list. For example, you could get a list of
recommended stocks, get their close prices in a table, and then merge the
recommended stock list with the table to display both the recommendation reason
and the statistical data.
""",
    category=ToolCategory.TABLE,
)
async def join_stock_list_to_table(
    args: JoinStockListTableArgs, context: PlanRunContext
) -> StockTable:
    new_stock_table = StockTable(columns=[StockTableColumn(data=args.stock_list)])
    joined_table = _join_two_tables(first=args.input_table, second=new_stock_table)
    return StockTable(
        columns=joined_table.columns,
        history=joined_table.history,
        prefer_graph_type=joined_table.prefer_graph_type,
    )


class GetStockListFromTableArgs(ToolArgs):
    input_table: Table


@tool(
    description="""Given a table with at least one column of stocks, extract
that column into a list of stock ID's.  This is very useful for e.g. filtering
on some numerical data in a table before extracting the stock list and fetching
other data with it.
The tool can be used to convert a table with a stock column from another tool
like get_portfolio_holdings into a list of stock ID's. This function can only be
used with actual tables, it cannot be used with either lists of texts or lists
of stocks.
Important: When stocks that are taken from a table are displayed to the client via the
output tool, they will see any important statistics that have been calculated the stock,
in table format. It is entirely redundant to display both the list of stocks and a table
they were extracted from, you must output only one or the other, never both!
""",
    category=ToolCategory.TABLE,
)
async def get_stock_identifier_list_from_table(
    args: GetStockListFromTableArgs, context: PlanRunContext
) -> List[StockID]:
    stock_column = None
    # Columns after the stock column
    rest_columns = None
    for i, col in enumerate(args.input_table.columns):
        if isinstance(col, StockTableColumn):
            stock_column = col
            rest_columns = args.input_table.columns[i + 1 :]
            break
    if not stock_column or not rest_columns:
        raise RuntimeError("Cannot extract list of stocks, no stock column in table!")
    # Don't update in place to prevent issues in case this table is used elsewhere.
    # Use a set to prevent duplicates.
    stocks = copy.deepcopy(stock_column.data)
    for col in rest_columns:
        for i, stock in enumerate(stocks):
            stock.history.append(
                HistoryEntry(
                    title=str(col.metadata.label),
                    entry_type=col.metadata.col_type,
                    unit=col.metadata.unit,
                    explanation=col.data[i],  # type: ignore
                )
            )

    seen_stocks = set()
    outputs = []
    # Do another loop to maintain stock ordering
    for stock in stocks:
        if stock in seen_stocks:
            continue
        outputs.append(stock)
        seen_stocks.add(stock)
    return outputs
