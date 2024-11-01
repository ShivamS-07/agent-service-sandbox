import copy
import datetime
import json
import os
import subprocess
import sys
import tempfile
import traceback
from itertools import chain
from json.decoder import JSONDecodeError
from typing import Any, Dict, List, Optional, Set, Tuple, Union, cast

import pandas as pd
from pydantic import ValidationError

from agent_service.GPT.constants import GPT4_O, NO_PROMPT, O1, SONNET
from agent_service.GPT.requests import GPT
from agent_service.io_type_utils import (
    ComplexIOBase,
    HistoryEntry,
    dump_io_type,
    load_io_type,
)
from agent_service.io_types.stock import StockID
from agent_service.io_types.stock_groups import StockGroups
from agent_service.io_types.table import (
    StockTable,
    StockTableColumn,
    Table,
    TableColumnMetadata,
    TableColumnType,
    object_histories_to_columns,
)
from agent_service.planner.errors import EmptyOutputError
from agent_service.tool import TOOL_DEBUG_INFO, ToolArgs, ToolCategory, tool
from agent_service.tools.stock_groups.utils import (
    add_stock_group_column,
    get_stock_group_input_tables,
    remove_stock_group_columns,
)
from agent_service.tools.table_utils.prompts import (
    DATAFRAME_SCHEMA_GENERATOR_MAIN_PROMPT,
    DATAFRAME_SCHEMA_GENERATOR_SYS_PROMPT,
    DATAFRAME_TRANSFORMER_MAIN_PROMPT,
    DATAFRAME_TRANSFORMER_OLD_CODE_TEMPLATE,
    PICK_GPT_MAIN_PROMPT,
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
        schema=TableColumnMetadata.to_gpt_schema(),
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
            schema=TableColumnMetadata.to_gpt_schema(),
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
        serialized = df.to_json(date_format="iso")
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
    curr_stock_values: Dict[StockID, Any],
    prev_stock_values: Dict[StockID, Any],
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
    curr_stock_values: Dict[StockID, Any],
    prev_stock_values: Dict[StockID, Any],
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

For things like percent change of price, or other per-stock, across time calculations, please
use the `get_statistic_data_for_companies` function instead,  do NOT use this function. In particular,
if a user expresses a statistic that is a simple mathematical combination of two simpler
statistics (earnings over assets, debt minus assets), you must pass that full mathematical expression
to the get_statistic_data_for_companies tool, avoid using this tool in those situations.
Similarity, if a user asks for some averaging of some statistic across time,
(average P/E over the last 3 years), you must NOT do the averaging in this function, but
instead pass full statistic to get_statistic_data_for_companies (i.e. the entire statistic would
be 'average P/E over the last 3 years').

The `transformation_description` argument is a free text description of a
transformation that will be applied to the table by an LLM, so feel free to be
detailed in your description of the desired transformation. Anything that could
be done in pandas is supported here. Simple table formatting should not use
this. It is better to be overly detailed than not detailed enough. Note again
that the input MUST be a table, not a list!

Note that if you are doing some kind of stock aggregation (e.g. averaging), it is critical that
you tell this tool explicitly what kind of stocks you are averaging, so the resulting row in the output
table can be properly labeled. That is, in such cases, the transformation_description should NOT be
something vague like 'average stocks' but rather something specific like 'average Healthcare stocks
in the QQQ'. Do this in all cases, but this is especially critical if you are doing comparisons of
aggregate statistics across different groupings of stocks.

Note that if you are trying to use this tool to calculate a sector average of some statistic, and
if you want a table/graph organized by sectors, you must use the table transform for stock groups.
Do not use this tool to calculate basic sector statistics when the user wants per sector output.
I repeat, this tool must NEVER, EVER be used for 'per sector' calculations!

However, if you want an average sector statistic for each stock, you may use this tool, but before
you call the tool you must join the table with your statistics for each stock with a table with the
sector for each stock. Not only is it essential you add a sector column to the input table,
you must be also be fully explicit in your transformation_description that you want a sector average
output for each stock, not per sector!
That is, do NOT say `calculate the average X for each sector` (if you are doing such a calculation,
you should be using the per stock groups transform tool), instead you must say `calculate the sector
average X for each stock, the output table must have a row for every stock`. You MUST do a per stock
calculation if the user wants a per stock output, if you do a per sector calculation and then try to
join it with per stock statistics, the tables will not join and you will fail at your task and be fired.
I repeat, if you are calling this tool with a request to calculate a per sector statistc, you are wrong
and you will be fired!!!!!!!!!

If you are doing a transform calculation where you are using columns of the table to do the calculation,
but the client clearly wants to keep the input columns or any other columns currently in the table in
the final output, you must always explicitly mention in the transformation description that you do not
wish to drop those columns. Otherwise, the default behavior of this tool when creating new columns is
to drop any columns that are not created in the process.
For example, if the client asks for a table with price, earnings, and price to earnings, and asks
you to calculate price to earnings manually from an input table of price and earnings, you would say
'calculate price to earnings, please keep the price and earnings columns'. Otherwise price and earnings
columns will be removed from the output, which will make the client very unhappy in this case. You should
always consider whether or not you wish to drop the columns every time you transform a table which generates
a new column.

Do not use this tool directly to calculate monthly/yearly change rates (e.g. revenue growth) for
stocks. You must use the get_statistic_for_companies tool in all cases where the calculation
can be done independently per stock, this tool should only be used for cross-stock calculations!
This is true even if you have already retrieved the underlying data you need for the calculation.

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
async def transform_table(
    args: TransformTableArgs, context: PlanRunContext
) -> Union[Table, StockTable]:
    logger = get_prefect_logger(__name__)
    old_schema: Optional[List[TableColumnMetadata]] = None
    old_code: Optional[str] = None
    prev_args = None
    prev_output = None
    try:  # since everything here is optional, put in try/except
        prev_run_info = await get_prev_run_info(context, "transform_table")
        if prev_run_info is not None:
            prev_args = TransformTableArgs.model_validate_json(prev_run_info.inputs_str)
            prev_output = prev_run_info.output  # type:ignore
            prev_other: Dict[str, str] = prev_run_info.debug  # type:ignore
            if prev_other:
                old_code = (
                    prev_other["code_second_attempt"]
                    if "code_second_attempt" in prev_other
                    else prev_other["code_first_attempt"]
                )
                old_schema = load_io_type(prev_other["table_schema"])  # type:ignore

    except Exception as e:
        logger.warning(f"Error getting info from previous run: {e}")

    debug_info: Dict[str, Any] = {}
    TOOL_DEBUG_INFO.set(debug_info)

    list_of_one_table = await transform_tables_helper(
        args.transformation_description,
        [args.input_table],
        context,
        old_code=old_code,
        old_schema=old_schema,
        debug_info=debug_info,
    )
    output_table = list_of_one_table[0]

    await tool_log(
        log=f"Transformed table has {len(output_table.columns[0].data)} rows", context=context
    )

    if output_table.get_num_rows() == 0:
        raise EmptyOutputError(message="Table transformation resulted in an empty table")

    if output_table.get_stock_column():
        output_table = StockTable(columns=output_table.columns)
        if context.task_id:
            try:  # since everything here is optional, put in try/except
                output_table.add_task_id_to_history(context.task_id)
                if (
                    context.diff_info is not None
                    and prev_args
                    and prev_output
                    and isinstance(args.input_table, StockTable)
                    and isinstance(prev_output, StockTable)
                    and args.transformation_description.lower().startswith("filter")
                ):
                    curr_input_table: StockTable = args.input_table  # type: ignore
                    prev_input_table: StockTable = prev_args.input_table  # type: ignore
                    curr_output_table: StockTable = output_table  # type: ignore
                    prev_output_table: StockTable = prev_output  # type: ignore
                    curr_stock_values = curr_input_table.get_values_for_stocks()
                    prev_stock_values = prev_input_table.get_values_for_stocks()
                    shared_input_stocks = set(curr_input_table.get_stocks()) & set(
                        prev_input_table.get_stocks()
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


async def transform_tables_helper(
    description: str,
    tables: List[Table],
    context: PlanRunContext,
    old_code: Optional[str] = None,
    old_schema: Optional[List[TableColumnMetadata]] = None,
    debug_info: Dict[str, Any] = {},
) -> List[Table]:
    logger = get_prefect_logger(__name__)

    input_col_metadata = [col.metadata for col in tables[0].columns]
    labels = [metadata.label for metadata in input_col_metadata]
    gpt_context = create_gpt_context(
        GptJobType.AGENT_TOOLS, context.agent_id, GptJobIdType.AGENT_ID
    )
    old_gpt = GPT(context=gpt_context)  # keep using turbo for the schema generation
    medium_gpt = GPT(model=GPT4_O, context=gpt_context)
    challenge = await medium_gpt.do_chat_w_sys_prompt(
        PICK_GPT_MAIN_PROMPT.format(task=description, labels=labels), NO_PROMPT
    )

    if challenge.lower().strip() == "easy":
        gpt = GPT(model=GPT4_O, context=gpt_context)
    else:
        gpt = GPT(model=O1, context=gpt_context)
        await tool_log("Brainstorming in advance of calculation", context)
    if old_schema:
        await tool_log(log="Using table schema from previous run", context=context)
        new_col_schema = old_schema

    else:
        await tool_log(log="Computing new table schema", context=context)
        new_col_schema = await gen_new_column_schema(
            old_gpt,
            transformation_description=description,
            current_table_cols=input_col_metadata,
        )

    debug_info["table_schema"] = dump_io_type(new_col_schema)
    logger.info(f"Table Schema: {debug_info['table_schema']}")
    await tool_log(log="Transforming table", context=context)
    data_dfs = [input_table.to_df(stocks_as_hashables=True) for input_table in tables]
    if old_code:
        old_code_section = DATAFRAME_TRANSFORMER_OLD_CODE_TEMPLATE.format(old_code=old_code)
    else:
        old_code_section = ""

    code = await gpt.do_chat_w_sys_prompt(
        main_prompt=DATAFRAME_TRANSFORMER_MAIN_PROMPT.format(
            col_schema=_dump_cols(input_col_metadata),
            output_schema=_dump_cols(new_col_schema),
            info=_get_df_info(data_dfs[0]),
            transform=description,
            col_type_explain=TableColumnType.get_type_explanations(),
            today=datetime.date.today(),
            error="",
            old_code=old_code_section,
        ),
        temperature=1.0,
        # sys_prompt=DATAFRAME_TRANSFORMER_SYS_PROMPT,
        sys_prompt=NO_PROMPT,
    )
    debug_info["code_first_attempt"] = code
    logger.info(f"Running transform code:\n{code}")
    output_dfs = []
    had_error = False
    error = None
    for data_df in data_dfs:
        output_df, error = _run_transform_code(df=data_df, code=code)

        if output_df is not None:
            for col_meta in new_col_schema:
                df_col = col_meta.label
                if df_col not in output_df.columns:
                    error = (
                        f"Output DF schema not correct! Requested column: "
                        f"'{df_col}' not in dataframe cols: {output_df.columns}"
                    )
        if output_df is None or error:
            had_error = True
            break
        try:
            table = Table.from_df_and_cols(
                columns=new_col_schema, data=output_df, stocks_are_hashable_objs=True
            )
        except Exception:
            error = traceback.format_exc()
            had_error = True
            break

        if table.get_num_rows() == 0:
            error = "Table transformation resulted in an empty table"
            had_error = True
            break

        output_dfs.append(output_df)
    if had_error:
        output_dfs = []
        logger.warning("Failed when transforming dataframe... trying again")
        logger.warning(f"Failing code:\n{code}")
        code = await gpt.do_chat_w_sys_prompt(
            main_prompt=DATAFRAME_TRANSFORMER_MAIN_PROMPT.format(
                col_schema=_dump_cols(input_col_metadata),
                output_schema=_dump_cols(new_col_schema),
                info=_get_df_info(data_dfs[0]),
                transform=description,
                col_type_explain=TableColumnType.get_type_explanations(),
                today=datetime.date.today(),
                error=(
                    "Your last code failed with this error. If it is a validation error, "
                    "it may be because you transformed the dataframe in such a way that "
                    "it is now unable to be read. "
                    "Please correct it:\n"
                    f"Last Code:\n\n{code}\n\n"
                    f"Error:\n{error}"
                ),
                old_code=old_code_section,
            ),
            # sys_prompt=DATAFRAME_TRANSFORMER_SYS_PROMPT,
            sys_prompt=NO_PROMPT,
            temperature=1.0,
        )
        debug_info["code_second_attempt"] = code
        logger.info(f"Running transform code:\n{code}")
        for data_df in data_dfs:
            output_df, error = _run_transform_code(df=data_df, code=code)
            if output_df is None:
                raise RuntimeError(f"Table transformation subprocess failed with:\n{error}")
            output_dfs.append(output_df)

    # ensure date-derived columns are actually strings (2024Q1, etc)
    output_tables = []
    for output_df in output_dfs:
        for col in new_col_schema:
            if col.col_type == TableColumnType.QUARTER:
                output_df[col.label] = (
                    pd.to_datetime(output_df[col.label]).dt.to_period("Q").astype(str)
                )
            if col.col_type == TableColumnType.YEAR or col.col_type == TableColumnType.MONTH:
                output_df[col.label] = output_df[col.label].astype(str)

        output_table = Table.from_df_and_cols(
            columns=new_col_schema, data=output_df, stocks_are_hashable_objs=True
        )
        output_tables.append(output_table)

    return output_tables


class PerStockGroupTransformTableInput(ToolArgs):
    input_table: StockTable
    stock_groups: StockGroups
    transformation_description: str


@tool(
    description="""This is a tool that allows you to apply an arbitrary table transformation
    as described in tranformation_description on each group in the input stock_groups. The data
    for all stocks is found in the input_table, the tool will extract the relevant rows from input_table
    for each stock, apply the transformation, and then build a new table with the results,
    including a new column which indicates the stock group(usually but not always replacing the Stock column).
    In terms of the properties of this tool, most of the instructions for the transform_table tool applies
    equally to this tool, please read them carefully.
    There is one important thing to note in the transformation_description: the transformation must
    not mention the stock groups directly or indirectly, and instead must simply describe the transformation
    applied to a single stock group. For example, if the client has asked for 'the average market cap
    of each sector in the S&P 500' and you have already created a StockGroup object which groups the stocks
    by sectors, you will NOT mention sectors or stock groups in your transformation_description passed to
    this tool, instead the transformation should simply describe what you will do for each group, e.g.
    'calculate the average market cap for the input stocks'. Again, it is extremely important that your
    transformation description does not reflect that you are applying this operation iteratively across
    stock groups, you must describe only the transformation that will be applied to each stock group.
    This is the single most important thing you must pay attention to! Also, when preparing data for this
    tool, you should not include the data used to create the stock groups (e.g. sector data) unless
    otherwise required for the calculation, the StockGroups object already contains that information,
    to have it in the input table too is entirely redundant and may result in duplicate columns.
    Otherwise, follow the instructions for transform_table.
""",
    category=ToolCategory.TABLE,
)
async def per_stock_group_transform_table(
    args: PerStockGroupTransformTableInput, context: PlanRunContext
) -> Union[Table, StockTable]:
    logger = get_prefect_logger(__name__)
    old_schema: Optional[List[TableColumnMetadata]] = None
    old_code: Optional[str] = None
    # TODO: Consider doing stock list diffing for stock group filtering
    # prev_args = None
    # prev_output = None
    try:  # since everything here is optional, put in try/except
        prev_run_info = await get_prev_run_info(context, "per_stock_group transform_table")
        if prev_run_info is not None:
            # prev_args = TransformTableArgs.model_validate_json(prev_run_info.inputs_str)
            # prev_output = prev_run_info.output  # type:ignore
            prev_other: Dict[str, str] = prev_run_info.debug  # type:ignore
            if prev_other:
                old_code = (
                    prev_other["code_second_attempt"]
                    if "code_second_attempt" in prev_other
                    else prev_other["code_first_attempt"]
                )
                old_schema = load_io_type(prev_other["table_schema"])  # type:ignore

    except Exception as e:
        logger.warning(f"Error getting info from previous run: {e}")

    debug_info: Dict[str, Any] = {}
    TOOL_DEBUG_INFO.set(debug_info)

    stock_group_input_tables = get_stock_group_input_tables(args.input_table, args.stock_groups)

    stock_group_output_tables = await transform_tables_helper(
        args.transformation_description,
        stock_group_input_tables,  # type:ignore
        context,
        old_code=old_code,
        old_schema=old_schema,
        debug_info=debug_info,
    )

    for table, group in zip(stock_group_output_tables, args.stock_groups.stock_groups):
        remove_stock_group_columns(table, args.stock_groups.header)
        add_stock_group_column(table, args.stock_groups.header, group.name)

    joined_table = stock_group_output_tables[0]
    for table in stock_group_output_tables[1:]:
        joined_table = _join_two_tables_vertically(joined_table, table)

    if joined_table.get_stock_column():
        return StockTable(columns=joined_table.columns)

    return joined_table


class JoinTableArgs(ToolArgs):
    input_tables: List[Table]
    # If true, join vertically instead of horizontally, resulting in more rows
    # instead of more columns.
    row_join: bool = False


def _join_two_tables_vertically(first: Table, second: Table) -> Table:
    output_cols = []
    for col1, col2 in zip(first.columns, second.columns):
        if col1.metadata.col_type == col2.metadata.col_type:
            new_col = copy.deepcopy(col1)
            new_col.data.extend(col2.data)
            new_col.union_history_with(col2)
            output_cols.append(new_col)

    output_table = Table(columns=output_cols, prefer_graph_type=first.prefer_graph_type)
    output_table.history = copy.deepcopy(first.history) + copy.deepcopy(second.history)
    output_table.dedup_history()
    return output_table


def _join_two_tables(first: Table, second: Table) -> Table:
    have_stock_columns = (
        first.get_stock_column() is not None and second.get_stock_column is not None
    )
    stock_col_type = TableColumnType.STOCK
    if not have_stock_columns:
        stock_col_type = TableColumnType.STRING
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
        if not first_stock_col and col.metadata.col_type == stock_col_type:
            first_stock_col = col
            continue
        first_other_cols.append(col.metadata)
    for col in second.columns:
        if not second_date_col and col.metadata.col_type.is_date_type():
            second_date_col = col
            continue
        if not second_stock_col and col.metadata.col_type == stock_col_type:
            second_stock_col = col
            continue
        second_other_cols.append(col.metadata)

    other_cols = first_other_cols + second_other_cols
    first_data = first.to_df()
    second_data = second.to_df()

    # Merge the stocks' histories together if necessary
    stock_hash_to_merged_stock_obj_map: Dict[StockID, StockID] = {}
    if first_stock_col and second_stock_col and stock_col_type == TableColumnType.STOCK:
        for val in chain(first_stock_col.data, second_stock_col.data):
            if not val:
                continue
            if val not in stock_hash_to_merged_stock_obj_map:
                stock_hash_to_merged_stock_obj_map[val] = val  # type: ignore
            else:
                stock_hash_to_merged_stock_obj_map[val].union_history_with(val)  # type: ignore

    # Go case by case:
    #   1. Join on stocks AND dates, do merge so weights can be spread across dates or stocks if needed
    #   2. Join on just stocks
    #   3. Join on just dates
    key_cols = []
    shared_cols = []
    if first_date_col and second_date_col:
        shared_cols.append(first_date_col.metadata)
    elif first_date_col:
        key_cols.append(first_date_col.metadata)
    elif second_date_col:
        key_cols.append(second_date_col.metadata)

    if first_stock_col and second_stock_col:
        shared_cols.append(first_stock_col.metadata)
    elif first_stock_col:
        key_cols.append(first_stock_col.metadata)
    elif second_stock_col:
        key_cols.append(second_stock_col.metadata)

    if not shared_cols:
        # Can't join on anything! Just concat
        return Table(columns=first.columns + second.columns)

    key_cols += shared_cols

    output_df = pd.merge(
        left=first_data,
        right=second_data,
        on=[col.label for col in shared_cols],
        how="outer",
        suffixes=(None, "_ignored"),
    )

    # Collapse rows with the same keys
    output_df = output_df.groupby(by=[col.label for col in key_cols]).first().reset_index()  # type: ignore
    output_col_metas = key_cols + other_cols  # type: ignore
    output_table = Table.from_df_and_cols(
        columns=output_col_metas, data=output_df, ignore_extra_cols=True
    )

    # Make sure we use the objects' full histories
    stock_col = output_table.get_stock_column()
    if stock_col:
        stock_col.data = [
            stock_hash_to_merged_stock_obj_map.get(stock, stock) for stock in stock_col.data  # type: ignore
        ]
    output_table.history = copy.deepcopy(first.history) + copy.deepcopy(second.history)
    output_table.dedup_history()
    output_table.dedup_columns()
    return output_table


@tool(
    description="""
    Given a list of input tables, attempt to join the tables into
    a single table. Ideally, the tables will share a column or two that can be used
    to join (e.g. a stock column or a date column). This will create a single table
    from the multiple inputs. If you want to transform multiple tables, they must be
    merged with this first. By default, the assumption is that the tables share only
    some columns, not all of them, and we want to create a table with all the columns
    of both.
    If we are joining two tables with the same columns and just want to create a table
    which has the rows from both tables, use row_join = True, this will commonly be used
    when we are constructing a final output table for comparison of the performance
    of different stock groups/baskets.
    When the task is joining tables to have more columns in a single table, the row_join
    must be False.
""",
    category=ToolCategory.TABLE,
)
async def join_tables(args: JoinTableArgs, context: PlanRunContext) -> Union[Table, StockTable]:
    if len(args.input_tables) == 0:
        raise RuntimeError("Cannot join an empty list of tables!")
    if len(args.input_tables) == 1:
        raise RuntimeError("Cannot join a list of tables with one element!")

    _join_table_func = _join_two_tables

    # There are two cases we want to consider here. First, if all the tables
    # have the EXACT same columns, then we always want to row join no matter
    # what. Second, if the "row_join" argument is set, and all the tables share
    # columns of the same type, we also use a row join.
    all_tables_same_metadata = True
    all_tables_columns_same_types = True
    first_table_metadata = [col.metadata for col in args.input_tables[0].columns]
    first_table_col_types = [meta.col_type for meta in first_table_metadata]
    for table in args.input_tables[1:]:
        table_metadata = [col.metadata for col in table.columns]
        if table_metadata != first_table_metadata:
            all_tables_same_metadata = False
        if [meta.col_type for meta in table_metadata] != first_table_col_types:
            all_tables_columns_same_types = False

    if all_tables_same_metadata:
        _join_table_func = _join_two_tables_vertically
    elif args.row_join and all_tables_columns_same_types:
        _join_table_func = _join_two_tables_vertically

    joined_table = args.input_tables[0]
    for table in args.input_tables[1:]:
        joined_table = _join_table_func(joined_table, table)

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
and the statistical data. However, it does NOT make sense to join a table to a list
if the table was created by `get_statistic_data_for_companies` using the list, just
use the table, joining the list is entirely redundant in that situation. Joining a
table to a list is only useful when the list has extra information that the table
does not, i.e. because a tool like get_stock_recommendations or per_stock_summarize_text
has been used.
""",
    category=ToolCategory.TABLE,
)
async def join_stock_list_to_table(
    args: JoinStockListTableArgs, context: PlanRunContext
) -> StockTable:
    stock_list_copied = copy.deepcopy(args.stock_list)
    col = StockTableColumn(data=stock_list_copied)
    additional_cols = object_histories_to_columns(objects=cast(List[ComplexIOBase], col.data))
    # We need to now clear out the history of the stocks in the table, so we
    # don't create duplicate columns.
    for stock in col.data:
        stock.history = []
    new_stock_table = StockTable(columns=[col] + additional_cols)
    joined_table = _join_two_tables(first=args.input_table, second=new_stock_table)
    return StockTable(
        columns=joined_table.columns,
        history=joined_table.history,
        prefer_graph_type=joined_table.prefer_graph_type,
    )


class CreateTableStockListArgs(ToolArgs):
    stock_list: List[StockID]


@tool(
    description="""Given a list of stocks, create a table from the list that
includes the stock identifier as well as supplemental data (e.g. scores,
reasoning, summaries etc.) about each stock generated from prior steps
(e.g. get_stock_recommendations, per_stock_summarize_text,
filter_stocks_by_profile_match, get_statistics_for_companies,
transform_table). If you are outputting a list of stocks, and a user asks for a
column to be added, removed, or changed, this function MUST be called first
before the list of stocks can be treated like a table and columns can be added
or removed.
""",
    category=ToolCategory.TABLE,
)
async def create_table_from_stock_list(
    args: CreateTableStockListArgs, context: PlanRunContext
) -> StockTable:
    stock_list_copied = copy.deepcopy(args.stock_list)
    col = StockTableColumn(data=stock_list_copied)
    additional_cols = object_histories_to_columns(objects=cast(List[ComplexIOBase], col.data))
    # We need to now clear out the history of the stocks in the table, so we
    # don't create duplicate columns.
    for stock in col.data:
        stock.history = []
    return StockTable(columns=[col] + additional_cols)


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
    if not stock_column:
        raise RuntimeError("Cannot extract list of stocks, no stock column in table!")
    # Don't update in place to prevent issues in case this table is used elsewhere.
    # Use a set to prevent duplicates.
    stocks = copy.deepcopy(stock_column.data)

    if rest_columns:
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
