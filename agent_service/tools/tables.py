import copy
import json
import os
import subprocess
import sys
import tempfile
from collections import defaultdict
from itertools import chain, zip_longest
from json.decoder import JSONDecodeError
from typing import Any, List, Optional, Tuple

import pandas as pd
from pydantic import ValidationError

from agent_service.GPT.requests import GPT
from agent_service.io_type_utils import ComplexIOBase, HistoryEntry
from agent_service.io_types.stock import StockID
from agent_service.io_types.table import (
    StockTableColumn,
    Table,
    TableColumn,
    TableColumnMetadata,
    TableColumnType,
)
from agent_service.tool import ToolArgs, ToolCategory, tool
from agent_service.tools.table_utils.prompts import (
    DATAFRAME_SCHEMA_GENERATOR_MAIN_PROMPT,
    DATAFRAME_SCHEMA_GENERATOR_SYS_PROMPT,
    DATAFRAME_TRANSFORMER_MAIN_PROMPT,
    DATAFRAME_TRANSFORMER_SYS_PROMPT,
)
from agent_service.tools.tool_log import tool_log
from agent_service.types import PlanRunContext
from agent_service.utils.gpt_logging import GptJobIdType, GptJobType, create_gpt_context
from agent_service.utils.prefect import get_prefect_logger


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
    command = f"pipenv run python {exec_code_path} -d {data_file} -c {code_file}"

    if sys.platform == "darwin":
        # This is mostly for testing purposes
        helper_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "table_utils/macos_sandbox_exec_config.sb"
        )
        command = f"sandbox-exec -f {helper_path} {command}"
    elif sys.platform == "win32":
        raise RuntimeError("Windows not support, can't safely run arbitrary code")
    else:
        command = f"unshare -n -r {command}"
    return command


def _run_transform_code(df: pd.DataFrame, code: str) -> Tuple[Optional[pd.DataFrame], str]:
    if code.startswith("```python"):
        code = code[9:]
    if code.endswith("```"):
        code = code[:-3]
    with tempfile.NamedTemporaryFile(mode="w+") as code_file, tempfile.NamedTemporaryFile(
        mode="w+"
    ) as data_file:
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


@tool(
    description="""This is a function that allows you to do aribtrary transformations on Table objects.
Tables are simply wrappers around pandas dataframes. For example, if you have a
table of stock prices, and you want to compute the rolling 7-day average, you can call:

    # your_table is a Table instance wrapping a pandas dataframe of price data
    transform_table(input_table=your_table, transformation_description='Compute the rolling 7-day average')

The `transformation_description` argument is a free text description of a
transformation that will be applied to the table by an LLM, so feel free to be
detailed in your description of the desired transformation. Ideally the
transformation should be something mathematical, or a pandas operation like
group-by. Anything that could be done in pandas. Simple table formatting should
not use this. It is better to be overly detailed than not detailed enough. Note
again that the input MUST be a table, not a list!
""",
    category=ToolCategory.TABLE,
)
async def transform_table(args: TransformTableArgs, context: PlanRunContext) -> Table:
    logger = get_prefect_logger(__name__)
    input_col_metadata = [col.metadata for col in args.input_table.columns]
    gpt_context = create_gpt_context(
        GptJobType.AGENT_TOOLS, context.agent_id, GptJobIdType.AGENT_ID
    )
    gpt = GPT(context=gpt_context)
    await tool_log(log="Computing new table schema", context=context)
    new_col_schema = await gen_new_column_schema(
        gpt,
        transformation_description=args.transformation_description,
        current_table_cols=input_col_metadata,
    )
    await tool_log(log="Transforming table", context=context)
    data_df = args.input_table.to_df(stocks_as_hashables=True)
    code = await gpt.do_chat_w_sys_prompt(
        main_prompt=DATAFRAME_TRANSFORMER_MAIN_PROMPT.format(
            col_schema=_dump_cols(input_col_metadata),
            output_schema=_dump_cols(new_col_schema),
            info=_get_df_info(data_df),
            transform=args.transformation_description,
            col_type_explain=TableColumnType.get_type_explanations(),
            error="",
        ),
        sys_prompt=DATAFRAME_TRANSFORMER_SYS_PROMPT,
    )
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
                error=(
                    "Your last code failed with this error, please correct it:\n"
                    f"Last Code:\n\n{code}\n\n"
                    f"Error:\n{error}"
                ),
            ),
            sys_prompt=DATAFRAME_TRANSFORMER_SYS_PROMPT,
        )
        output_df, error = _run_transform_code(df=data_df, code=code)
        if output_df is None:
            raise RuntimeError(f"Table transformation subprocess failed with:\n{error}")

    return Table.from_df_and_cols(
        columns=new_col_schema, data=output_df, stocks_are_hashable_objs=True
    )


class JoinTableArgs(ToolArgs):
    input_tables: List[Table]


def _join_two_tables(first: Table, second: Table) -> Table:
    # Find columns to join by, ideally a date column, stock column, or both
    first_stock_col = None
    second_stock_col = None
    first_date_col = None
    second_date_col = None
    other_cols = []

    # First, find the date and stock columns for both tables if they are present.
    for col in first.columns:
        if not first_date_col and col.metadata.col_type in (
            TableColumnType.DATE,
            TableColumnType.DATETIME,
        ):
            first_date_col = col
            continue
        if not first_stock_col and col.metadata.col_type == TableColumnType.STOCK:
            first_stock_col = col
            continue
    for col in second.columns:
        if not second_date_col and col.metadata.col_type in (
            TableColumnType.DATE,
            TableColumnType.DATETIME,
        ):
            second_date_col = col
            continue
        if not second_stock_col and col.metadata.col_type == TableColumnType.STOCK:
            second_stock_col = col
            continue

    for col in first.columns + second.columns:
        # Collect up all the columns that won't be joined on
        if (
            (first_stock_col and col.metadata == first_stock_col.metadata)
            or (first_date_col and col.metadata == first_date_col.metadata)
            or (second_stock_col and col.metadata == second_stock_col.metadata)
            or (second_date_col and col.metadata == second_date_col.metadata)
        ):
            continue
        other_cols.append(col.metadata)

    # Go case by case:
    #   1. Join on stocks AND dates
    #   2. Join on just stocks
    #   3. Join on just dates

    if first_stock_col and second_stock_col and first_date_col and second_date_col:
        join_col_meta = [first_date_col.metadata, first_stock_col.metadata]
        first_dict = first.to_dict(key_cols=[first_date_col.metadata, first_stock_col.metadata])
        second_dict = second.to_dict(key_cols=[second_date_col.metadata, second_stock_col.metadata])
    elif first_stock_col and second_stock_col:
        join_col_meta = [first_stock_col.metadata]
        first_dict = first.to_dict(key_cols=[first_stock_col.metadata])
        second_dict = second.to_dict(key_cols=[second_stock_col.metadata])
    elif first_date_col and second_date_col:
        join_col_meta = [first_date_col.metadata]
        first_dict = first.to_dict(key_cols=[first_date_col.metadata])
        second_dict = second.to_dict(key_cols=[second_date_col.metadata])
    else:
        # Can't join on anything! Just concat
        return Table(columns=first.columns + second.columns)

    # At this point, we have two dictionaries mapping the "indexed" values being
    # joined on to the row data. # Doesn't need to be fast, doing it in a simple
    # way. Create a list for each output column, and iteratively append data to
    # the list.
    col_data_list: List[List[Any]] = [[] for _ in chain(join_col_meta, other_cols)]

    # We need this in the case where two *different* objects have the same hash
    # value. In that case, we'll want to be able to merge them. E.g. StockID's
    # are hashed by GBI ID, but we need to merge their individual histories.
    index_val_to_object_map = defaultdict(list)
    for key in chain(first_dict.keys(), second_dict.keys()):
        if not isinstance(key, tuple):
            key = (key,)
        for component in key:
            index_val_to_object_map[component].append(component)

    for key in sorted(set(first_dict.keys()).union(second_dict.keys())):
        # The data in the first and second tables associated with this key
        # (e.g. a StockID)
        first_data = first_dict.get(key, [])
        second_data = second_dict.get(key, [])
        key_tup = (key,) if not isinstance(key, tuple) else key
        new_key_component_list = []
        # For some tables, key here is (date, stock). Otherwise it could be
        # (date) or (stock)
        for key_component in key_tup:
            key_objects = index_val_to_object_map[key_component]
            # We need to merge all the key objects together into a single key
            new_key_component = key_objects[0]
            # key_objects is a list of distinct objects that hash to the same
            # value. E.g. two stocks with the same GBI ID but different
            # histories.
            for obj in key_objects[1:]:
                if isinstance(obj, ComplexIOBase) and isinstance(new_key_component, ComplexIOBase):
                    new_key_component = new_key_component.union_history_with(obj)
            new_key_component_list.append(new_key_component)
        key_tup = tuple(new_key_component_list)

        # zip_longest will fill in "None" if one list is bigger than the
        # other. E.g. in the case where a row exists in the first table but not
        # in the second, we want to fill in None for the columns only in table 2.
        for val, col_data in zip_longest(chain(key_tup, first_data, second_data), col_data_list):
            col_data.append(val)

    output_cols = []
    for meta, data in zip(chain(join_col_meta, other_cols), col_data_list):
        output_cols.append(TableColumn(metadata=meta, data=data))

    return Table(columns=output_cols)


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

    return joined_table


class GetStockListFromTableArgs(ToolArgs):
    input_table: Table


@tool(
    description="""
    Given a table with at least one column of stocks, extract
    that column into a list of stock ID's.  This is very useful for e.g. filtering
    on some numerical data in a table before extracting the stock list and fetching
    other data with it.
    The tool can be used to convert a table with a stock column from another tool "
    "like get_portfolio_holdings into a list of stock ID's.
    This function can only be used with actual tables, it cannot be used with either
lists of texts or lists of stocks
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
    return list(set(stocks))
