import json
import os
import subprocess
import sys
import tempfile
from json.decoder import JSONDecodeError
from typing import List, Optional, Tuple

import pandas as pd
from pydantic import ValidationError

from agent_service.GPT.requests import GPT
from agent_service.io_types.table import (
    STOCK_ID_COL_NAME_DEFAULT,
    Table,
    TableColumn,
    TableColumnType,
)
from agent_service.io_types.text import StockAlignedTextGroups, Text
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


def _dump_cols(cols: List[TableColumn]) -> str:
    return json.dumps([col.model_dump(mode="json") for col in cols])


def _strip_code_markers(gpt_output: str, lang: str) -> str:
    if gpt_output.startswith(f"```{lang}"):
        gpt_output = gpt_output[len(f"```{lang}") :]
    if gpt_output.endswith("```"):
        gpt_output = gpt_output[:-3]

    return gpt_output.strip()


async def gen_new_column_schema(
    gpt: GPT, transformation_description: str, current_table_cols: List[TableColumn]
) -> List[TableColumn]:
    logger = get_prefect_logger(__name__)
    prompt = DATAFRAME_SCHEMA_GENERATOR_MAIN_PROMPT.format(
        schema=TableColumn.schema_json(),
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
        return [TableColumn.model_validate(item) for item in cols]
    except (ValidationError, JSONDecodeError) as e:
        prompt = DATAFRAME_SCHEMA_GENERATOR_MAIN_PROMPT.format(
            schema=TableColumn.schema_json(),
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
        return [TableColumn.model_validate(item) for item in cols]


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
    Index: {df.index}
    """


@tool(
    description="""This is a function that allows you to do aribtrary transformations on Table objects.
Tables are simply wrappers around pandas dataframes. For example, if you have a
table of stock prices, and you want to compute the rolling 7-day average, you can call:

    # your_table is a Table instance wrapping a pandas dataframe of price data
    transform_table(input_table=your_table, transformation_description='Compute the rolling 7-day average')

The `transformation_description` argument is a free text description of a
transformation that will be applied to the table by an LLM, so feel free to be
detailed in your description of the desired transformation. It can include
anything from mathematical operations to formatting, etc. Anything that could be
done in pandas. It is better to be overly detailed than not detailed enough.
Note again that the input MUST be a table, not a list!
""",
    category=ToolCategory.TABLE,
)
async def transform_table(args: TransformTableArgs, context: PlanRunContext) -> Table:
    logger = get_prefect_logger(__name__)
    gpt_context = create_gpt_context(
        GptJobType.AGENT_TOOLS, context.agent_id, GptJobIdType.AGENT_ID
    )
    gpt = GPT(context=gpt_context)
    await tool_log(log="Computing new table schema", context=context)
    new_col_schema = await gen_new_column_schema(
        gpt,
        transformation_description=args.transformation_description,
        current_table_cols=args.input_table.columns,
    )
    await tool_log(log="Transforming table", context=context)
    code = await gpt.do_chat_w_sys_prompt(
        main_prompt=DATAFRAME_TRANSFORMER_MAIN_PROMPT.format(
            col_schema=_dump_cols(args.input_table.columns),
            output_schema=_dump_cols(new_col_schema),
            info=_get_df_info(args.input_table.data),
            transform=args.transformation_description,
            col_type_explain=TableColumnType.get_type_explanations(),
            error="",
        ),
        sys_prompt=DATAFRAME_TRANSFORMER_SYS_PROMPT,
    )
    logger.info(f"Running transform code:\n{code}")
    output_df, error = _run_transform_code(df=args.input_table.data, code=code)
    if output_df is None:
        logger.warning("Failed when transforming dataframe... trying again")
        logger.warning(f"Failing code:\n{code}")
        code = await gpt.do_chat_w_sys_prompt(
            main_prompt=DATAFRAME_TRANSFORMER_MAIN_PROMPT.format(
                col_schema=_dump_cols(args.input_table.columns),
                output_schema=_dump_cols(new_col_schema),
                info=_get_df_info(args.input_table.data),
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
        output_df, error = _run_transform_code(df=args.input_table.data, code=code)
        if output_df is None:
            raise RuntimeError(f"Table transformation subprocess failed with:\n{error}")

    return Table(columns=new_col_schema, data=output_df)


class JoinTableArgs(ToolArgs):
    input_tables: List[Table]


def _join_two_tables(first: Table, second: Table) -> Table:
    # Find columns to join by, ideally a date column, stock column, or both
    first_stock_cols = None
    second_stock_cols = None
    first_date_cols = None
    second_date_cols = None
    other_cols = []

    # First, find the date and stock columns for both dataframes if they are present.
    for col, df_col in zip(first.columns, first.data.columns):
        if not first_date_cols and col.col_type in (TableColumnType.DATE, TableColumnType.DATETIME):
            first_date_cols = (col, df_col)
            continue
        if not first_stock_cols and col.col_type == TableColumnType.STOCK:
            first_stock_cols = (col, df_col)
            continue
    for col, df_col in zip(second.columns, second.data.columns):
        if not second_date_cols and col.col_type in (
            TableColumnType.DATE,
            TableColumnType.DATETIME,
        ):
            second_date_cols = (col, df_col)
            continue
        if not second_stock_cols and col.col_type == TableColumnType.STOCK:
            second_stock_cols = (col, df_col)
            continue

    for col in first.columns + second.columns:
        # Collect up all the columns that won't be joined on
        if (
            (first_stock_cols and col == first_stock_cols[0])
            or (first_date_cols and col == first_date_cols[0])
            or (second_stock_cols and col == second_stock_cols[0])
            or (second_date_cols and col == second_date_cols[0])
        ):
            continue
        other_cols.append(col)

    # Ideally we'd never need these suffixes, but included just in case so we
    # don't show anything crazy
    join_suffixes = (" (one)", " (two)")

    # Go case by case:
    #   1. Join on stocks AND dates
    #   2. Join on just stocks
    #   3. Join on just dates
    if first_stock_cols and second_stock_cols and first_date_cols and second_date_cols:
        output_cols = [first_date_cols[0], first_stock_cols[0]] + other_cols
        df = pd.merge(
            left=first.data,
            right=second.data,
            how="outer",
            left_on=[first_date_cols[1], first_stock_cols[1]],
            right_on=[second_date_cols[1], second_stock_cols[1]],
            suffixes=join_suffixes,
        )
    elif first_stock_cols and second_stock_cols:
        output_cols = [first_stock_cols[0]] + other_cols
        df = pd.merge(
            left=first.data,
            right=second.data,
            how="outer",
            left_on=first_stock_cols[1],
            right_on=second_stock_cols[1],
            suffixes=join_suffixes,
        )
    elif first_date_cols and second_date_cols:
        output_cols = [first_date_cols[0]] + other_cols
        df = pd.merge(
            left=first.data,
            right=second.data,
            how="outer",
            left_on=first_date_cols[1],
            right_on=second_date_cols[1],
            suffixes=join_suffixes,
        )
    else:
        # Can't join on anything! Just concat
        output_cols = first.columns + second.columns
        df = pd.concat([first.data, second.data], axis=1)
    return Table(
        columns=output_cols,
        data=df,
    )


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


class CreateStockTextGroupTableArgs(ToolArgs):
    stock_text_group: StockAlignedTextGroups


@tool(
    description="""Given a StockAlignedTextGroups object, create a table with a
stock column and columns for each associated text. This should be used before
joining a StockAlignedTextGroups object with other tables.
""",
    category=ToolCategory.TABLE,
)
async def create_table_from_stock_text_groups(
    args: CreateStockTextGroupTableArgs, context: PlanRunContext
) -> Table:
    data = [
        [gbi, Text.get_all_strs(text_group)]
        for gbi, text_group in args.stock_text_group.val.items()
    ]
    df = pd.DataFrame(data=data, columns=[STOCK_ID_COL_NAME_DEFAULT, "Text"])
    columns = [
        TableColumn(label=STOCK_ID_COL_NAME_DEFAULT, col_type=TableColumnType.STOCK),
        TableColumn(label="Text", col_type=TableColumnType.STRING),
    ]
    return Table(columns=columns, data=df)


class GetStockListFromTableArgs(ToolArgs):
    input_table: Table


@tool(
    description="""Given a table with at least one column of stocks, extract
that column into a list of stock ID's.  This is very useful for e.g. filtering
on some numerical data in a table before extracting the stock list and fetching
other data with it.
""",
    category=ToolCategory.TABLE,
)
async def get_stock_identifier_list_from_table(
    args: GetStockListFromTableArgs, context: PlanRunContext
) -> List[int]:
    for col, df_col in zip(args.input_table.columns, args.input_table.data.columns):
        if col.col_type == TableColumnType.STOCK:
            return args.input_table.data[df_col].unique().tolist()

    raise RuntimeError("Cannot extract list of stocks, no stock column in table!")
