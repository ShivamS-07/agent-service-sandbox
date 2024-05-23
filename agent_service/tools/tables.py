import json
import logging
import os
import subprocess
import sys
import tempfile
from json.decoder import JSONDecodeError
from typing import List, Optional, Tuple

import pandas as pd
from pydantic import ValidationError

from agent_service.GPT.requests import GPT
from agent_service.io_types.table import Table, TableColumn
from agent_service.tool import ToolArgs, ToolCategory, tool
from agent_service.tools.tool_log import tool_log
from agent_service.types import PlanRunContext
from agent_service.utils.gpt_logging import GptJobIdType, GptJobType, create_gpt_context
from agent_service.utils.prompt_utils import Prompt

logger = logging.getLogger(__name__)

DATAFRAME_SCHEMA_GENERATOR_MAIN_PROMPT = Prompt(
    name="DATAFRAME_SCHEMA_GENERATOR_MAIN_PROMPT",
    template="""
You will be given a json object describing the columns of a pandas
dataframe. You should transform this json into another json with the same schema
representing a dataframe after a transformation is applied. The transformation
will be described to you, and you should produce only the json output for the
new columns. If the output columns are identical to the inputs, simply return an
empty list. Note that you should try to keep the table vertically aligned if
possible (that is, more rows than columns).

Use descriptive column names so that someone looking at the schema would know
immediately what the table has inside it. Please make sure that the column order
makes sense for a viewer as if it were being viewed in a table. For example, a
date column or a stock ID column (if they are present) should be on the left
side and could be marked as an index.

If the transformation description does not relate AT ALL to pandas or any sort
of dataframe transformation, please just return the dataframe unchanged. You
should still not output anything other than json.

It is very important that you return only an empty list if the output schema is
the same as the input.

Below is the json schema of the json you should produce. You should produce a
list of these objects, one for each column. Note that if the column type is
"stock", it means that the column contains stock identifiers.

JSON Schema:

    {schema}

col_label_is_stock_id is set to true if the *column's label itself* is a stock
identifier. NOT if the column contains stocks.

The transformation that will be applied to the dataframe is:
{transform}

Here is the json describing the input dataframe, one object per column:
{input_cols}

Please produce your json in the same format describing the columns after the
transformation has been applied. Please produce ONLY this json.

{error}
""",
)

DATAFRAME_SCHEMA_GENERATOR_SYS_PROMPT = Prompt(
    name="DATAFRAME_SCHEMA_GENERATOR_SYS_PROMPT",
    template="""
You are a quantitative data scientist who is extremely proficient at both finance and coding.
""",
).format()


DATAFRAME_TRANSFORMER_MAIN_PROMPT = Prompt(
    name="DATAFRAME_TRANSFORMER_MAIN_PROMPT",
    template="""
You will be given the description of a pandas dataframe stored in the variable
`df`, as well as a description of a transformation to apply to the
dataframe. You will produce ONLY python code that utilizes pandas to apply these
transformations to the dataframe. The resultant dataframe must also be called
`df`. The variable name must be kept the same, however the data stored in the
variable `df` should be what the user asks for. The data inside `df` can be
transformed arbitrarily to match the desired output.

You can assume the following code is already written:

    import datetime
    import math

    import numpy as np
    import pandas as pd

    df = pd.DataFrame(...)  # input dataframe

Write only the continuation for this code with no preamble or follow up text. DO
NOT INCLUDE ANY OTHER IMPORTS.

If the transformation description does not relate AT ALL to pandas or any sort
of dataframe transformation, please just return the dataframe unchanged. You
should still not output anything other than code.

The input dataframe's column schema, including the index if present is
below. Note that if a 'column' is marked as an index, it is NOT actually a
column, it is the index. For example, if the first column is a date column as is
marked as an index, then the dataframe actually has a datetime index and NOT a
column named 'Date'. Input dataframe's schema is below:
    {col_schema}

The output dataframe's desired column schema, including the index if
present. The code you write should create a dataframe with columns (and index)
that conform to this schema.
    {output_schema}

Make sure you index the column that is marked as an index. If it is a date or
datetime column, use pd.to_datetime to convert it to a DatettimeIndex.

The input dataframe's overall info: {info}

The transformation description:
    {transform}

{error}
Write your code below. Make sure you reassign the output to the same variable `df`.
Your code:

""",
)

DATAFRAME_TRANSFORMER_SYS_PROMPT = Prompt(
    name="DATAFRAME_TRANSFORMER_SYS_PROMPT",
    template="""
You are a quantitative data scientist who is extremely proficient with Python
and Pandas. You will use python and pandas to write code that applies numerical
transformations to a pandas dataframe based on the instructions given to
you. Please comment your code for each step you take, as if the person reading
it was only minimally technical. Your managers sometimes like to see your code,
so they need to understand what it's doing.
""",
).format()  # format immediately since no arguments


def _dump_cols(cols: List[TableColumn]) -> str:
    return json.dumps([col.model_dump(mode="json") for col in cols])


def _strip_code_markers(gpt_output: str, lang: str) -> str:
    if gpt_output.startswith(f"```{lang}"):
        gpt_output = gpt_output[len(f"```{lang}") :]
    if gpt_output.endswith("```"):
        gpt_output = gpt_output[:-3]

    return gpt_output


async def gen_new_column_schema(
    gpt: GPT, transformation_description: str, current_table_cols: List[TableColumn]
) -> List[TableColumn]:
    prompt = DATAFRAME_SCHEMA_GENERATOR_MAIN_PROMPT.format(
        schema=TableColumn.schema_json(),
        transform=transformation_description,
        input_cols=_dump_cols(current_table_cols),
        error="",
    )
    res = await gpt.do_chat_w_sys_prompt(
        main_prompt=prompt, sys_prompt=DATAFRAME_SCHEMA_GENERATOR_SYS_PROMPT
    )
    json_str = _strip_code_markers(res, lang="json")
    try:
        cols = json.loads(json_str)
        if not cols:
            # Empty object = unchanged
            return current_table_cols
        return [TableColumn(**item) for item in cols]
    except (ValidationError, JSONDecodeError) as e:
        prompt = DATAFRAME_SCHEMA_GENERATOR_MAIN_PROMPT.format(
            schema=TableColumn.schema_json(),
            transform=transformation_description,
            input_cols=_dump_cols(current_table_cols),
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
        return [TableColumn(**item) for item in cols]


class TransformTableArgs(ToolArgs):
    input_table: Table
    transformation_description: str


def _get_command(df: pd.DataFrame, code_file: str) -> str:
    exec_code_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "table_utils/pandas_exec.py"
    )
    # Reset the index to maintain the index's name
    serialized = df.reset_index().to_json()
    command = f"pipenv run python {exec_code_path} -d '{serialized}' -c {code_file}"

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
    with tempfile.NamedTemporaryFile(mode="w+") as f:
        f.write(code)
        f.flush()
        command: str = _get_command(df=df, code_file=f.name)
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

""",
    category=ToolCategory.TABLE,
)
async def transform_table(args: TransformTableArgs, context: PlanRunContext) -> Table:
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
            error="",
        ),
        sys_prompt=DATAFRAME_TRANSFORMER_SYS_PROMPT,
    )
    output_df, error = _run_transform_code(df=args.input_table.data, code=code)
    if output_df is None:
        logger.warning("Failed when transforming dataframe... trying again")
        code = await gpt.do_chat_w_sys_prompt(
            main_prompt=DATAFRAME_TRANSFORMER_MAIN_PROMPT.format(
                col_schema=_dump_cols(args.input_table.columns),
                output_schema=_dump_cols(new_col_schema),
                info=_get_df_info(args.input_table.data),
                transform=args.transformation_description,
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
