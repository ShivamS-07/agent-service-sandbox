import copy
import datetime
import inspect
import json
import os
import subprocess
import sys
import tempfile
import traceback
from collections import defaultdict
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
    TableColumn,
    TableColumnMetadata,
    TableColumnType,
    object_histories_to_columns,
)
from agent_service.planner.errors import EmptyOutputError
from agent_service.tool import (
    TOOL_DEBUG_INFO,
    ToolArgMetadata,
    ToolArgs,
    ToolCategory,
    tool,
)
from agent_service.tools.stock_groups.utils import (
    add_stock_group_column,
    get_stock_group_input_tables,
    remove_stock_group_columns,
)
from agent_service.tools.table_utils.join_utils import (
    add_extra_group_cols,
    check_for_index_overlap,
    expand_dates_across_tables,
    preprocess_heterogeneous_tables_before_joining,
)
from agent_service.tools.table_utils.prompts import (
    DATAFRAME_SCHEMA_GENERATOR_MAIN_PROMPT,
    DATAFRAME_SCHEMA_GENERATOR_SYS_PROMPT,
    DATAFRAME_TRANSFORMER_MAIN_PROMPT,
    PICK_GPT_MAIN_PROMPT,
    TABLE_ADD_DIFF_MAIN_PROMPT,
    TABLE_REMOVE_DIFF_MAIN_PROMPT,
    UPDATE_DATAFRAME_TRANSFORMER_MAIN_PROMPT,
)
from agent_service.tools.tool_log import tool_log
from agent_service.types import PlanRunContext
from agent_service.utils.async_db import get_async_db
from agent_service.utils.async_utils import gather_with_concurrency
from agent_service.utils.date_utils import get_now_utc
from agent_service.utils.gpt_logging import GptJobIdType, GptJobType, create_gpt_context
from agent_service.utils.pagerduty import pager_wrapper
from agent_service.utils.prefect import get_prefect_logger
from agent_service.utils.prompt_utils import FilledPrompt, Prompt
from agent_service.utils.tool_diff import get_prev_run_info


def _dump_cols(cols: List[TableColumnMetadata]) -> str:
    return json.dumps([col.to_json_dict_for_gpt() for col in cols])


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

    # Hidden args
    no_cache: bool = False
    transform_code: Optional[str] = None
    target_schema: Optional[List[TableColumnMetadata]] = None
    template_task_id: Optional[str] = None
    arg_metadata = {
        "no_cache": ToolArgMetadata(hidden_from_planner=True),
        "transform_code": ToolArgMetadata(hidden_from_planner=True),
        "target_schema": ToolArgMetadata(hidden_from_planner=True),
        "template_task_id": ToolArgMetadata(hidden_from_planner=True),
    }


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


def cleanup_generated_python(code: str) -> str:
    code = code.strip()
    if code.endswith("```"):
        code = code[:-3]

    if code.startswith("```python"):
        code = code[9:]
        lines = code.splitlines()
        found_back_quote = False
        # GPT sometimes (always?) wraps code in ```python ... ```
        # additionally it sometimes writes an explanation after the last ```
        # backquote is not a valid python char so we can remove it
        # and convert anything after that to a comment
        for i, line in enumerate(lines):
            if found_back_quote:
                lines[i] = "# " + line
            elif line.strip().startswith("```"):
                # this can also false positive inside of a """ block
                found_back_quote = True
                lines[i] = ""
        code = "\n".join(lines)

    if "import" in code or "from" in code:
        # sometimes GPT imports pandas even though we told it not to/not needed
        # we can safely comment out these lines because if it is just pandas,
        # we already imported it
        # if it is some random module, we would have failed it before in pandas_exec
        # it will now fail in a different spot when we try to exec() it because code
        # using the imported module will fail
        lines = code.splitlines()
        for i, line in enumerate(lines):
            # technically this could find a false positive inside of a """ block
            # but our checks in pandas_exec would have the same problem
            if "import" in line and (line.startswith("from ") or line.startswith("import ")):
                lines[i] = "# " + line

        code = "\n".join(lines)

    return code


def _run_transform_code(df: pd.DataFrame, code: str) -> Tuple[Optional[pd.DataFrame], str]:
    code = cleanup_generated_python(code)
    delete = (
        sys.platform != "win32"
    )  # turning off delete is necessary for temp file to work in Windows
    with (
        tempfile.NamedTemporaryFile(mode="w+", delete=delete) as code_file,
        tempfile.NamedTemporaryFile(mode="w+", delete=delete) as data_file,
    ):
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

If you are doing a calculation that requires you to distinguish individual or groups of rows of your
table, for instance correlating the stock performance of some group of stocks with another stock or
group of stocks, note that this tool cannot distinguish rows of the table unless you create the data
you need to distinguish separately and then join it together with the join_table tool in row_join mode
with explict table_names that create an extra column, 'Group' that can be filtered over. It is not
possible to filter any securities directly in the context of the transform table tool. After you do the
table join, you should call this tool, stating that explicitly in your transformation description that
you must filter the using the 'Group' column. For example, if you were asked to create a table which
correlates the performance of TSX 60 stocks with S&P 500 stocks, you would first calculate TSX 60 and
S&P 500 performance in separate tables, then call table join with table_names = ["TSX 60", "S&P 500"],
then use the transformation description "calculate the correlation between TSX 60 and S&P 500
stocks using the provided table with statistics for both stock index. You should get the TSX 60 data
by filtering the 'Group' column for 'TSX 60', and get the S&P 500 performance by filtering the Group
Column 'S&P 500'". Whenever you have included 'table_names' in a table_join, and then need to do a
calculation using those distinctions, it is critical that you provide the exact table_name strings in
the transformation_description using single quotes (in this case, 'TSX 60' and 'S&P 500'), otherwise
the transform_table tool will be unable to filter the stocks, your calculation will fail, and you will
be fired.

Note that if you are filtering on the results of a correlation and need to extract specific stocks
using a later call to get_stock_identifier_list_from_table, you must also include in your transformation
an explicit instruction (e.g. 'Output the TSX stocks as the first column') to put the column you need
to extract stocks from first in your table, otherwise you will extract the wrong stocks!

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
If you are filtering stocks, the transformation description must begin with the word `Filter`.
If you only need the stock ids from a table, then after using this tool, you should use the
`get_stock_identifier_list_from_table` tool to extract the stock ids from the table. DO NOT pass
the table directly to another tool that expects a list of stocks.
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
    old_description = None
    old_date = None
    try:  # since everything here is optional, put in try/except
        prev_run_info = await get_prev_run_info(context, "transform_table")
        if prev_run_info is None and args.template_task_id:
            template_context = copy.deepcopy(context)
            template_context.task_id = args.template_task_id
            # Turn off plan run filter for subplanner case (is same plan run)
            prev_run_info = await get_prev_run_info(
                template_context, "transform_table", plan_run_filter=False
            )
        if prev_run_info is not None and not args.no_cache:
            prev_args = TransformTableArgs.model_validate_json(prev_run_info.inputs_str)
            old_description = prev_args.transformation_description

            prev_output = prev_run_info.output  # type:ignore
            prev_other: Dict[str, str] = prev_run_info.debug  # type:ignore
            if prev_other:
                old_code = (
                    prev_other["code_second_attempt"]
                    if "code_second_attempt" in prev_other
                    else prev_other["code_first_attempt"]
                )
                old_schema = load_io_type(prev_other["table_schema"])  # type:ignore
                old_date = prev_other["date"] if "date" in prev_other else None

    except Exception as e:
        logger.exception(f"Error creating diff info from previous run: {e}")
        pager_wrapper(
            current_frame=inspect.currentframe(),
            module_name=__name__,
            context=context,
            e=e,
            classt="AgentUpdateError",
            summary="Failed to get previous run info or getting default stock list",
        )

    debug_info: Dict[str, Any] = {}
    TOOL_DEBUG_INFO.set(debug_info)

    list_of_one_table = await transform_tables_helper(
        args.transformation_description,
        [args.input_table],
        context,
        old_code=old_code,
        old_schema=old_schema,
        old_description=old_description,
        old_date=old_date,
        debug_info=debug_info,
        manual_code=args.transform_code,
        manual_new_col_schema=args.target_schema,
    )
    output_table = list_of_one_table[0]

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

    # special case where we are doing filters over multiple dates
    # e.g. (filter to stocks with gains in each of the last 5 years)
    # And want to include all the relevant data in the output table.

    if args.transformation_description.startswith("Filter to stocks"):
        stock_column = output_table.get_stock_column()
        date_column = output_table.get_date_column()
        if (
            stock_column
            and date_column
            and len(output_table.columns) == 3
            and len(set(stock_column.data)) != len(stock_column.data)
        ):
            other_column = output_table.columns[-1]
            for other_column in output_table.columns:  # should be last column, but just in case
                if other_column != date_column and other_column != stock_column:
                    break

            stat_lookup: Dict[StockID, Dict[str, float]] = defaultdict(dict)
            for i in range(len(stock_column.data)):
                stat_lookup[stock_column.data[i]][date_column.data[i]] = other_column.data[i]  # type: ignore
            new_stock_column = StockTableColumn(
                title=stock_column.title,
                history=stock_column.history,
                data=cast(List[StockID], list(set(stock_column.data))),
            )
            new_columns: List[TableColumn] = [new_stock_column]
            for date in set(date_column.data):
                new_column = copy.deepcopy(other_column)
                new_column.metadata.label = f"{new_column.metadata.label}, {date}"
                new_column.data = []
                for stock in new_stock_column.data:
                    if date in stat_lookup[stock]:
                        new_column.data.append(stat_lookup[stock][date])  # type: ignore
                    else:
                        new_column.data.append(None)
                new_columns.append(new_column)
            output_table = StockTable(columns=new_columns)

    await tool_log(
        log=f"Transformed table has {len(output_table.columns[0].data)} rows", context=context
    )

    return output_table


async def transform_tables_helper(
    description: str,
    tables: List[Table],
    context: PlanRunContext,
    old_code: Optional[str] = None,
    old_schema: Optional[List[TableColumnMetadata]] = None,
    old_description: Optional[str] = None,
    old_date: Optional[str] = None,
    debug_info: Dict[str, Any] = {},
    # Manually passed in code, overrides everything
    manual_code: Optional[str] = None,
    manual_new_col_schema: Optional[List[TableColumnMetadata]] = None,
) -> List[Table]:
    logger = get_prefect_logger(__name__)

    input_col_metadata = [col.metadata for col in tables[0].columns]
    labels = [metadata.label for metadata in input_col_metadata]
    gpt_context = create_gpt_context(
        GptJobType.AGENT_TOOLS, context.agent_id, GptJobIdType.AGENT_ID
    )

    medium_gpt = GPT(model=GPT4_O, context=gpt_context)

    today = get_now_utc().date()

    data_dfs = [input_table.to_df(stocks_as_hashables=True) for input_table in tables]

    if manual_code and manual_new_col_schema:
        code = manual_code
        new_col_schema = manual_new_col_schema
    elif not old_code or not old_schema:
        old_gpt = GPT(context=gpt_context)  # keep using turbo for the schema generation
        await tool_log(log="Computing new table schema", context=context)
        new_col_schema = await gen_new_column_schema(
            old_gpt,
            transformation_description=description,
            current_table_cols=input_col_metadata,
        )

        logger.info(f"Table Schema: {new_col_schema}")
        await tool_log(log="Planning table transformation", context=context)

        challenge = await medium_gpt.do_chat_w_sys_prompt(
            PICK_GPT_MAIN_PROMPT.format(task=description, labels=labels), NO_PROMPT
        )

        if challenge.lower().strip() == "easy":
            gpt = medium_gpt
        else:
            gpt = GPT(model=O1, context=gpt_context)
            await tool_log("Brainstorming in advance of calculation", context)

        code = await gpt.do_chat_w_sys_prompt(
            main_prompt=DATAFRAME_TRANSFORMER_MAIN_PROMPT.format(
                col_schema=_dump_cols(input_col_metadata),
                output_schema=_dump_cols(new_col_schema),
                info=_get_df_info(data_dfs[0]),
                transform=description,
                col_type_explain=TableColumnType.get_type_explanations(),
                today=today,
                error="",
            ),
            temperature=1.0,
            # sys_prompt=DATAFRAME_TRANSFORMER_SYS_PROMPT,
            sys_prompt=NO_PROMPT,
        )
        code = cleanup_generated_python(code)

    else:
        gpt = medium_gpt
        await tool_log(log="Using table schema from previous run", context=context)
        new_col_schema = old_schema
        main_prompt = UPDATE_DATAFRAME_TRANSFORMER_MAIN_PROMPT.format(
            old_code=old_code,
            description=description,
            old_description=old_description,
            date=today.isoformat(),
            old_date=old_date,
        )
        result = await gpt.do_chat_w_sys_prompt(main_prompt, NO_PROMPT)
        if result.strip().lower() != "no change":
            await tool_log(
                log="Updating table transformation plan from previous run", context=context
            )
            code = cleanup_generated_python(result)
        else:
            await tool_log(
                log="Used same table transformation plan as previous run", context=context
            )
            code = old_code

    debug_info["table_schema"] = dump_io_type(new_col_schema)
    debug_info["date"] = today.isoformat()
    debug_info["code_first_attempt"] = code
    logger.info(f"Running transform code:\n{code}")
    output_dfs = []
    had_error = False
    error = None
    await tool_log(log="Executing transformation", context=context)
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
            Table.from_df_and_cols(
                columns=new_col_schema, data=output_df, stocks_are_hashable_objs=True
            )
        except Exception:
            error = traceback.format_exc()
            had_error = True
            break

        output_dfs.append(output_df)

    if had_error:
        gpt = medium_gpt
        debug_info["code_first_attempt_error"] = error
        output_dfs = []
        logger.warning("Failed when transforming dataframe... trying again")
        logger.warning("first attempt failed with error:")
        logger.warning(error)
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
            ),
            # sys_prompt=DATAFRAME_TRANSFORMER_SYS_PROMPT,
            sys_prompt=NO_PROMPT,
            temperature=1.0,
        )

        code = cleanup_generated_python(code)
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
                    pd.to_datetime(output_df[col.label].replace(" ", "", regex=True))
                    .dt.to_period("Q")
                    .astype(str)
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
    The output of this tool is always a table. Make sure you always label the output variable of this
    tool as a table, e.g. sector_performance_table.
""",
    category=ToolCategory.STOCK_GROUPS,
)
async def per_stock_group_transform_table(
    args: PerStockGroupTransformTableInput, context: PlanRunContext
) -> Union[Table, StockTable]:
    logger = get_prefect_logger(__name__)
    old_schema: Optional[List[TableColumnMetadata]] = None
    old_code: Optional[str] = None
    old_description = None
    old_date = None
    # TODO: Consider doing stock list diffing for stock group filtering
    # prev_args = None
    # prev_output = None
    try:  # since everything here is optional, put in try/except
        prev_run_info = await get_prev_run_info(context, "per_stock_group transform_table")
        if prev_run_info is not None:
            prev_args = TransformTableArgs.model_validate_json(prev_run_info.inputs_str)
            # prev_output = prev_run_info.output  # type:ignore
            old_description = prev_args.transformation_description
            prev_other: Dict[str, str] = prev_run_info.debug  # type:ignore
            if prev_other:
                old_code = (
                    prev_other["code_second_attempt"]
                    if "code_second_attempt" in prev_other
                    else prev_other["code_first_attempt"]
                )
                old_date = prev_other["date"] if "date" in prev_other else None
                old_schema = load_io_type(prev_other["table_schema"])  # type:ignore

    except Exception as e:
        logger.exception(f"Error creating diff info from previous run: {e}")
        pager_wrapper(
            current_frame=inspect.currentframe(),
            module_name=__name__,
            context=context,
            e=e,
            classt="AgentUpdateError",
            summary="Failed to get previous run info or getting default stock list",
        )

    debug_info: Dict[str, Any] = {}
    TOOL_DEBUG_INFO.set(debug_info)

    stock_group_input_tables = get_stock_group_input_tables(args.input_table, args.stock_groups)

    stock_group_output_tables = await transform_tables_helper(
        args.transformation_description,
        stock_group_input_tables,  # type:ignore
        context,
        old_code=old_code,
        old_schema=old_schema,
        old_description=old_description,
        old_date=old_date,
        debug_info=debug_info,
    )

    for table, group in zip(stock_group_output_tables, args.stock_groups.stock_groups):
        remove_stock_group_columns(table, args.stock_groups.header)
        add_stock_group_column(table, args.stock_groups.header, group.name)

    joined_table = stock_group_output_tables[0]
    for table in stock_group_output_tables[1:]:
        joined_table = _join_two_tables_vertically(joined_table, table, add_group_col=False)

    await tool_log(
        log=f"Transformed table has {len(joined_table.columns[0].data)} rows", context=context
    )

    if joined_table.get_stock_column():
        return StockTable(columns=joined_table.columns)

    return joined_table


class JoinTableArgs(ToolArgs):
    input_tables: List[Table]
    # If true, join vertically instead of horizontally, resulting in more rows
    # instead of more columns.
    row_join: bool = False
    table_names: Optional[List[str]] = None
    add_columns: bool = False


def _join_two_tables_vertically(
    first: Table,
    second: Table,
    table_names: Optional[List[str]] = None,
    idx: int = 0,
    add_group_col: bool = True,
    first_table_is_primary: bool = False,
) -> Table:
    if add_group_col and (table_names or check_for_index_overlap(first, second)):
        if idx == 0:
            first_name = table_names[0] if table_names else f"Group {idx + 1}"
            first = add_extra_group_cols(first, first_name)
        second_name = (
            table_names[idx + 1]
            if table_names and len(table_names) > idx + 1
            else f"Group {idx + 2}"
        )
        second = add_extra_group_cols(second, second_name)

    output_cols = []
    for col1, col2 in zip(first.columns, second.columns):
        if col1.metadata.col_type == col2.metadata.col_type:
            new_col = copy.deepcopy(col1)
            new_col.data.extend(col2.data)
            new_col.union_history_with(col2)
            # Cell citations must also be handled, map them to the new row
            col_len = len(new_col.data)
            for row_num, citations in col2.metadata.cell_citations.items():
                new_col.metadata.cell_citations[row_num + col_len] = citations
            if new_col.metadata.row_descs and col2.metadata.row_descs:
                for row_num, descs in col2.metadata.row_descs.items():
                    new_col.metadata.row_descs[row_num + col_len] = descs
            output_cols.append(new_col)

    output_table = Table(columns=output_cols, prefer_graph_type=first.prefer_graph_type)
    output_table.history = copy.deepcopy(first.history) + copy.deepcopy(second.history)
    output_table.dedup_history()
    return output_table


def _join_two_tables(
    first: Table,
    second: Table,
    table_names: Optional[List[str]] = None,
    idx: int = 0,
    add_group_col: bool = True,
    first_table_is_primary: bool = False,
) -> Table:
    first = copy.deepcopy(first)
    second = copy.deepcopy(second)

    first, second = preprocess_heterogeneous_tables_before_joining(first, second)
    first, second = expand_dates_across_tables(first, second)

    # After preprocessing, check again to see if we should join the tables
    # vertically. There may be cases where preprocessing allows this.
    first_table_metadata = [col.metadata for col in first.columns]
    second_table_metadata = [col.metadata for col in second.columns]

    if first_table_metadata == second_table_metadata:
        return _join_two_tables_vertically(first, second, add_group_col=False)

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
        how="left" if first_table_is_primary else "outer",
        suffixes=(None, "_ignored"),
    )

    # Collapse rows with the same keys
    output_df = (
        output_df.groupby(by=[col.label for col in key_cols], sort=False).first().reset_index()  # type: ignore
    )
    output_col_metas = key_cols + other_cols  # type: ignore
    output_table = Table.from_df_and_cols(
        columns=output_col_metas, data=output_df, ignore_extra_cols=True
    )

    # Make sure we use the objects' full histories
    stock_col = output_table.get_stock_column()
    if stock_col:
        stock_col.data = [
            stock_hash_to_merged_stock_obj_map.get(stock, stock)  # type: ignore
            for stock in stock_col.data
        ]
    output_table.history = copy.deepcopy(first.history) + copy.deepcopy(second.history)
    output_table.dedup_history()
    output_table.dedup_columns()
    return output_table


async def _modify_join_args_smartly_with_gpt(
    context: PlanRunContext, args: JoinTableArgs
) -> JoinTableArgs:
    """
    Use GPT to 'smartly' modify the join args. Specifically used for row_join for now.
    """
    if not args.input_tables:
        return args

    if args.table_names and args.row_join:
        # For now don't handle this case, assume it works
        return args

    gpt_context = create_gpt_context(
        GptJobType.AGENT_TOOLS, context.agent_id, GptJobIdType.AGENT_ID
    )
    gpt = GPT(context=gpt_context)
    prev_run_info = await get_prev_run_info(context=context, tool_name="join_tables")
    if prev_run_info and json.loads(prev_run_info.debug.get("should_use_row_join", "false")):
        args.row_join = True
        return args

    _, plan = await get_async_db().get_execution_plan_for_run(context.plan_run_id)
    rest_of_plan = plan.get_plan_after_task(task_id=context.task_id) if context.task_id else plan

    table_schemas = []
    for i, table in enumerate(args.input_tables):
        metadatas = [col.metadata for col in table.columns]
        table_schema_str = ", ".join((f"'{meta.label}' ({meta.col_type})" for meta in metadatas))
        table_schemas.append(f"Table {i}: {table_schema_str}")

    schema_str = "\n\n".join(table_schemas)

    prompt = Prompt(
        name="SET_JOIN_TABLE_ARGS_PROMPT",
        template="""
        You are a financial analyst. You wrote an automation script in python
        for your client, and now you're tweaking one of the function calls to
        make it work correctly. You're calling a function called `join_tables`
        that takes some tables of data and joins them together. You must set the
        value of the one of the arguments to this task: `row_join`.

        If we are joining tables with the same or similar columns and just want
        to create a table which has the rows from all tables, use row_join =
        True.  This will commonly be used when we are constructing a final
        output table for comparison of the performance of different stock
        groups/baskets.  When the task is joining tables to have more columns in
        a single table, the row_join must be False. row_join=True will
        essentially stack the tables on top of each other, and row_join=False
        will horizontally join, creating columns for each input table. If data
        column names are different, ONLY set row_join to true if you're very
        confident that the two columns represent the same thing (e.g. "Stock
        Price" and "Price of Stock", etc).

        To help you decide, here are the rest of the steps in your script:

        {rest_of_plan}

        And here are the column schemas of the tables:
        {schema_str}

        Some additional things to be aware of:
        1. If you are asked to graph multiple datasets, make sure that either
           the datasets are distinguished with some sort of 'dataset' column or
           that they are kept as separate columns (row_join=False). Otherwise,
           the graph won't be able to distinguish between the datasets.

        Please output your value for row_join as a single boolean 'true' or
        'false' and NOTHING else.
        """,
    ).format(rest_of_plan=rest_of_plan.get_formatted_plan(numbered=True), schema_str=schema_str)

    response = await gpt.do_chat_w_sys_prompt(
        main_prompt=prompt, sys_prompt=FilledPrompt(filled_prompt="")
    )
    response = response.lower().strip()
    if response == "true":
        args.row_join = True
    else:
        args.row_join = False

    debug_info: Dict[str, Any] = {"should_use_row_join": args.row_join}
    TOOL_DEBUG_INFO.set(debug_info)

    return args


@tool(
    description="""
    Given a list of input tables, attempt to join the tables into
    a single table. Ideally, the tables will share a column or two that can be used
    to join (e.g. a stock column or a date column). This will create a single table
    from the multiple inputs. If you want to transform multiple tables, they must be
    merged with this first. By default, the assumption is that the tables share only
    some columns, not all of them, and we want to create a table with all the columns
    of both.
    If we are joining tables with the same columns and just want to create a table
    which has the rows from all tables, use row_join = True, this will commonly be used
    when we are constructing a final output table for comparison of the performance
    of different stock groups/baskets.
    When the task is joining tables to have more columns in a single table, the row_join
    must be False.
    If we are joining tables where we need to distinguish the data from the input tables in the
    output in a way that wouldn't be represented in the column names (for instance, creating a
    bar chart which compares the sector allocation of stock indexes), you should include
    names of the input tables as an aligned list of strings in the optional argument
    table_names (e.g. ["S&P 500", "TSX"]). You do NOT need to do this when you are just
    directly combining the output of the company statistics tool.
    Always output the add_column argument even though it has a default value.
    The add_columns argument must be set to True whenever the client mentions adding/including
    column to an existing main table, especially where the main table is the result of any filtering
    or ranking steps.
    For example, if the client said 'Create a table with the top 10 stocks by market cap, include a column
    with their performance', you would pass the table with the top 10 stocks as the first table in the list, 
    tables with the stock performance as the second table and set add_columns=True.
    This will result in a table join that preserves the rows of the primary table, the information
    in any other tables will simply be added to those rows when possible. Again, you absolutely must
    set add_columns=True when one of the input tables is the direct or indirect result of any filtering
    and/or ranking, or when adding or including columns is mentioned by the client, do not forget!!!
    You must set add_columns=False when the user is simply combining equally important
    columns of data, for instance 'create a table with market cap, performance, and P/E for TSX',
    Generally you will only use add_columns=True when the client specifically mentions some kind of
    addition or inclusion of extra information after an initial analysis.

""",
    category=ToolCategory.TABLE,
)
async def join_tables(args: JoinTableArgs, context: PlanRunContext) -> Union[Table, StockTable]:
    logger = get_prefect_logger(__name__)

    if len(args.input_tables) == 0:
        raise RuntimeError("Cannot join an empty list of tables!")
    if len(args.input_tables) == 1:
        raise RuntimeError("Cannot join a list of tables with one element!")

    _join_table_func = _join_two_tables

    if args.row_join:
        # Make sure the row join is valid
        try:
            args = await _modify_join_args_smartly_with_gpt(context=context, args=args)
        except Exception:
            logger.exception("GPT failed to update join table args")

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
    for idx, table in enumerate(args.input_tables[1:]):
        joined_table = _join_table_func(
            joined_table, table, args.table_names, idx, first_table_is_primary=args.add_columns
        )

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
    description="""Given a table with at least one column of stocks, this function extracts
that column into a list of stock ID's.  This is very useful for e.g. filtering
on some numerical data in a table before extracting the stock list and fetching
other data with it.
The tool can be used to convert a table with a stock column from another tool
like get_portfolio_holdings into a list of stock ID's. This function can only be
used with actual tables, it cannot be used with either lists of texts or lists
of stocks. This tool is also useful when you used transform_table to filter and now want to get
stock ids from the filtered table to be used in another tool.
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
