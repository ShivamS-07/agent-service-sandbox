import hashlib
import inspect
import json
import uuid
from copy import deepcopy
from typing import Any, Dict, Optional, Tuple, cast

import numpy as np

from agent_service.io_type_utils import IOType, TableColumnType
from agent_service.io_types.stock import StockID
from agent_service.io_types.table import StockTable, Table, TableColumn, TableColumnMetadata
from agent_service.io_types.text import StockText, Text
from agent_service.io_types.text_objects import StockTextObject
from agent_service.planner.planner_types import ExecutionPlan
from agent_service.planner.sub_plan_executor import run_plan_simple
from agent_service.tool import TOOL_DEBUG_INFO, ToolArgs, ToolCategory, tool
from agent_service.tools.tables import JoinTableArgs, join_tables
from agent_service.tools.tool_log import tool_log
from agent_service.types import AgentUserSettings, ChatContext, Message, PlanRunContext
from agent_service.utils.async_utils import gather_with_concurrency
from agent_service.utils.feature_flags import get_ld_flag
from agent_service.utils.output_utils.output_construction import (
    PreparedOutput,
    prepare_list_of_stock_texts,
    prepare_list_of_texts,
)
from agent_service.utils.pagerduty import pager_wrapper
from agent_service.utils.prefect import get_prefect_logger
from agent_service.utils.tool_diff import get_prev_run_info

FIRST_PASS_TOOLS = set(["transform_table", "get_statistic_data_for_companies"])


def reproducible_uuid4(data: str) -> str:
    # Hash the input data to produce a deterministic output
    hash_object = hashlib.sha256(data.encode("utf-8"))
    hashed_bytes = hash_object.digest()

    # Take the first 16 bytes to create a UUID
    truncated_bytes = hashed_bytes[:16]

    # Create a UUID from the modified bytes
    return str(uuid.UUID(bytes=bytes(truncated_bytes)))


def _string_for_hashing(val: Any) -> str:
    if isinstance(val, list):
        return str(tuple(_string_for_hashing(item) for item in val))
    elif isinstance(val, dict):
        return str(tuple(sorted(val.items())))
    elif isinstance(val, StockID):
        return str(val.gbi_id)
    elif isinstance(val, Text):
        return str(val.id)
    else:
        return str(val)


def get_inputs_str(row: dict[str, IOType]) -> str:
    return "Inputs:\n" + "\n".join(f"{key} = {value}" for key, value in row.items()) + "\n"


def get_subplan_chat_context(directions: str, row: dict[str, IOType]) -> ChatContext:
    return ChatContext(
        messages=[Message(is_user_message=True, message=f"{get_inputs_str(row)}{directions}")]
    )


def instantiate_subplan_with_task_ids(
    plan: ExecutionPlan,
    row: dict[str, IOType],
) -> Tuple[ExecutionPlan, Dict[str, Dict[str, IOType]]]:
    row_plan = deepcopy(plan)
    task_id_mapping = {}
    for step in row_plan.nodes:
        org_task_id = step.tool_task_id
        text_buffer = [step.tool_task_id]
        for key in sorted(row.keys()):
            val = row[key]
            text_buffer.append(_string_for_hashing(val))
        text_buffer_str = "-".join(text_buffer)
        new_task_id = reproducible_uuid4(text_buffer_str)
        step.tool_task_id = new_task_id
        if step.tool_name in FIRST_PASS_TOOLS:
            task_id_mapping[new_task_id] = {"template_task_id": org_task_id}
        if step.tool_name == "summarize_texts":
            task_id_mapping[new_task_id] = {
                "plan_str": get_inputs_str(row) + plan.get_formatted_plan(numbered=True)
            }
    return row_plan, task_id_mapping  # type: ignore


async def _handle_list_output(output: list) -> IOType:
    fixed_output: IOType = "Unknown List Output!"
    if isinstance(output[0], StockID):
        output = cast(list[StockID], output)
        # TODO make this a bit cleaner
        stock_text_objs = [
            StockTextObject(
                gbi_id=stock.gbi_id,
                symbol=stock.symbol,
                company_name=stock.company_name,
                index=0,
            )
            for stock in output
        ]
        fixed_output = Text(text_objects=stock_text_objs)  # type: ignore
    elif isinstance(output[0], StockText):
        fixed_output = await prepare_list_of_stock_texts(output)
    elif isinstance(output[0], Text):
        fixed_output = await prepare_list_of_texts(output)

    return fixed_output


async def convert_per_row_results_to_table(
    input_table: Table,
    results: list[list[IOType]],
    context: PlanRunContext,
) -> Table:
    if not results:
        return input_table
    # This will require a lot of iteration, for now just convert everything to strings
    new_cols: list[TableColumn] = []
    # Create empty columns first
    for output in results[0]:
        title = str(output.title) if hasattr(output, "title") else "Unknown ???"  # type: ignore
        new_cols.append(
            TableColumn(
                # TODO handle other types
                metadata=TableColumnMetadata(label=title, col_type=TableColumnType.STRING),
                data=[],
            )
        )

    sub_tables = []
    for results_row in results:
        for i, output in enumerate(results_row):
            if isinstance(output, PreparedOutput):
                # Unwrap prepared outputs
                output = output.val

            fixed_output = output
            if isinstance(output, list) and output:
                fixed_output = await _handle_list_output(output=output)
            elif isinstance(output, Table) and len(results_row) == 1:
                # For now, do this table joining ONLY if the sub-plan returned
                # exactly one table. Other cases will be handled in the future.
                sub_tables.append(output)
                continue
            elif not isinstance(output, Text):
                fixed_output = f"Unhandled type: {type(output)}"
            new_cols[i].data.append(fixed_output)

    if sub_tables:
        # First, stack the output sub-tables into a single table
        joined_sub_table = await join_tables(
            args=JoinTableArgs(input_tables=sub_tables, row_join=True),
            context=context,
        )
        joined_sub_table = cast(Table, joined_sub_table)
        # Then, join the single new table to the input table
        new_table = await join_tables(
            args=JoinTableArgs(input_tables=[input_table, joined_sub_table]), context=context
        )
        new_table = cast(Table, new_table)
    else:
        new_table = deepcopy(input_table)
        new_table.columns.extend(new_cols)

    if new_table.get_stock_column():
        return StockTable(
            history=new_table.history,
            title=new_table.title,
            columns=new_table.columns,
            prefer_graph_type=new_table.prefer_graph_type,
        )
    return new_table


def pick_row_idx_for_initial_run(table: Table) -> int:
    date_column = table.get_date_column()
    if not date_column:
        return 0  # just return the first

    # get the oldest date, most sensible for using update logic
    return np.argmin(date_column.data)  # type: ignore


def subplanner_enabled(user_id: Optional[str], user_settings: Optional[AgentUserSettings]) -> bool:
    result = get_ld_flag("table-subplanner-enabled", default=False, user_context=user_id)
    return result


class PerRowProcessingArgs(ToolArgs):
    table: Table
    per_row_operations: str


@tool(
    description="""
This function takes a table and a description of operations using any of the tools at your
disposal and adds one or possibly multiple columns to the table by processing of one or
more of the cells of each row to create new cells in that same row.
The per_row_operations string should explain how to use the information in a single
row of the input table to create the new contents of that same row. For example, if you
have a table that which has a column corresponding to stocks, and another column corresponding
to the dates when that stock hit their 52-week high, and the client wants a news summary that 
attempts to explain the 52-week high using news over the month before their 52-week high, an
appropriate per_row_operations description would be:
`Create a date range consisting of a month before the provided date. Retrieve news texts for the
provided stock over that date range. Summarize that news with the goal of explaining why the stock
is up. Output the summary.`
Your per_row_operations description should be plain English, it will be the job of another analyst
to convert it into a subroutine which will be applied to every row of the table. It is not easy to write
a correct per_row_operations string! Do not ever copy the client's wording, which as aimed at the final
result, and not the specific operations needed at the per row level.
Please strictly follow the following rules associated with the contents of the per_row_operations string,
if you fail to follow any of these rules you will be summarily fired:

1. The per row operation description must not refer in any way to an input table or an output
table, it must never mention tables or columns or cells or rows unless some of its internal processing
requires the use of table that it is generating internally (this is very rare). Do NOT, under any
circumstances discuss getting variables from a table or adding the output into the output table, doing
that is not the job of the analyst who will read your description, including any such reference will
confuse them and cause them to make a mistake. For example, the client may ask you to 'add a column with
stock price change for the earnings date', but your per_row_operations description must simply talk
about deriving an individual stock price change, it must never mention a `column`, a `row`, or a `table`,
and you must avoid the words 'add' or 'include' your description entirely, at a per row processing 
level you are ALWAYS generating some output, use verbs such as 'calculate' (e.g. calculate the price
change), 'summarize' (i.e. summarize the news),  'find', (i.e. find the URL for their customer service), etc.
2. You must never, ever begin your per row summary by saying you will 'find' or 'retrieve' the information
that is already directly available in the input table. That step will happen automatically, including
any mention of it will deeply confuse your collegue responsible for writing the per row script. Assume
you have all the information from the table fully on-hand and refer to such information as the 'provided X'
(e.g. the provided date, the provided stock, etc.). Only discuss further operations the uses those inputs
to create your output(s).
be done on that information
3. Your description must only discuss the operations needed for a single row, you must NEVER discuss
the main iteration of this tool in your description in any way. You never use words such as 'each' or
'every' in your description unless there is some kind of iteration that is happening as part of the
processing for a single row. In cases where you would use 'each' or 'every, instead say 'the provided'!
4. You should not output what is already in the input table. 
5. Do not include in your description any mention of what the items in the table you are processing
mean, where they came from, unless absolutely required for the processing you are doing.
For instance if you have derived tables which provide instances (stock/date pairs) of major drawdowns
and now need to do some further analysis with those dates, simply refer to `the provided date` in your
per_row_operations description, you must not under any circumstances mention that it is a date with a
major drawdown! For relevant columns of the table you need to reference, just refer to 'the provided X',
never provide more information about any variable than what you need to in order to solve the problem at hand!
6. Be concise, but you should mention all the operations that this tool needs to do to get from the
input cells in one row to the output cell(s) in the same row. (but again, do not mention cells to discuss
any operations at the level of the table!). Although you should not name the tools directly and keep
your description readable (your description must be comprehensible to non-coders, no function names)
you must always make sure there is some plausible tool in your full list of tools that could be applied
to accomplish every operation you mention.
7. Your last step must always involve explicitly outputting the item (or items) that will become the new
cell (or cells) of the table. If the user is asking for multiple columns added to the table, you must
have a separate output sentence for each column requested. An output sentence should always be of the form
'Output the X', where X is the contents of a single column. For example, if the user asks for two
columns, one with a price change, one with a summary, you would finish your description by saying 
'Output the price change. Output the summary. Again, never, ever mention cells or columns!'

The per row outputs can be of any type (including tables themselves), this tool will convert automatically
convert them, in aggregate, to a valid table column.

The final output of this tool is a table consisting of the original table including one or more new columns
created by applying the per row operations described to every row of the table. Since the output tabel
has the entire contents of the input table, you should never, ever join the input and ouput tables
together explicitly in your code. Generally, you should never be calling prepare_output on both the
input table to this tool as well as the output, that would be very redundant. Instead you should only
display the output to the client.
""",
    category=ToolCategory.TABLE,
    enabled_for_subplanner=False,
    enabled_checker_func=subplanner_enabled,
)
async def per_row_processing(args: PerRowProcessingArgs, context: PlanRunContext) -> Table:
    logger = get_prefect_logger(name="__main__")
    table = args.table

    prev_run_info = None
    subplan_template = None
    rows = []
    try:  # since everything associated with diffing is optional, put in try/except
        # Update mode
        prev_run_info = await get_prev_run_info(context, "per_row_processing")
        if prev_run_info is not None:
            prev_args = PerRowProcessingArgs.model_validate_json(prev_run_info.inputs_str)
            prev_table_metadata = [col.metadata for col in prev_args.table.columns]
            curr_table_metadata = [col.metadata for col in args.table.columns]
            if prev_table_metadata == curr_table_metadata:
                subplan_template = ExecutionPlan.from_dict(
                    json.loads(prev_run_info.debug["subplan_template"])
                )
                await tool_log("Loaded sub-plan from previous run", context=context)
                rows = table.iterate_over_rows(use_variables=True)

    except Exception as e:
        logger.exception(f"Error loading subplan from previous run: {e}")
        pager_wrapper(
            current_frame=inspect.currentframe(),
            module_name=__name__,
            context=context,
            e=e,
            classt="AgentUpdateError",
            summary="Failed to update per_row_processing",
        )

    sub_plan_context = deepcopy(context)
    sub_plan_context.skip_task_logging = True
    if subplan_template is None:
        from agent_service.planner.planner import Planner

        await tool_log(
            f"Creating sub-plan using the following directions: {args.per_row_operations}",
            context=context,
        )
        variables = table.get_variables_from_table()
        subplanner = Planner(
            agent_id=context.agent_id, user_id=context.user_id, send_chat=False, is_subplanner=True
        )
        subplan_template = await subplanner.create_subplan(args.per_row_operations, variables)

        if subplan_template:
            logger.info(f"Subplan template:\n{subplan_template.get_formatted_plan()}")
        else:
            raise (Exception("failed to create plan"))

        # do an initial run when needed
        rows = table.iterate_over_rows(use_variables=True)

        if any([step.tool_name in FIRST_PASS_TOOLS for step in subplan_template.nodes]):
            row = rows[pick_row_idx_for_initial_run(table)]
            logger.info("Doing initial run to get individual tool templates")

            inst_sub_plan_context = deepcopy(sub_plan_context)
            inst_sub_plan_context.chat = get_subplan_chat_context(args.per_row_operations, row)

            await run_plan_simple(subplan_template, inst_sub_plan_context, variable_lookup=row)

            logger.info("Finished initial run")

    debug_info: Dict[str, Any] = {"subplan_template": subplan_template.model_dump_json()}
    TOOL_DEBUG_INFO.set(debug_info)

    tasks = []

    logger.info(f"Applying subplan to table of {len(rows)} rows")

    for row in rows:
        per_row_subplan, task_id_args = instantiate_subplan_with_task_ids(subplan_template, row)

        inst_sub_plan_context = deepcopy(sub_plan_context)
        inst_sub_plan_context.chat = get_subplan_chat_context(args.per_row_operations, row)

        tasks.append(
            run_plan_simple(
                per_row_subplan,
                inst_sub_plan_context,
                variable_lookup=row,
                supplemental_args=task_id_args,
            )
        )

    await tool_log("Executing sub-plan...", context=context)
    results: list[list[IOType]] = await gather_with_concurrency(tasks, n=25)

    logger.info("Agglomerating individual results to single output table")

    return await convert_per_row_results_to_table(
        input_table=table, results=results, context=context
    )


class StringBuilderArgs(ToolArgs):
    template: str
    mapping: Dict[str, IOType]


@tool(
    description="""
This function takes a template string in python f-string format and a dictionary that maps strings
corresponding to placeholders in the template string (indicated in the template string by curly
brackets, e.g {date}) to variables of any type that will fill those slots. The output is a string
where the placeholders have been replaced with a string representation of the object within the variable.
Generally, this tool should only be used for variables which are initialized before the beginning of
the script, not those generated within the script/plan.
For example, if `date` is a provided variable and the user asks to do some summary over texts from
the month prior to the date, then you will build a string input to the get_date_range by first calling
this tool with the template "30 day period ending on {date}" and you would pass {"date":date} as the
mapping, which, if the date was "2024-12-18", would result in a the string: "30 day period ending on
2024-12-18", which could then be passed to the get_date_range tool. If you just want to convert the
date itself to a string, you can use the template "{date}".  You must always use this tool to
build a string in this manner, you cannot use the regular python formatting! Other than date range,
you will often use this tool to construct strings such as topics, descriptions, or profiles corresponding
to the string arguments of other tools. Another example: if you had a variable
Name that corresponded to a name of a person and wanted do a web search for their address, you would
construction an appropriate query/topic for a search by passing in "{name}'s address" as the template
and {"name":Name} as the mapping. Note that only this tool will accept f-strings, you must
never, ever use brackets like this in strings passed to any other tool (even other tools which have
wildcards will never use brackets.    
""",
    category=ToolCategory.TEXT_RETRIEVAL,  # the TEXT_RETRIEVAL tools are always included!
    enabled_for_subplanner=True,
    enabled=False,
)
async def string_builder(args: StringBuilderArgs, context: PlanRunContext) -> str:
    return args.template.format(**args.mapping)
