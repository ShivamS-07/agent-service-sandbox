import hashlib
import uuid
from copy import deepcopy
from typing import Any, Dict, Optional, Tuple, cast

from agent_service.io_type_utils import IOType, TableColumnType
from agent_service.io_types.stock import StockID
from agent_service.io_types.table import StockTable, Table, TableColumn, TableColumnMetadata
from agent_service.io_types.text import StockText, Text
from agent_service.io_types.text_objects import StockTextObject
from agent_service.planner.planner_types import ExecutionPlan
from agent_service.planner.sub_plan_executor import run_plan_simple
from agent_service.tool import ToolArgs, ToolCategory, tool
from agent_service.tools.tables import JoinTableArgs, join_tables
from agent_service.tools.tool_log import tool_log
from agent_service.types import AgentUserSettings, PlanRunContext
from agent_service.utils.async_utils import gather_with_concurrency
from agent_service.utils.feature_flags import get_ld_flag
from agent_service.utils.output_utils.output_construction import (
    PreparedOutput,
    prepare_list_of_stock_texts,
    prepare_list_of_texts,
)
from agent_service.utils.prefect import get_prefect_logger

FIRST_PASS_TOOLS = set(["transform_table", "get_statistic_data_for_companies"])


def reproducible_uuid4(data: str) -> str:
    # Hash the input data to produce a deterministic output
    hash_object = hashlib.sha256(data.encode("utf-8"))
    hashed_bytes = hash_object.digest()

    # Take the first 16 bytes to create a UUID
    truncated_bytes = hashed_bytes[:16]

    # Create a UUID from the modified bytes
    return str(uuid.UUID(bytes=bytes(truncated_bytes)))


def _hash_value(val: Any) -> Any:
    if isinstance(val, list):
        return tuple(_hash_value(item) for item in val)
    elif isinstance(val, dict):
        return tuple(sorted(val.items()))
    else:
        return hash(val)


def instantiate_subplan_with_task_ids(
    plan: ExecutionPlan, row: dict[str, IOType]
) -> Tuple[ExecutionPlan, Dict[str, Dict[str, IOType]]]:
    row_plan = deepcopy(plan)
    task_id_mapping = {}
    for step in row_plan.nodes:
        org_task_id = step.tool_task_id
        text_buffer = [step.tool_task_id]
        for key in sorted(row.keys()):
            val = row[key]
            text_buffer.append(str(_hash_value(val)))
        text_buffer_str = "-".join(text_buffer)
        new_task_id = reproducible_uuid4(text_buffer_str)
        step.tool_task_id = new_task_id
        if step.tool_name in FIRST_PASS_TOOLS:
            task_id_mapping[new_task_id] = {"template_task_id": org_task_id}
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
            fixed_output = output
            if isinstance(output, PreparedOutput):
                # Unwrap prepared outputs
                output = output.val

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
        new_table = await join_tables(
            args=JoinTableArgs(input_tables=sub_tables, row_join=True),
            context=context,
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


def subplanner_enabled(user_id: Optional[str], user_settings: Optional[AgentUserSettings]) -> bool:
    result = get_ld_flag("table-subplanner-enabled", default=False, user_context=user_id)
    return result


class PerRowProcessingArgs(ToolArgs):
    table: Table
    per_row_operations: str


@tool(
    description="""
This function takes a table and a description of operations using any of the tools at your
disposal. The per_row_operations string should explain how to use the information in a single
row of this table to create some new output that your client wants. For example, if you have a
table that which has a column corresponding to stocks, and another column corresponding to the
dates when that stock hit their 52-week high, and the client wants a news summary that 
attempts to explain the 52-week high using news over the month before their 52-week high, an
appropriate per_row_operations description would be:
`Create a date range consisting of a month before the provided date. Retrieve news texts for the
provided stock over that date range. Summarize that news with the goal of explaining why the stock
is up`
Your description should be plain English, it will be the job of another analyst to convert
it into a subroutine which will be applied to every row of the table. Generally you should not
refer to tools by their exact name (in the above example, we did not directly mention the
`get_date_range` tool in the first sentence even though that is what we would use). Your description
must always discuss the operations needed for a single row, you must NEVER discuss the
iteration over the table in your description (avoid words such as 'each' or 'every' unless
there is some kind of iteration within the process for a single row). It is also extremely important
that you do not include in your description any mention of operations you have already done, for
instance if you have derived tables which provide instances (stock/date pairs) of major drawdowns
and now need to do some further analysis with those dates, simply refer to `the provided date`,
do not mention that it is a date with a major drawdown unless you intend for this tool to calculate
drawdowns! Be concise, but be sure to mention every column and every operation that this tool needs
to get to the output the client wants. Although you do not need to name the tools, you must make sure
there is some plausible tool in your full list of tools that could be applied to accomplish every
operation you mention.
The output of this tool is a table consisting of the original table including one or more new columns
created by applying the per row operations described to every row of the table. In general, calling
prepare_output on both the input to this tool and the output to this tool is redundant, you should
only display the output to the client.
""",
    category=ToolCategory.TABLE,
    enabled_for_subplanner=False,
    enabled_checker_func=subplanner_enabled,
)
async def per_row_processing(args: PerRowProcessingArgs, context: PlanRunContext) -> Table:
    logger = get_prefect_logger(name="__main__")
    table = args.table

    # TODO: Skip planning if an update

    from agent_service.planner.planner import Planner

    await tool_log("Creating sub-plan...", context=context)
    variables = table.get_variables_from_table()  # probably need IOtypes for type checking?
    subplanner = Planner(
        agent_id=context.agent_id, user_id=context.user_id, send_chat=False, is_subplanner=True
    )
    subplan_template = await subplanner.create_subplan(args.per_row_operations, variables)

    if subplan_template:
        logger.info(f"Subplan template:\n{subplan_template.get_formatted_plan()}")
    else:
        raise (Exception("failed to create plan"))

    # do an initial run when needed
    rows = table.iterate_over_rows()

    if any([step.tool_name in FIRST_PASS_TOOLS for step in subplan_template.nodes]):
        logger.info("Doing initial run to get individual tool templates")
        # just grab first row and run it
        await run_plan_simple(subplan_template, context, variable_lookup=rows[0])

        logger.info("Finished initial run")

    # TODO: Save the subplan template in debug dict

    tasks = []

    logger.info(f"Applying plan to table of {len(rows)} rows")

    for row in rows:
        per_row_subplan, task_id_args = instantiate_subplan_with_task_ids(subplan_template, row)

        # TODO check cache, skip if no need to update
        tasks.append(
            run_plan_simple(
                per_row_subplan, context, variable_lookup=row, supplemental_args=task_id_args
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
the script, not those generated withinthe script/plan.
For example, if `date` is a provided variable and the user asks to do some summary over texts from
the month prior to the date, then you will build a string input to the get_date_range by first calling
this tool with the template "30 day period ending on {date}" and you would pass {"date":date} as the
mapping, which, if the date was "2024-12-18", would result in a the string: "30 day period ending on
2024-12-18", which could then be passed to the get_date_range tool. You must always use this tool to
build a string in this manner, you cannot use the regular python formatting! Other than date range,
you will often use this tool to construct strings such as topics, descriptions, or profiles corresponding
to the string arguments of other tools. Note that only this tool will accept f-strings, you must
never, ever use brackets like this in strings passed to any other tool (even other tools which have
wildcards will never use brackets.    
""",
    category=ToolCategory.TEXT_RETRIEVAL,  # the TEXT_RETRIEVAL tools are always included!
    enabled_for_subplanner=True,
    enabled=False,
)
async def string_builder(args: StringBuilderArgs, context: PlanRunContext) -> str:
    return args.template.format(**args.mapping)