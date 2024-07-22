# type: ignore
import logging
from typing import List, Optional

from agent_service.GPT.requests import GPT
from agent_service.io_type_utils import TableColumnType
from agent_service.io_types.graph import LineGraph
from agent_service.io_types.table import Table, TableColumn
from agent_service.io_types.text import Text
from agent_service.utils.prompt_utils import Prompt

logger = logging.getLogger(__name__)


def find_column(columns: List[TableColumn], col_type: TableColumnType) -> Optional[TableColumn]:
    for column in columns:
        if column.metadata.col_type == col_type:
            return column
    return None


async def validate_text(llm: GPT, output_text: Text, prompt: str) -> None:
    assert isinstance(output_text, Text), f"Expected type: Text. Actual type: {type(output_text)}"
    assert output_text.val, "Expected non empty string"
    sys_prompt = Prompt(
        "You are a financial analyst tasked with verifying if a given response is related to a statement",
        "VERIFY_OUTPUT_RELATED_TO_PROMPT_SYS",
    )
    main_prompt = Prompt(
        f"Is this response {output_text.val} related to this statement {prompt}. Return only True or False on first "
        f"line. On a newline give a very brief reason behind your decision.",
        "VERIFY_OUTPUT_RELATED_TO_PROMPT_MAIN",
    )
    res = await llm.do_chat_w_sys_prompt(
        sys_prompt.format(),
        main_prompt.format(output_text=output_text, prompt=prompt),
        max_tokens=50,
    )
    result, reason = res.split("\n")
    assert (
        result.lower() == "true"
    ), f"GPT doesn't think that the output answers the prompt because\n{reason}"


async def compare_with_expected_text(
    llm: GPT, output_text: Text, prompt: str, expected_text: str
) -> None:
    assert isinstance(output_text, Text), f"Expected type: Text. Actual type: {type(output_text)}"
    assert output_text.val, "Expected non empty string"
    sys_prompt = Prompt(
        "You are tasked with comparing the actual and expected output for a prompt provided to an AI financial "
        "analyst being tested",
        "COMPARE_TEXT_PROMPT_SYS",
    )
    main_prompt = Prompt(
        f"Prompt is {prompt}. Actual response is {output_text.val}. Expected response is {expected_text}."
        f"Is the given response similar to the expected response for the prompt? Return only True or False on first "
        f"line. On a newline give a very brief reason behind your decision.",
        "COMPARE_TEXT_PROMPT_MAIN",
    )
    res = await llm.do_chat_w_sys_prompt(
        sys_prompt.format(),
        main_prompt.format(output_text=output_text, prompt=prompt),
        max_tokens=50,
    )
    result, reason = res.split("\n")
    assert (
        result.lower() == "true"
    ), f"GPT doesn't think that the actual and expected text outputs are similar because\n{reason}."


def validate_table_and_get_columns(
    output_stock_table: Table, column_types: List[TableColumnType]
) -> List[TableColumn]:
    assert isinstance(
        output_stock_table, Table
    ), f"Expected type: Table. Actual type: {type(output_stock_table)}."
    assert len(output_stock_table.columns) > 0, "Zero columns in table"
    columns = []
    for column_type in column_types:
        column = find_column(output_stock_table.columns, column_type)
        assert column, f"{column} not found in table"
        columns.append(column)
    return columns


def validate_line_graph(output_line_graph: LineGraph) -> None:
    assert isinstance(
        output_line_graph, LineGraph
    ), f"Expected type: LineGraph. Actual type: {type(output_line_graph)}"
    assert len(output_line_graph.data) > 0, "Empty line graph"
