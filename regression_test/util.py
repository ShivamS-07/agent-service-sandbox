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
    assert isinstance(output_text, Text), "Output is not of type Text"
    assert output_text.val, "Expected non empty string"
    sys_prompt = Prompt(
        "You are a financial analyst tasked with verifying if a given response is related to a statement",
        "VERIFY_OUTPUT_RELATED_TO_PROMPT_SYS",
    )
    main_prompt = Prompt(
        f"Is this response {output_text.val} related to this statement {prompt}. Return only True or False.",
        "VERIFY_OUTPUT_RELATED_TO_PROMPT_MAIN",
    )
    res = await llm.do_chat_w_sys_prompt(
        sys_prompt.format(),
        main_prompt.format(output_text=output_text, prompt=prompt),
        max_tokens=50,
    )
    assert (
        res.lower() == "true"
    ), "Issue with the text output. ChatGPT doesn't think that the output answers the prompt"


async def compare_with_expected_text(
    llm: GPT, output_text: Text, prompt: str, expected_text: str
) -> None:
    assert isinstance(output_text, Text), "Output is not of type Text"
    assert output_text.val, "Expected non empty string"
    sys_prompt = Prompt(
        "You are tasked with comparing the actual and expected output for a prompt provided to a financial analyst",
        "COMPARE_TEXT_PROMPT_SYS",
    )
    main_prompt = Prompt(
        f"Prompt is {prompt}. Actual response is {output_text.val}. Expected response is {expected_text}."
        f"Is the given response similar to the expected response for the prompt? Return only True or False.",
        "COMPARE_TEXT_PROMPT_MAIN",
    )
    res = await llm.do_chat_w_sys_prompt(
        sys_prompt.format(),
        main_prompt.format(output_text=output_text, prompt=prompt),
        max_tokens=50,
    )
    assert res.lower() == "true", (
        "Issue with the text output. "
        "ChatGPT doesn't think that the actual and expected outputs are similar"
    )


def validate_table_and_get_columns(
    output_stock_table: Table, column_types: List[TableColumnType]
) -> List[TableColumn]:
    assert isinstance(output_stock_table, Table), "Output is not of type Table"
    assert len(output_stock_table.columns) > 0, "Expects non-zero number of columns in table"
    columns = []
    for column_type in column_types:
        column = find_column(output_stock_table.columns, column_type)
        assert column
        columns.append(column)
    return columns


def validate_line_graph(output_line_graph: LineGraph) -> None:
    assert isinstance(output_line_graph, LineGraph), "Output is not of type LineGraph"
    assert len(output_line_graph.data) > 0, "Empty line graph"
