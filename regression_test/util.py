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


async def validate_and_compare_text(llm: GPT, output_text: Text, prompt: str) -> None:
    assert isinstance(output_text, Text), "Output is not of type Text"
    assert output_text.val, "Expected non empty string"
    sys_prompt = Prompt(
        "You are a financial analyst tasked with verifying if a given response is related to a statement",
        "VERIFY_OUTPUT_RELATED_TO_PROMPT_SYS",
    )
    main_prompt = Prompt(
        f"Is this response {output_text} related to this statement {prompt}. Return only True or False.",
        "VERIFY_OUTPUT_RELATED_TO_PROMPT_MAIN",
    )
    res = await llm.do_chat_w_sys_prompt(
        sys_prompt.format(),
        main_prompt.format(output_text=output_text, prompt=prompt),
        max_tokens=50,
    )
    assert res == "True" or res == "true", "Issue with text"


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
