from typing import List, cast

from agent_service.io_type_utils import TableColumnType
from agent_service.io_types.stock import StockID
from agent_service.io_types.stock_groups import StockGroups
from agent_service.io_types.table import (
    StockTable,
    StockTableColumn,
    Table,
    TableColumn,
    TableColumnMetadata,
)
from agent_service.utils.prefect import get_prefect_logger


def get_stock_group_input_tables(
    stock_table: StockTable, stock_groups: StockGroups
) -> List[StockTable]:
    logger = get_prefect_logger(__name__)
    group_table_lookup = {}
    stock_table_lookup = {}
    for group in stock_groups.stock_groups:
        new_columns: List[TableColumn] = []
        for column in stock_table.columns:
            if column.metadata.col_type == TableColumnType.STOCK:
                new_columns.append(StockTableColumn(metadata=column.metadata, data=[]))
            else:
                new_columns.append(TableColumn(metadata=column.metadata, data=[]))
        group_table = StockTable(columns=new_columns)
        group_table_lookup[group.name] = group_table
        for stock in group.stocks:
            stock_table_lookup[stock] = group_table

    stock_column = stock_table.get_stock_column()
    if not stock_column:
        raise RuntimeError("Input table to per_stock_group_table_transform is not a StockTable")

    stocks = cast(List[StockID], stock_column.data)

    for row_idx, stock in enumerate(stocks):
        if stock not in stock_table_lookup:
            logger.warning(f"Missing {stock.symbol} in input table")
        group_table = stock_table_lookup[stock]
        for col_idx in range(len(stock_table.columns)):
            group_table.columns[col_idx].data.append(stock_table.columns[col_idx].data[row_idx])

    output_tables = []
    for group in stock_groups.stock_groups:  # list of tables must align with stock_groups
        output_tables.append(group_table_lookup[group.name])
    return output_tables


def remove_stock_group_columns(table: Table, group_header: str) -> None:
    to_delete = []
    for i, column in enumerate(table.columns):  # get rid of existing columns if they exist
        if column.metadata.label == "Stock Group" or column.metadata.label == group_header:
            to_delete.append(i)
    to_delete.sort(reverse=True)
    for i in to_delete:
        del table.columns[i]


def add_stock_group_column(table: Table, groups_header: str, group_name: str) -> None:
    new_stock_group_column = TableColumn(
        metadata=TableColumnMetadata(label=groups_header, col_type=TableColumnType.STRING),
        data=[group_name] * len(table.columns[0].data),
    )
    table.columns.insert(0, new_stock_group_column)
