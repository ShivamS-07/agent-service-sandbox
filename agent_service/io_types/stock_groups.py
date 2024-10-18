from typing import List, Optional, cast

from agent_service.io_type_utils import ComplexIOBase, IOType, TableColumnType, io_type
from agent_service.io_types.stock import StockID
from agent_service.io_types.table import (
    Table,
    TableColumn,
    TableColumnMetadata,
    object_histories_to_columns,
)
from agent_service.utils.output_utils.output_construction import PreparedOutput


@io_type
class StockGroup(ComplexIOBase):
    name: str
    stocks: List[StockID]


@io_type
class StockGroups(ComplexIOBase):
    stock_groups: List[StockGroup]
    header: str = "Stock Group"

    async def split_into_components(self, main_title: Optional[str] = None) -> List[IOType]:
        output = [
            PreparedOutput(val=stock_group.stocks, title=stock_group.name)
            for stock_group in self.stock_groups
        ]
        # This will display summaries added using per_stock_group_summarize_text
        extra_columns = object_histories_to_columns(
            objects=cast(List[ComplexIOBase], self.stock_groups)
        )
        if extra_columns and main_title:
            name_column = TableColumn(
                metadata=TableColumnMetadata(label=self.header, col_type=TableColumnType.STRING),
                data=[group.name for group in self.stock_groups],
            )
            output.append(
                PreparedOutput(val=Table(columns=[name_column] + extra_columns), title=main_title)
            )
        return output  # type: ignore
