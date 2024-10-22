from typing import List, Optional, cast

from agent_service.io_type_utils import ComplexIOBase, IOType, TableColumnType, io_type
from agent_service.io_types.stock import StockID
from agent_service.io_types.table import (
    Table,
    TableColumn,
    TableColumnMetadata,
    object_histories_to_columns,
)
from agent_service.io_types.text import Text
from agent_service.io_types.text_objects import StockTextObject
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
        # This will display summaries added using per_stock_group_summarize_text
        columns = []

        name_column = TableColumn(
            metadata=TableColumnMetadata(label=self.header, col_type=TableColumnType.STRING),
            data=[group.name for group in self.stock_groups],
        )

        columns.append(name_column)

        extra_columns = object_histories_to_columns(
            objects=cast(List[ComplexIOBase], self.stock_groups)
        )

        columns.extend(extra_columns)

        stock_column_texts = []
        for stock_group in self.stock_groups:
            stock_objs = []
            for stock in stock_group.stocks:
                stock_objs.append(
                    StockTextObject(
                        gbi_id=stock.gbi_id,
                        symbol=stock.symbol,
                        company_name=stock.company_name,
                        isin=stock.isin,
                        index=0,
                    )
                )
            stock_column_texts.append(Text(text_objects=stock_objs))  # type:ignore

        stocks_column = TableColumn(
            metadata=TableColumnMetadata(label="Stocks", col_type=TableColumnType.STRING),
            data=stock_column_texts,  # type:ignore
        )
        columns.append(stocks_column)

        output = [PreparedOutput(val=Table(columns=columns), title=main_title or self.header)]

        try:  # display table for each stock group if they were selected by profile filtering
            if any(
                [
                    history_entry.title.startswith("Connection to")
                    for history_entry in self.stock_groups[0].stocks[0].history
                    if history_entry.title
                ]
            ):
                output.extend(
                    [
                        PreparedOutput(val=stock_group.stocks, title=stock_group.name)
                        for stock_group in self.stock_groups
                    ]
                )
        except IndexError:
            pass
        return output  # type: ignore
