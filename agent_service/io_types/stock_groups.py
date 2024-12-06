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
from agent_service.utils.stock_metadata import get_stock_metadata


@io_type
class StockGroup(ComplexIOBase):
    name: str
    stocks: List[StockID]
    ref_stock: Optional[StockID] = None


@io_type
class StockGroups(ComplexIOBase):
    stock_groups: List[StockGroup]
    header: str = "Stock Group"
    stock_list_header: str = "Stocks"

    async def split_into_components(self, main_title: Optional[str] = None) -> List[IOType]:
        # This will display summaries added using per_stock_group_summarize_text
        from agent_service.utils.async_db import get_async_db

        db = get_async_db()
        stocks = [stock for stock_group in self.stock_groups for stock in stock_group.stocks]
        stocks.extend(
            [stock_group.ref_stock for stock_group in self.stock_groups if stock_group.ref_stock]
        )
        stock_meta = await get_stock_metadata(gbi_ids=[stock.gbi_id for stock in stocks], pg=db.pg)
        columns = []

        data: List[str | Text] = []
        for group in self.stock_groups:
            if not group.ref_stock:
                data.append(group.name)
                continue
            meta = stock_meta.get(group.ref_stock.gbi_id)
            if meta:
                stock_text_obj = StockTextObject(**meta.model_dump(), index=0)
            else:
                stock_text_obj = StockTextObject(
                    gbi_id=group.ref_stock.gbi_id,
                    symbol=group.ref_stock.symbol,
                    company_name=group.ref_stock.company_name,
                    isin=group.ref_stock.isin,
                    index=0,
                )
            data.append(Text(text_objects=[stock_text_obj]))

        name_column = TableColumn(
            metadata=TableColumnMetadata(label=self.header, col_type=TableColumnType.STRING),
            data=data,  # type: ignore
        )

        columns.append(name_column)

        stock_column_texts = []
        for stock_group in self.stock_groups:
            stock_objs = []
            for stock in stock_group.stocks:
                meta = stock_meta.get(stock.gbi_id)
                if meta:
                    stock_text_obj = StockTextObject(**meta.model_dump(), index=0)
                else:
                    stock_text_obj = StockTextObject(
                        gbi_id=stock.gbi_id,
                        symbol=stock.symbol,
                        company_name=stock.company_name,
                        isin=stock.isin,
                        index=0,
                    )
                stock_objs.append(stock_text_obj)
            stock_column_texts.append(Text(text_objects=stock_objs))  # type:ignore

        stocks_column = TableColumn(
            metadata=TableColumnMetadata(
                label=self.stock_list_header, col_type=TableColumnType.STRING
            ),
            data=stock_column_texts,  # type:ignore
        )
        columns.append(stocks_column)

        extra_columns = object_histories_to_columns(
            objects=cast(List[ComplexIOBase], self.stock_groups)
        )

        columns.extend(extra_columns)

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
                        if stock_group.ref_stock
                        is None  # don't do this for competitor stock groups
                    ]
                )
        except IndexError:
            pass
        return output  # type: ignore
