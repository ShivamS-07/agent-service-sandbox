import datetime
from typing import Any, Dict, List, Literal, Optional, Union

import numpy as np
import pandas as pd
from pydantic import BaseModel
from typing_extensions import Self

from agent_service.io_type_utils import (
    ComplexIOBase,
    IOType,
    PrimitiveType,
    TableColumnType,
    io_type,
)
from agent_service.io_types.output import Output, OutputType
from agent_service.io_types.stock import StockID
from agent_service.utils.boosted_pg import BoostedPG
from agent_service.utils.stock_metadata import StockMetadata

STOCK_ID_COL_NAME_DEFAULT = "Security"


@io_type
class TableColumnMetadata(ComplexIOBase):
    label: PrimitiveType
    col_type: TableColumnType
    unit: Optional[str] = None


@io_type
class TableColumn(ComplexIOBase):
    metadata: TableColumnMetadata
    data: List[Optional[IOType]]

    def to_output_column(self) -> "TableOutputColumn":
        # TODO switch GBI ID's to tickers if needed, etc.
        return TableOutputColumn(
            name=str(self.metadata.label),
            col_type=self.metadata.col_type,
            unit=self.metadata.unit,
        )


@io_type
class StockTableColumn(TableColumn):
    metadata: TableColumnMetadata = TableColumnMetadata(
        label=STOCK_ID_COL_NAME_DEFAULT, col_type=TableColumnType.STOCK
    )
    data: List[StockID]  # type: ignore


@io_type
class DateTableColumn(TableColumn):
    metadata: TableColumnMetadata = TableColumnMetadata(label="Date", col_type=TableColumnType.DATE)
    data: List[datetime.date]  # type: ignore


@io_type
class Table(ComplexIOBase):
    columns: List[TableColumn]

    def get_num_rows(self) -> int:
        if not self.columns:
            return 0
        return len(self.columns[0].data)

    def to_gpt_input(self) -> str:
        return f"[Table with {self.get_num_rows()} rows and {len(self.columns)} columns]"

    def to_df(
        self, stocks_as_tickers_only: bool = False, stocks_as_hashables: bool = False
    ) -> pd.DataFrame:
        data = {}
        for col in self.columns:
            data[col.metadata.label] = col.data
            if isinstance(col, StockTableColumn):
                if stocks_as_tickers_only:
                    data[col.metadata.label] = list(
                        map(lambda stock: stock.symbol or stock.isin if stock else stock, col.data)
                    )
                elif stocks_as_hashables:
                    data[col.metadata.label] = [stock.to_hashable() for stock in col.data]

        return pd.DataFrame(data=data)

    def to_dict(self, key_cols: List[TableColumnMetadata]) -> Dict[Any, List[IOType]]:
        """
        Creates a dictionary of data from the table, where the key is a tuple
        from the values in key_cols, and the value is a list of the other
        columns in the row.
        """
        df = self.to_df()
        df.set_index(keys=[col.label for col in key_cols], inplace=True)
        return df.transpose().to_dict(orient="list")  # type: ignore

    @classmethod
    def from_df_and_cols(
        cls,
        columns: List[TableColumnMetadata],
        data: pd.DataFrame,
        stocks_are_hashable_objs: bool = False,
    ) -> Self:
        out_columns: List[TableColumn] = []
        data = data.replace(np.nan, None)
        for col_meta, df_col in zip(columns, data.columns):
            if col_meta.col_type == TableColumnType.DATE:
                data[df_col] = data[df_col].apply(
                    func=lambda val: val.date() if isinstance(val, pd.Timestamp) else val
                )
                out_columns.append(DateTableColumn(metadata=col_meta, data=data[df_col].to_list()))
            elif col_meta.col_type == TableColumnType.DATETIME:
                data[df_col] = data[df_col].apply(
                    func=lambda val: val.to_pydatetime() if isinstance(val, pd.Timestamp) else val
                )
                out_columns.append(DateTableColumn(metadata=col_meta, data=data[df_col].to_list()))
            elif col_meta.col_type == TableColumnType.STOCK:
                stocks = data[df_col].to_list()
                if stocks_are_hashable_objs:
                    stocks = [StockID.from_hashable(stock) for stock in stocks]
                out_columns.append(StockTableColumn(metadata=col_meta, data=stocks))
            else:
                out_columns.append(TableColumn(metadata=col_meta, data=data[df_col].to_list()))

        return cls(columns=out_columns)

    async def to_rich_output(self, pg: BoostedPG, title: str = "") -> Output:
        output_cols = []
        is_first_col = True
        # Use a dataframe for convenience
        df = self.to_df()
        for df_col, col in zip(df.columns, self.columns):
            output_col = col.to_output_column()
            if output_col.col_type == TableColumnType.STOCK:
                # Map to StockMetadata
                df[df_col] = df[df_col].map(
                    lambda val: (
                        StockMetadata(
                            gbi_id=val.gbi_id,
                            symbol=val.symbol,
                            isin=val.isin,
                            company_name=val.company_name,
                        )
                        if isinstance(val, StockID)
                        else val
                    )
                )
                if is_first_col:
                    # Automatically highlight the first column if it's a stock column
                    output_col.is_highlighted = True
            elif (
                output_col.col_type in (TableColumnType.DATE, TableColumnType.DATETIME)
                and is_first_col
            ):
                # Automatically highlight the first column if it's a date column
                output_col.is_highlighted = True

            output_cols.append(output_col)
            is_first_col = False

        df = df.replace(np.nan, None)
        rows = df.values.tolist()

        return TableOutput(title=title, columns=output_cols, rows=rows)


CellType = Union[PrimitiveType, StockMetadata]


@io_type
class StockTable(Table):
    """
    Wrapper around a table, used really only for type hinting.
    """

    pass


class TableOutputColumn(BaseModel):
    """
    Column metadata necessary for the frontend.
    """

    name: str
    col_type: TableColumnType
    # For things like currency, frontend will handle truncation for numbers.
    unit: Optional[str] = None
    # If e.g. the first column is of special importance and should be
    # highlighted for all rows.
    is_highlighted: bool = False


class TableOutput(Output):
    output_type: Literal[OutputType.TABLE] = OutputType.TABLE
    title: str = ""
    columns: List[TableOutputColumn] = []
    rows: List[List[Optional[CellType]]]
