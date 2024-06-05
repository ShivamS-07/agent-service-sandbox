import datetime
import enum
from typing import List, Literal, Optional, Union

import numpy as np
import pandas as pd
from pydantic import BaseModel

from agent_service.io_type_utils import ComplexIOBase, IOType, PrimitiveType, io_type
from agent_service.io_types.output import Output, OutputType
from agent_service.io_types.stock import StockID
from agent_service.utils.boosted_pg import BoostedPG
from agent_service.utils.stock_metadata import StockMetadata

STOCK_ID_COL_NAME_DEFAULT = "Security"


class TableColumnType(str, enum.Enum):
    # Raw values
    INTEGER = "integer"
    STRING = "string"
    FLOAT = "float"
    BOOLEAN = "boolean"

    # A currency valued number
    CURRENCY = "currency"
    DATE = "date"  # YYYY-MM-DD
    DATETIME = "datetime"  # yyyy-mm-dd + ISO timestamp

    # Float value where 1.0 = 100%
    PERCENT = "percent"

    # Values for showing changes, anything above zero = green, below zero = red
    DELTA = "delta"  # Raw float delta
    PCT_DELTA = "pct_delta"  # Float delta value where 1.0 = 100% change

    # Special type that has stock metadata
    STOCK = "stock"

    @staticmethod
    def get_type_explanations() -> str:
        """
        Get a string to explain to the LLM what each table column type means (if
        not obvious).
        """
        return (
            "- 'currency': A column containing a price or other float with a currency attached. "
            "In this case the 'unit' is the currency ISO, please keep that consistent.\n"
            "- 'date/datetime': A column containing a python date or datetime object."
            "- 'percent': A column containing a percent value float. 100% is equal to 1.0, NOT 100. "
            "E.g. 25 percent is represented as 0.25.\n"
            "- 'delta': A float value representing a raw change over time. E.g. price change day over day.\n"
            "- 'pct_delta': A float value representing a change over time as a percent. "
            "100% is equal to 1.0 NOT 100. E.g. percent change of price day over day.\n"
            "- 'stock': A special column containing stock identifier information."
        )


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
    title: Optional[str] = None
    columns: List[TableColumn]

    def get_num_rows(self) -> int:
        if not self.columns:
            return 0
        return len(self.columns[0].data)

    def to_gpt_input(self) -> str:
        return f"[Table with {self.get_num_rows()} rows and {len(self.columns)} columns]"

    def to_df(self, stocks_as_tickers_only: bool = False) -> pd.DataFrame:
        data = {}
        for col in self.columns:
            if isinstance(col, StockTableColumn) and stocks_as_tickers_only:
                data[col.metadata.label] = list(
                    map(lambda stock: stock.symbol or stock.isin if stock else stock, col.data)
                )
            else:
                data[col.metadata.label] = col.data

        return pd.DataFrame(data=data)

    @staticmethod
    def from_df_and_cols(
        columns: List[TableColumnMetadata], data: pd.DataFrame, title: Optional[str] = None
    ) -> "Table":
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
                out_columns.append(StockTableColumn(metadata=col_meta, data=data[df_col].to_list()))
            else:
                out_columns.append(TableColumn(metadata=col_meta, data=data[df_col].to_list()))

        return Table(columns=out_columns, title=title)

    async def to_rich_output(self, pg: BoostedPG) -> Output:
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

        rows = df.values.tolist()

        return TableOutput(title=self.title, columns=output_cols, rows=rows)


CellType = Union[PrimitiveType, StockMetadata]


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
    title: Optional[str] = None
    columns: List[TableOutputColumn] = []
    rows: List[List[Optional[CellType]]]
