import enum
from typing import Any, Callable, List, Literal, Optional, Union

import numpy as np
import pandas as pd
from pydantic import BaseModel
from pydantic.functional_serializers import field_serializer
from pydantic.functional_validators import field_validator

from agent_service.io_type_utils import ComplexIOBase, PrimitiveType, io_type
from agent_service.io_types.output import Output, OutputType
from agent_service.utils.boosted_pg import BoostedPG
from agent_service.utils.stock_metadata import StockMetadata, get_stock_metadata


class TableColumnType(str, enum.Enum):
    # Raw values
    INTEGER = "integer"
    STRING = "string"
    FLOAT = "float"
    BOOLEAN = "boolean"

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


@io_type
class TableColumn(ComplexIOBase):
    label: PrimitiveType
    col_type: TableColumnType
    unit: Optional[str] = None

    def to_output_column(self) -> "TableOutputColumn":
        # TODO switch GBI ID's to tickers if needed, etc.
        return TableOutputColumn(
            name=str(self.label),
            col_type=self.col_type,
            unit=self.unit,
        )


def _convert_timestamp_cols(cols: List[TableColumn], df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a dataframe, check for Timestamp columns and convert to date/datetime.
    """
    for col, df_col in zip(cols, df.columns):
        if col.col_type == TableColumnType.DATE:
            df[df_col] = df[df_col].apply(
                func=lambda val: val.date() if isinstance(val, pd.Timestamp) else val
            )
        elif col.col_type == TableColumnType.DATETIME:
            df[df_col] = df[df_col].apply(
                func=lambda val: val.to_pydatetime() if isinstance(val, pd.Timestamp) else val
            )

    return df


@io_type
class Table(ComplexIOBase):
    title: Optional[str] = None
    columns: List[TableColumn]
    data: pd.DataFrame

    def to_gpt_input(self) -> str:
        return f"[Table with {len(self.data)} rows and {len(self.data.columns)} columns]"

    async def to_rich_output(self, pg: BoostedPG) -> Output:
        df = self.data.copy(deep=True)

        # Fetch all stock metadata needed in one call
        gbi_ids_to_fetch: List[int] = []
        for df_col, col in zip(df.columns, self.columns):
            if col.col_type == TableColumnType.STOCK:
                gbi_ids_to_fetch.extend(df[df_col])

        stock_metadata = {}
        if gbi_ids_to_fetch:
            stock_metadata = await get_stock_metadata(pg=pg, gbi_ids=gbi_ids_to_fetch)

        output_cols = []
        is_first_col = True
        for df_col, col in zip(df.columns, self.columns):
            output_col = col.to_output_column()
            if col.col_type == TableColumnType.STOCK:
                df[df_col] = [stock_metadata[gbi_id] for gbi_id in df[df_col]]
                if is_first_col:
                    # Automatically highlight the first column if it's a stock column
                    output_col.is_highlighted = True
            elif col.col_type in (TableColumnType.DATE, TableColumnType.DATETIME) and is_first_col:
                # Automatically highlight the first column if it's a date column
                output_col.is_highlighted = True

            output_cols.append(output_col)
            is_first_col = False

        df = df.replace(np.nan, None)
        rows = df.values.tolist()

        return TableOutput(title=self.title, columns=output_cols, rows=rows)

    def model_post_init(self, __context: Any) -> None:
        self.data = _convert_timestamp_cols(cols=self.columns, df=self.data)
        return super().model_post_init(__context)

    @field_validator("data", mode="before")
    @classmethod
    def _deserializer(cls, data: Any) -> Any:
        if isinstance(data, dict):
            data = pd.DataFrame.from_dict(data)
            # No index
            data = data.reset_index(drop=True)
        return data

    @field_serializer("data", mode="wrap")
    @classmethod
    def _field_serializer(cls, data: Any, dumper: Callable) -> Any:
        if isinstance(data, pd.DataFrame):
            data = data.to_dict()
        return dumper(data)


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
