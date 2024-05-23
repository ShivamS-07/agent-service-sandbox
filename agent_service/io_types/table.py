import enum
from typing import Any, Callable, List, Literal, Optional, Union, cast

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
    # keep track of this so that we can easily transform for the frontend later
    col_label_is_stock_id: bool = False
    is_indexed: bool = False

    def to_output_column(self) -> "TableOutputColumn":
        # TODO switch GBI ID's to tickers if needed, etc.
        return TableOutputColumn(
            name=str(self.label),
            col_type=self.col_type,
            unit=self.unit,
            is_highlighted=self.is_indexed,
        )


@io_type
class Table(ComplexIOBase):
    title: Optional[str] = None
    columns: List[TableColumn]
    data: pd.DataFrame

    def to_gpt_input(self) -> str:
        return f"[Table with {len(self.data)} rows and {len(self.data.columns)} columns]"

    async def to_rich_output(self, pg: BoostedPG) -> Output:
        df = self.data.copy(deep=True)
        # We already have a column for the index
        df = df.reset_index()

        # Fetch all stock metadata needed in one call
        gbi_ids_to_fetch: List[int] = []
        for df_col, col in zip(df.columns, self.columns):
            if col.col_label_is_stock_id:
                gbi_ids_to_fetch.append(cast(int, col.label))
            if col.col_type == TableColumnType.STOCK:
                gbi_ids_to_fetch.extend(df[df_col])

        stock_metadata = {}
        if gbi_ids_to_fetch:
            stock_metadata = await get_stock_metadata(pg=pg, gbi_ids=gbi_ids_to_fetch)

        output_cols = []
        for df_col, col in zip(df.columns, self.columns):
            output_col = col.to_output_column()
            if col.col_label_is_stock_id:
                output_col.name = stock_metadata[int(col.label)].symbol  # type: ignore
            if col.col_type == TableColumnType.STOCK:
                df[df_col] = [stock_metadata[gbi_id] for gbi_id in df[df_col]]

            output_cols.append(output_col)

        df = df.replace(np.nan, None)
        rows = df.values.tolist()

        return TableOutput(title=self.title, columns=output_cols, rows=rows)

    @field_validator("data", mode="before")
    @classmethod
    def _deserializer(cls, data: Any) -> Any:
        if isinstance(data, dict):
            data = pd.DataFrame.from_dict(data)
            # Recreate index from first column
            data = data.set_index(data.columns[0])
            try:
                # Try to convert the index to a DatetimeIndex if it's
                # compatible, otherwise just leave as is.
                data.index = pd.to_datetime(data.index)
            except Exception:
                pass
        return data

    @field_serializer("data", mode="wrap")
    @classmethod
    def _field_serializer(cls, data: Any, dumper: Callable) -> Any:
        if isinstance(data, pd.DataFrame):
            # Reset index to maintain index name
            data = data.reset_index().to_dict()
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
