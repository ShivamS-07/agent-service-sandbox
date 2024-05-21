import enum
from typing import Any, Callable, List, Literal, Optional, Union

import pandas as pd
from pydantic import BaseModel
from pydantic.functional_serializers import field_serializer
from pydantic.functional_validators import field_validator

from agent_service.io_type_utils import ComplexIOBase, PrimitiveType, io_type
from agent_service.io_types.output import Output, OutputType


class TableColumnType(str, enum.Enum):
    # Raw values
    INTEGER = "integer"
    STRING = "string"
    FLOAT = "float"
    BOOLEAN = "boolean"

    CURRENCY = "currency"
    DATE = "date"  # YYYY-MM-DD
    DATETIME = "datetime"  # y/m/d + ISO timestamp

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
    col_label_is_gbi_id: bool = False

    def to_output_column(self) -> "TableOutputColumn":
        # TODO switch GBI ID's to tickers if needed, etc.
        return TableOutputColumn(name="", col_type=self.col_type, unit=self.unit)


@io_type
class Table(ComplexIOBase):
    title: Optional[str] = None
    columns: List[TableColumn]
    data: pd.DataFrame

    def to_gpt_input(self) -> str:
        return f"[Table with {len(self.data)} rows and {len(self.data.columns)} columns]"

    def convert_gbi_ids_to_stock_metadata_objects(
        self,
    ) -> pd.DataFrame:
        """
        Called before output conversion.
        """
        # TODO
        return self.data.copy(deep=True)

    def to_rich_output(self) -> Output:
        df = self.convert_gbi_ids_to_stock_metadata_objects()
        rows = df.values.tolist()
        # TODO handle highlighted first column, etc
        return TableOutput(
            title=self.title, columns=[col.to_output_column() for col in self.columns], rows=rows
        )

    @field_validator("data", mode="before")
    @classmethod
    def _deserializer(cls, data: Any) -> Any:
        if isinstance(data, dict):
            data = pd.DataFrame.from_dict(data)
        return data

    @field_serializer("data", mode="wrap")
    @classmethod
    def _field_serializer(cls, data: Any, dumper: Callable) -> Any:
        if isinstance(data, pd.DataFrame):
            data = data.to_dict()
        return dumper(data)


class StockMetadata(BaseModel):
    gbi_id: int
    symbol: str
    company_name: str


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
    rows: List[List[CellType]]
