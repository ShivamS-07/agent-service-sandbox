import enum
from typing import Any, Callable, List, Literal, Optional, Union

import numpy as np
import pandas as pd
from pydantic import BaseModel
from pydantic.functional_serializers import field_serializer
from pydantic.functional_validators import field_validator

from agent_service.io_type_utils import (
    ComplexIOBase,
    PrimitiveType,
    io_type,
    load_io_type_dict,
)
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

        output_cols = []
        is_first_col = True
        for df_col, col in zip(df.columns, self.columns):
            output_col = col.to_output_column()
            if col.col_type == TableColumnType.STOCK:
                if is_first_col:
                    # Map to symbol or isin
                    df[df_col] = df[df_col].map(
                        lambda val: (val.symbol or val.isin) if isinstance(val, StockID) else val
                    )
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
        def _apply_func(val: Any) -> Any:
            # This allows us to store ANY arbitrary IOType in dataframes, and
            # they will be serialized and deserialized automatically.
            if isinstance(val, dict):
                try:
                    return load_io_type_dict(val)
                except Exception:
                    pass
            return val

        if isinstance(data, dict):
            data = pd.DataFrame.from_dict(data)
            # No index
            data = data.reset_index(drop=True)
            data = data.applymap(_apply_func)
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
