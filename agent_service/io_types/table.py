import datetime
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Union, cast

import numpy as np
import pandas as pd
from pydantic import BaseModel
from typing_extensions import Self

from agent_service.io_type_utils import (
    Citation,
    ComplexIOBase,
    IOType,
    PrimitiveType,
    ScoreOutput,
    TableColumnType,
    io_type,
)
from agent_service.io_types.graph import GraphType
from agent_service.io_types.output import CitationID, Output, OutputType
from agent_service.io_types.stock import StockID
from agent_service.utils.async_utils import gather_with_concurrency, to_awaitable
from agent_service.utils.boosted_pg import BoostedPG
from agent_service.utils.stock_metadata import StockMetadata

STOCK_ID_COL_NAME_DEFAULT = "Security"
SCORE_COL_NAME_DEFAULT = "Score"

MAX_DATAPOINTS_FOR_GPT = 50


@dataclass
class RowDescription:
    name: str
    explanation: Optional[str]


@io_type
class TableColumnMetadata(ComplexIOBase):
    label: PrimitiveType
    col_type: TableColumnType
    unit: Optional[str] = None
    row_descs: Optional[Dict[int, List[RowDescription]]] = None


@io_type
class TableColumn(ComplexIOBase):
    metadata: TableColumnMetadata
    data: List[Optional[IOType]]

    async def to_gpt_input(self, use_abbreviated_output: bool = True) -> str:
        if use_abbreviated_output:
            return f"<TableColumn of type '{self.metadata.col_type.value}' with {len(self.data)} datapoints>"

        data_to_show = self.data
        if len(self.data) > MAX_DATAPOINTS_FOR_GPT:
            threshold = MAX_DATAPOINTS_FOR_GPT // 2
            data_to_show = [*self.data[:threshold], "...", *self.data[threshold:]]
        items = await gather_with_concurrency(
            [
                item.to_gpt_input() if isinstance(item, ComplexIOBase) else to_awaitable(str(item))
                for item in data_to_show
            ]
        )
        col_str = ", ".join(items)
        return f"<Column '{self.metadata.label}' Data: {col_str}>"

    def is_data_identical(self) -> bool:
        if len(set(self.data)) == 1 and len(self.data) > 1:
            return True
        return False

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


def object_histories_to_columns(objects: List[ComplexIOBase]) -> List[TableColumn]:
    """
    Given a set of objects potentially with histories, aggregate those histories
    into columns and return them.
    """
    entry_title_to_col_map = {}
    score_col = None
    for obj_i, obj in enumerate(objects):
        # TODO remove this eventually, just needed until we make sure histories
        # never have duplicates.
        obj.dedup_history()
        # Special logic for scores. Create a single score column with aggregated scores.
        stock_score = ScoreOutput.from_entry_list(obj.history)
        if stock_score and not score_col:
            score_col = TableColumn(
                metadata=TableColumnMetadata(
                    label=SCORE_COL_NAME_DEFAULT, col_type=TableColumnType.SCORE
                ),
                data=[],
            )
        if stock_score and score_col:
            score_col.data.append(stock_score)
        elif not stock_score and score_col:
            # If we have a score column, but this object has no score, just add None
            score_col.data.append(None)

        # Now create a separate column for every entry type in the
        # history. Entry types are grouped by "title".
        for entry in obj.history:
            if not entry.title or not entry.explanation:
                continue
            if entry.title not in entry_title_to_col_map:
                # create the column
                col = TableColumn(
                    metadata=TableColumnMetadata(
                        label=entry.title, col_type=entry.entry_type, unit=entry.unit
                    ),
                    data=[None] * len(objects),
                )
                col.data[obj_i] = entry.explanation
                entry_title_to_col_map[entry.title] = col
            else:
                entry_title_to_col_map[entry.title].data[obj_i] = entry.explanation

    # Make sure the score column is the first one.
    if score_col:
        columns = [score_col] + list(entry_title_to_col_map.values())
    else:
        columns = list(entry_title_to_col_map.values())
    return columns


@io_type
class Table(ComplexIOBase):
    # Table creators can choose to prefer what kind of visualization to use in charting
    # tools. This is utilized by `make_generic_graph` to decide how to represent the graph
    # and is called when the user is not specific enough for the agent to have decided
    # to use one specific graph type over another.
    columns: List[TableColumn]
    prefer_graph_type: Optional[GraphType] = None

    def get_num_rows(self) -> int:
        if not self.columns:
            return 0
        return len(self.columns[0].data)

    async def to_gpt_input(self, use_abbreviated_output: bool = True) -> str:
        columns = await gather_with_concurrency([col.to_gpt_input() for col in self.columns])
        col_strings = "\n".join(columns)
        return f"<Table with {self.get_num_rows()} rows and columns:\n{col_strings}\n>\n"

    def get_stock_column(self) -> Optional[TableColumn]:
        for col in self.columns:
            if col.metadata.col_type == TableColumnType.STOCK:
                return col
        return None

    def get_score_column(self) -> Optional[TableColumn]:
        for col in self.columns:
            if col.metadata.col_type == TableColumnType.SCORE:
                return col
        return None

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
        expected_column_names_list = {col.label for col in columns}
        data = data.replace(np.nan, None)
        df_columns = [col for col in data.columns if col in expected_column_names_list]
        for col_meta, df_col in zip(columns, df_columns):
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
        fixed_cols = []
        output_cols = []
        is_first_col = True
        citations = []
        for col_ref in self.columns:
            # Make sure we don't mutate the original object
            col = deepcopy(col_ref)
            # A single input column might map to multiple output columns, so we
            # need a list here.
            additional_cols = []
            additional_output_cols: List[TableOutputColumn] = []
            # First handle citations
            col_citations = await Citation.resolve_all_citations(
                citations=self.get_all_citations(), db=pg
            )
            citations.extend(col_citations)

            output_col = col.to_output_column()
            # Next handle special transformations
            if col.metadata.col_type == TableColumnType.STOCK:
                # Get expanded columns for stock scores, etc.
                # We know col.data is a list of StockID's
                additional_cols = object_histories_to_columns(
                    objects=cast(List[ComplexIOBase], col.data)
                )
                additional_output_cols.extend((col.to_output_column() for col in additional_cols))
                # Map to StockMetadata
                col.data = [
                    (
                        StockMetadata(
                            gbi_id=val.gbi_id,
                            symbol=val.symbol,
                            isin=val.isin,
                            company_name=val.company_name,
                        )
                        if isinstance(val, StockID)
                        else val
                    )
                    for val in col.data
                ]
                if is_first_col:
                    # Automatically highlight the first column if it's a stock column
                    output_col.is_highlighted = True

            elif col.metadata.col_type.is_date_type() and is_first_col:
                # Automatically highlight the first column if it's a date column
                output_col.is_highlighted = True

            fixed_cols.append(col)
            fixed_cols.extend(additional_cols)

            # include references to the relevant citations
            output_col.citation_refs = [cit.id for cit in col_citations]
            output_cols.append(output_col)
            output_cols.extend(additional_output_cols)
            is_first_col = False

        # At this point, fixed_cols and output_cols match up with each
        # other. Create a table from fixed_cols so that we can easily convert to
        # a row-based schema.
        fixed_table = Table(columns=fixed_cols)
        df = fixed_table.to_df()
        df = df.replace(np.nan, None)
        # Make sure we sort before creating output (if necessary)
        score_col = fixed_table.get_score_column()
        if score_col:
            df = df.sort_values(by=str(score_col.metadata.label), ascending=False)
        rows = df.values.tolist()

        return TableOutput(title=title, columns=output_cols, rows=rows, citations=citations)

    def delete_data_before_start_date(self, start_date: datetime.date) -> None:
        date_column_idx = None

        for i, column in enumerate(self.columns):
            if column.metadata.col_type == TableColumnType.DATE:
                date_column_idx = i
                break

        if date_column_idx is None:
            return

        date_data = self.columns[date_column_idx].data
        to_delete_rows = set([i for i, date in enumerate(date_data) if date < start_date])  # type: ignore

        for column in self.columns:
            column.data = [
                datapoint for i, datapoint in enumerate(column.data) if i not in to_delete_rows
            ]

        return


CellType = Union[PrimitiveType, StockMetadata, ScoreOutput]


@io_type
class StockTable(Table):
    """
    Wrapper around a table, used really only for type hinting.
    """

    async def to_rich_output(self, pg: BoostedPG, title: str = "") -> Output:
        # TODO expand stock histories into new columns, aggregate scores, etc.
        return await super().to_rich_output(pg, title)

    def get_stocks(self) -> List[StockID]:
        for column in self.columns:
            if column.metadata.col_type == TableColumnType.STOCK:
                stocks: List[StockID] = column.data  # type: ignore
                return stocks
        return []

    def get_values_for_stocks(self) -> Dict[str, Any]:
        for column in self.columns:
            if column.metadata.col_type == TableColumnType.STOCK:
                stock_label = column.metadata.label
        df = self.to_df()
        df.set_index(keys=stock_label, inplace=True)
        df_dict: Dict[str, Any] = df.to_dict("index")  # type: ignore
        return df_dict


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
    # Refers back to citations in the base TableOutput object.
    citation_refs: List[CitationID] = []


class TableOutput(Output):
    output_type: Literal[OutputType.TABLE] = OutputType.TABLE
    title: str = ""
    columns: List[TableOutputColumn] = []
    rows: List[List[Optional[CellType]]]
