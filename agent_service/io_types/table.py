import datetime
import logging
from copy import deepcopy
from dataclasses import dataclass
from typing import Annotated, Any, Dict, List, Literal, Optional, Union, cast

import numpy as np
import pandas as pd
from pydantic import BaseModel, field_serializer
from typing_extensions import Self

from agent_service.io_type_utils import (
    Citation,
    ComplexIOBase,
    HistoryEntry,
    IOType,
    PrimitiveType,
    ScoreOutput,
    TableColumnType,
    io_type,
)
from agent_service.io_types.citations import CitationID
from agent_service.io_types.graph import GraphType
from agent_service.io_types.output import Output, OutputType
from agent_service.io_types.stock import StockID
from agent_service.io_types.text import Text, TextOutput
from agent_service.utils.async_utils import gather_with_concurrency, to_awaitable
from agent_service.utils.boosted_pg import BoostedPG
from agent_service.utils.stock_metadata import StockMetadata, get_stock_metadata

STOCK_ID_COL_NAME_DEFAULT = "Security"
SCORE_COL_NAME_DEFAULT = "Criteria Match"

MAX_DATAPOINTS_FOR_GPT = 50

logger = logging.getLogger(__name__)


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
    data_src: List[str] = []

    @classmethod
    def to_gpt_schema(cls) -> Dict[str, str]:
        schema = {
            "label": "Union[str, datetime.date, datetime.datetime]",
            "col_type": "str",
            "unit": "Optional[str]",
            "data_src": "List[str]",
        }
        return schema


@io_type
class TableColumn(ComplexIOBase):
    metadata: TableColumnMetadata
    data: List[Optional[IOType] | ScoreOutput]

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
    data: List[Optional[datetime.date]]  # type: ignore


@io_type
class DatetimeTableColumn(TableColumn):
    metadata: TableColumnMetadata = TableColumnMetadata(
        label="Date & Time", col_type=TableColumnType.DATETIME
    )
    data: List[Optional[datetime.datetime]]  # type: ignore


def object_histories_to_columns(
    objects: List[ComplexIOBase],
) -> List[TableColumn]:
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
                data=[None] * len(objects),
            )
        if score_col:
            score_col.data[obj_i] = stock_score  # type: ignore

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
                entry_title_to_col_map[entry.title] = col
            current_col = entry_title_to_col_map[entry.title]
            current_col.data[obj_i] = entry.explanation
            if (
                isinstance(entry.explanation, str)
                and entry.entry_type == TableColumnType.STRING
                and entry.citations
            ):
                # If the explanation is a string and has citations (e.g. the
                # output from a filter tool), store as a Text object in the
                # table so that citations can be correctly resolved later.
                current_col.data[obj_i] = Text(
                    val=entry.explanation, history=[HistoryEntry(citations=entry.citations)]
                )

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

    # If set to true, will reduce large tables when serializing
    should_subsample_large_table: bool = False
    table_was_reduced: bool = False

    @field_serializer("columns", when_used="json")
    def _subsample_large_table(self, columns: List[TableColumn]) -> List[TableColumn]:
        """
        When we're serializing to json, for massive tables we want to cut down
        on the number of rows so that we don't cause issues.
        """
        MAX_ROW_COUNT = 100000

        num_rows = self.get_num_rows()
        if not self.should_subsample_large_table or num_rows <= MAX_ROW_COUNT:
            return columns

        date_col = self.get_date_column()
        stock_col = self.get_stock_column()
        if not date_col or not stock_col:
            # Right now we only care about timeseries data, that's really the
            # only time this ever happens anyway.
            return columns

        logger.info(f"Reducing size of table with {num_rows} rows for serialization...")
        df = self.to_df(dates_to_timestamps=True)
        stock_col_name = str(stock_col.metadata.label)
        keep_every_nth = max(2, num_rows // MAX_ROW_COUNT)
        reduced = df.groupby(stock_col_name, as_index=False).apply(
            lambda x: x.iloc[::keep_every_nth]
        )
        new_table = Table.from_df_and_cols(columns=[col.metadata for col in columns], data=reduced)
        self.table_was_reduced = True
        return new_table.columns

    def get_num_rows(self) -> int:
        if not self.columns:
            return 0
        return len(self.columns[0].data)

    async def to_gpt_input(self, use_abbreviated_output: bool = True) -> str:
        columns = await gather_with_concurrency([col.to_gpt_input() for col in self.columns])
        col_strings = "\n".join(columns)
        return f"<Table with {self.get_num_rows()} rows and columns:\n{col_strings}\n>\n"

    def get_first_col_of_type(self, col_type: TableColumnType) -> Optional[TableColumn]:
        for col in self.columns:
            if col.metadata.col_type == col_type:
                return col
        return None

    def get_stock_column(self) -> Optional[TableColumn]:
        return self.get_first_col_of_type(TableColumnType.STOCK)

    def get_date_column(self) -> Optional[TableColumn]:
        for col in self.columns:
            if col.metadata.col_type.is_date_type():
                return col
        return None

    @staticmethod
    def _convert_to_start_date(
        df: pd.DataFrame, time_column: Any, current_type: TableColumnType
    ) -> pd.DataFrame:
        """Convert the time column to the start date of each period for uniformity."""
        if current_type == TableColumnType.YEAR:
            df[time_column] = pd.to_datetime(df[time_column].astype(str) + "-01-01")

        elif current_type == TableColumnType.QUARTER:
            year = df[time_column].astype(str).str[:4]
            quarter = df[time_column].astype(str).str[-1].astype(int)
            df[time_column] = pd.to_datetime(
                year + "-" + ((quarter - 1) * 3 + 1).astype(str) + "-01"
            )

        elif current_type == TableColumnType.MONTH:
            df[time_column] = pd.to_datetime(df[time_column].astype(str) + "-01")

        return df

    @staticmethod
    def _resample_with_group_by(
        df: pd.DataFrame, time_col: Any, group_col: Any, resample_val: str = "D"
    ) -> pd.DataFrame:
        expanded_dfs = []
        # Step 2: Group by the stock column
        for _, stock_df in df.groupby(group_col):
            # Set the time column as the index to make it easier to resample
            stock_df = stock_df.set_index(time_col)

            # Step 3: resample for each stock
            resampled_df = stock_df.resample(resample_val).ffill()

            # Step 4: Reset index so that time is a column again
            resampled_df = resampled_df.reset_index()

            # Add the expanded stock data to our list
            expanded_dfs.append(resampled_df)

        # Concatenate all expanded dataframes back into a single DataFrame
        df = pd.concat(expanded_dfs, ignore_index=True)
        return df

    @staticmethod
    def _expand_to_daily_with_ffill(
        df: pd.DataFrame,
        time_column: Any,
        current_type: TableColumnType,
        stock_column: Optional[str] = None,
    ) -> pd.DataFrame:
        """Expands the time column to daily frequency with forward-fill."""
        df = Table._convert_to_start_date(df, time_column, current_type)

        if not stock_column:
            df = df.set_index(time_column)
            full_date_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq="D")
            df = df.reindex(full_date_range).ffill().reset_index()
            df = df.rename(columns={"index": time_column})
        else:
            df = Table._resample_with_group_by(
                df, time_col=time_column, group_col=stock_column, resample_val="D"
            )

        return df

    @staticmethod
    def _aggregate_to_target_granularity(
        df: pd.DataFrame,
        time_column: Any,
        target_type: TableColumnType,
        target_col_name: str,
        stock_column: Optional[str] = None,
    ) -> pd.DataFrame:
        """Aggregate a daily DataFrame back to the specified target granularity."""
        df[time_column] = pd.to_datetime(df[time_column])  # Ensure the column is in datetime format

        if target_type == TableColumnType.MONTH:
            # Resample to monthly, taking the last available value for each month
            if stock_column:
                df = Table._resample_with_group_by(
                    df, time_col=time_column, group_col=stock_column, resample_val="M"
                )
            else:
                df = df.set_index(time_column).resample("M").ffill().reset_index()
            df[time_column] = df[time_column].dt.strftime("%Y-%m")

        elif target_type == TableColumnType.QUARTER:
            # Resample to quarterly, taking the last available value for each quarter
            if stock_column:
                df = Table._resample_with_group_by(
                    df, time_col=time_column, group_col=stock_column, resample_val="Q"
                )
            else:
                df = df.set_index(time_column).resample("Q").ffill().reset_index()
            df[time_column] = (
                df[time_column].dt.to_period("Q").dt.strftime("%YQ%q")
            )  # Format to e.g., '2024Q1'

        elif target_type == TableColumnType.YEAR:
            # Resample to yearly, taking the last available value for each year
            if stock_column:
                df = Table._resample_with_group_by(
                    df, time_col=time_column, group_col=stock_column, resample_val="Y"
                )
            else:
                df = df.set_index(time_column).resample("Y").ffill().reset_index()
            df[time_column] = df[time_column].dt.year  # Keep only the year
            df[time_column] = df[time_column].astype(str)

        df = df.rename(columns={time_column: target_col_name})
        return df

    def convert_table_to_time_granularity(self, target_col: TableColumnMetadata) -> "Table":
        target_type = target_col.col_type
        if not TableColumnType.is_date_type(target_type):
            return self
        date_col = self.get_date_column()
        stock_col_name = None
        stock_col = self.get_stock_column()
        if not date_col or date_col.metadata.col_type == target_type:
            return self
        if stock_col:
            stock_col_name = str(stock_col.metadata.label)

        current_type = date_col.metadata.col_type
        time_column = date_col.metadata.label

        # First, convert to dates
        df = self.to_df()

        df = self._expand_to_daily_with_ffill(
            df, time_column=time_column, current_type=current_type, stock_column=stock_col_name
        )
        df = self._aggregate_to_target_granularity(
            df,
            time_column=time_column,
            target_type=target_type,
            stock_column=stock_col_name,
            target_col_name=str(target_col.label),
        )

        date_col.metadata.label = target_col.label
        date_col.metadata.col_type = target_type
        new_table = self.from_df_and_cols(columns=[col.metadata for col in self.columns], data=df)
        if isinstance(self, StockTable):
            return StockTable(
                history=deepcopy(self.history),
                columns=new_table.columns,
                prefer_graph_type=self.prefer_graph_type,
                should_subsample_large_table=self.should_subsample_large_table,
                table_was_reduced=self.table_was_reduced,
            )
        return Table(
            history=deepcopy(self.history),
            columns=new_table.columns,
            prefer_graph_type=self.prefer_graph_type,
            should_subsample_large_table=self.should_subsample_large_table,
            table_was_reduced=self.table_was_reduced,
        )

    def get_score_column(self) -> Optional[TableColumn]:
        return self.get_first_col_of_type(TableColumnType.SCORE)

    def dedup_columns(self) -> None:
        col_name_data_map: Dict[PrimitiveType, List[Optional[IOType]]] = {}
        new_cols = []
        for col in self.columns:
            if (
                col.metadata.label in col_name_data_map
                and col.data == col_name_data_map[col.metadata.label]
            ):
                continue
            col_name_data_map[col.metadata.label] = col.data
            new_cols.append(col)
        self.columns = new_cols

    def to_df(
        self,
        stocks_as_tickers_only: bool = False,
        stocks_as_hashables: bool = False,
        dates_to_timestamps: bool = False,
    ) -> pd.DataFrame:
        data = {}
        for col in self.columns:
            data[col.metadata.label] = col.data
            if dates_to_timestamps and col.metadata.col_type in (
                TableColumnType.DATE,
                TableColumnType.DATETIME,
            ):
                data[col.metadata.label] = pd.to_datetime(col.data)  # type: ignore
            elif isinstance(col, StockTableColumn):
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
        ignore_extra_cols: bool = False,
    ) -> Self:
        def _val_to_dt(
            val: Any, convert_to_date: bool = False
        ) -> Optional[Union[datetime.date, datetime.datetime]]:
            if val is None:
                return None
            if isinstance(val, pd.Timestamp):
                val = val.to_pydatetime() if not convert_to_date else val.date()
            return val

        if data.empty:
            data = pd.DataFrame({col.label: [] for col in columns})
        out_columns: List[TableColumn] = []
        data = data.replace({np.nan: None})
        # this was accidentally? enforcing the column labels to line up with
        # the physical column layout, even though we grab the data by column name/label instead
        for col_meta in columns:
            df_col = col_meta.label
            if df_col not in data.columns:
                if ignore_extra_cols:
                    continue
                raise RuntimeError(
                    f"Requested label: {df_col=} not in dataframe cols: {data.columns}"
                )

            if col_meta.col_type == TableColumnType.DATE:
                vals = [_val_to_dt(ts, convert_to_date=True) for ts in data[df_col].to_list()]
                out_columns.append(DateTableColumn(metadata=col_meta, data=vals))
            elif col_meta.col_type == TableColumnType.DATETIME:
                vals = [_val_to_dt(ts) for ts in data[df_col].to_list()]
                out_columns.append(DatetimeTableColumn(metadata=col_meta, data=vals))  # type: ignore
            elif col_meta.col_type == TableColumnType.STOCK:
                stocks = data[df_col].to_list()
                if stocks_are_hashable_objs:
                    stocks = [StockID.from_hashable(stock) for stock in stocks]
                out_columns.append(StockTableColumn(metadata=col_meta, data=stocks))
            else:
                out_columns.append(TableColumn(metadata=col_meta, data=data[df_col].to_list()))

        return cls(columns=out_columns)

    async def to_rich_output(self, pg: BoostedPG, title: str = "") -> Output:
        fixed_cols: List[TableColumn] = []
        output_cols = []
        is_first_col = True
        table_citations = await Citation.resolve_all_citations(
            citations=self.get_all_citations(), db=pg
        )
        for col_ref in self.columns:
            # Make sure we don't mutate the original object
            col = deepcopy(col_ref)
            # A single input column might map to multiple output columns, so we
            # need a list here.
            additional_cols = []
            additional_output_cols: List[TableOutputColumn] = []
            # First handle citations
            col_citations = await Citation.resolve_all_citations(
                citations=col.get_all_citations(), db=pg
            )
            table_citations.extend(col_citations)

            output_col = col.to_output_column()
            # Next handle special transformations
            if col.metadata.col_type == TableColumnType.STOCK:
                # Get expanded columns for stock scores, etc.
                # We know col.data is a list of StockID's
                additional_cols = object_histories_to_columns(
                    objects=cast(List[ComplexIOBase], col.data)
                )
                additional_output_cols.extend((col.to_output_column() for col in additional_cols))
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

        # Now do a final step of mapping, especially for citations
        final_cols = []
        for fixed_col in fixed_cols:
            if fixed_col.metadata.col_type == TableColumnType.STRING:
                output_texts = await Text.multi_text_rich_output(
                    pg=pg,
                    texts=[
                        text if isinstance(text, Text) else Text(val=str(text))
                        for text in fixed_col.data
                    ],
                )
                new_data = []
                for text in output_texts:
                    if isinstance(text, TextOutput):
                        new_data.append(text.val)
                        table_citations.extend(text.citations)
                    else:
                        new_data.append(text)
                fixed_col.data = new_data  # type: ignore
            final_cols.append(fixed_col)

        # At this point, final_cols and output_cols match up with each
        # other. Create a table from fixed_cols so that we can easily convert to
        # a row-based schema.
        fixed_table = Table(columns=final_cols)
        fixed_table.dedup_columns()
        all_gbi_ids = {
            stock.gbi_id
            for col in fixed_table.columns
            for stock in col.data
            if isinstance(stock, StockID)
        }
        metadata_map = await get_stock_metadata(gbi_ids=list(all_gbi_ids), pg=pg)
        df = fixed_table.to_df()
        # Make sure we sort before creating output (if necessary)
        score_col = fixed_table.get_score_column()
        if score_col:
            df = df.sort_values(by=str(score_col.metadata.label), ascending=False)
        df = df.applymap(lambda val: metadata_map[val.gbi_id] if isinstance(val, StockID) else val)
        df = df.replace({np.nan: None})
        rows = df.values.tolist()

        if rows and len(output_cols) > len(rows[0]):
            # Dedup here just like we dedped the table itself with dedup_columns
            final_output_cols = []
            output_cols_set = set()
            for out_col in output_cols:
                if out_col in output_cols_set:
                    continue
                output_cols_set.add(out_col)
                final_output_cols.append(out_col)
            output_cols = final_output_cols
        return TableOutput(title=title, columns=output_cols, rows=rows, citations=table_citations)

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

    def delete_date_column(self) -> None:
        date_column_idx = None

        for i, column in enumerate(self.columns):
            if (
                column.metadata.col_type == TableColumnType.DATE
                or column.metadata.col_type == TableColumnType.QUARTER
            ):
                date_column_idx = i
                break

        if date_column_idx is None:
            return

        self.columns.pop(i)


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

    def get_values_for_stocks(self) -> Dict[StockID, Any]:
        for column in self.columns:
            if column.metadata.col_type == TableColumnType.STOCK:
                stock_label = column.metadata.label
        df = self.to_df()
        df.set_index(keys=stock_label, inplace=True)
        df_dict: Dict[StockID, Any] = df.to_dict("index")  # type: ignore
        return df_dict

    def add_task_id_to_history(self, task_id: str) -> None:
        for column in self.columns:
            if column.metadata.col_type == TableColumnType.STOCK:
                stocks: List[StockID] = column.data  # type: ignore
                column.data = [
                    stock.inject_history_entry(HistoryEntry(task_id=task_id)) for stock in stocks
                ]


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

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, TableOutputColumn):
            return False
        return (
            self.name == other.name
            and self.col_type == other.col_type
            and self.unit == other.unit
            and self.citation_refs == other.citation_refs
        )

    def __hash__(self) -> int:
        return hash((self.name, self.col_type, self.unit, tuple(self.citation_refs)))


# We need to redefine and reorder PrimitiveType to correctly parse
# strings. Otherwise, a string representing a year ("2022") will be converted to
# some weird datetime. This only impacts cases where we serialize and
# deserialize TableOutput specifically, which is really only done when caching.
CellPrimitiveType = Annotated[
    Union[
        str,
        int,
        bool,
        float,
        datetime.date,
        datetime.datetime,
    ],
    "left_to_right",
]
OutputCellType = Union[StockMetadata, ScoreOutput, CellPrimitiveType]


class TableOutput(Output):
    output_type: Literal[OutputType.TABLE] = OutputType.TABLE
    title: str = ""
    columns: List[TableOutputColumn] = []
    rows: List[List[Optional[OutputCellType]]]
