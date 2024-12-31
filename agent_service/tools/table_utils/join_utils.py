import enum
from itertools import zip_longest
from typing import Optional, Tuple

from agent_service.io_type_utils import TableColumnType
from agent_service.io_types.stock import StockID
from agent_service.io_types.table import Table, TableColumn, TableColumnMetadata

DEFAULT_GROUP_COL_NAME = "Group"
DEFAULT_VALUE_COL_NAME = "Value"


def remove_group_col_with_one_member(table: Table, group_col: TableColumn) -> Table:
    """
    For a stock timeseries table with just one stock, we can remove the stock
    column and change the label of the data column. This is useful for joining a
    stock table with one stock to a non-stock table (e.g. FED data).
    """
    date_col = table.get_date_column()
    if not date_col:
        return table

    new_cols = [date_col]
    for col in table.columns:
        if col in (group_col, date_col):
            continue
        tag = (
            group_col.data[0].symbol
            if isinstance(group_col.data[0], StockID)
            else group_col.data[0]
        )
        col.metadata.label = f"{tag} {col.metadata.label}"
        new_cols.append(col)

    return Table(
        history=table.history,
        title=table.title,
        columns=new_cols,
        prefer_graph_type=table.prefer_graph_type,
    )


def expand_dates_across_tables(first: Table, second: Table) -> Tuple[Table, Table]:
    first_date_col = first.get_date_column()
    second_date_col = second.get_date_column()

    if (
        first_date_col
        and second_date_col
        and first_date_col.metadata.col_type != second_date_col.metadata.col_type
    ):
        first_date_col_type = first_date_col.metadata.col_type
        second_date_col_type = second_date_col.metadata.col_type
        granularity_hierarchy = {
            TableColumnType.YEAR: 1,
            TableColumnType.QUARTER: 2,
            TableColumnType.MONTH: 3,
            TableColumnType.DATE: 4,
            TableColumnType.DATETIME: 4,
        }
        if (
            first_date_col_type in granularity_hierarchy
            and second_date_col_type in granularity_hierarchy
        ):
            if (
                granularity_hierarchy[first_date_col_type]
                < granularity_hierarchy[second_date_col_type]
            ):
                # This means the first table is less granular, need to convert
                # it to the second table's granularity
                first = first.convert_table_to_time_granularity(second_date_col.metadata)
            elif (
                granularity_hierarchy[first_date_col_type]
                > granularity_hierarchy[second_date_col_type]
            ):
                second = second.convert_table_to_time_granularity(first_date_col.metadata)

    return (first, second)


class TableType(enum.Enum):
    STOCK_TIMESERIES = 1
    STRING_TIMESERIES = 2
    TIMESERIES_ONLY = 3
    STOCK_ONLY = 4
    STRING_ONLY = 5
    OTHER = -1

    @staticmethod
    def get_from_table(table: Table) -> "TableType":
        stock_col = table.get_stock_column()
        date_col = table.get_date_column()
        string_col = table.get_first_col_of_type(TableColumnType.STRING)

        if date_col:
            if stock_col:
                return TableType.STOCK_TIMESERIES
            elif string_col:
                return TableType.STRING_TIMESERIES
            else:
                return TableType.TIMESERIES_ONLY
        else:
            if stock_col:
                return TableType.STOCK_ONLY
            elif string_col:
                return TableType.STRING_ONLY

        return TableType.OTHER


def _stock_col_to_strings(col: TableColumn, rename_to: Optional[str] = None) -> None:
    if col.metadata.col_type != TableColumnType.STOCK:
        return
    col.metadata.col_type = TableColumnType.STRING
    if rename_to:
        col.metadata.label = rename_to
    col.data = [
        val.symbol or val.company_name if isinstance(val, StockID) else val for val in col.data
    ]


def _expand_timeseries_only_to_grouped(cols: list[TableColumn]) -> list[TableColumn]:
    new_cols = []
    seen_date_col = False
    for col in cols:
        if TableColumnType.is_date_type(col.metadata.col_type):
            seen_date_col = True
            new_cols.append(col)
        elif seen_date_col:
            # Add a new column representing the group, using the label of the
            # value column as the group name.
            new_col = TableColumn(
                metadata=TableColumnMetadata(
                    label=DEFAULT_GROUP_COL_NAME, col_type=TableColumnType.STRING
                ),
                data=[str(col.metadata.label)] * len(col.data),  # type: ignore
            )
            new_cols.append(new_col)
            new_cols.append(col)

        else:
            new_cols.append(col)

    return new_cols


def preprocess_heterogeneous_tables_before_joining(
    first: Table, second: Table
) -> Tuple[Table, Table]:
    # When it comes to joining, there are a few categories of tables:
    # timeseries + value
    # timeseries + stock/string + value
    # stock/string + value

    # We need to consider joins across all these various subtypes, and try to handle
    # them correctly. This will make the planner's life easier.
    first_group_col = first.get_stock_column()
    second_group_col = second.get_stock_column()
    first_date_col = first.get_date_column()
    second_date_col = second.get_date_column()

    # These cases are handled in preprocess_homogeneous_tables_before_joining
    if first_group_col and second_group_col and first_date_col and second_date_col:
        return first, second
    if first_group_col and second_group_col and not first_date_col and not second_date_col:
        return first, second
    if not first_group_col and not second_group_col and first_date_col and second_date_col:
        return first, second
    if not first_group_col and not second_group_col and not first_date_col and not second_date_col:
        return first, second

    first_table_type = TableType.get_from_table(first)
    second_table_type = TableType.get_from_table(second)

    # In the below cases, we want to add an extra column to the timeseries only
    # table. For example, if the first table has a stock timeseries and the
    # second table has a timeseries only representing some macro data, we want
    # to be able to merge them.
    if (
        first_table_type
        in (
            TableType.STOCK_TIMESERIES,
            TableType.STRING_TIMESERIES,
        )
        and second_table_type == TableType.TIMESERIES_ONLY
    ):
        if first_group_col and len(set(first_group_col.data)) == 1:  # type: ignore
            # If there is only one stock in the stock timeseries, we can just
            # remove it and join directly to the non-stock timeseries.
            first = remove_group_col_with_one_member(first, first_group_col)
            first_table_type = TableType.TIMESERIES_ONLY
        else:
            new_cols = _expand_timeseries_only_to_grouped(second.columns)
            second.columns = new_cols
            second_table_type = TableType.STRING_TIMESERIES

    elif (
        second_table_type
        in (
            TableType.STOCK_TIMESERIES,
            TableType.STRING_TIMESERIES,
        )
        and first_table_type == TableType.TIMESERIES_ONLY
    ):
        if second_group_col and len(set(second_group_col.data)) == 1:  # type: ignore
            second = remove_group_col_with_one_member(second, second_group_col)
            second_table_type = TableType.TIMESERIES_ONLY
        else:
            new_cols = _expand_timeseries_only_to_grouped(first.columns)
            first.columns = new_cols
            first_table_type = TableType.STRING_TIMESERIES

    first_is_string_table = first_table_type in (TableType.STRING_ONLY, TableType.STRING_TIMESERIES)
    second_is_string_table = second_table_type in (
        TableType.STRING_ONLY,
        TableType.STRING_TIMESERIES,
    )

    if (first_is_string_table and second_group_col) or (second_is_string_table and first_group_col):
        # In this case we have a string-based table and a stock-based table, so
        # convert the stock col to strings. We also want to rename the column,
        # since it no longer only will contain stocks.
        if first_is_string_table:
            string_table = first
            stock_table = second
        else:
            string_table = second
            stock_table = first

        for col in stock_table.columns:
            _stock_col_to_strings(col, rename_to=DEFAULT_GROUP_COL_NAME)
        for col in string_table.columns:
            if col.metadata.col_type == TableColumnType.STRING:
                col.metadata.label = DEFAULT_GROUP_COL_NAME
                break

    # NOTE: This is a bit hacky :-/
    # One last special case. If we have two tables with disjoint indexes and
    # value columns of the same type, consolidate the columns into the same
    # names. This is very helpful if we're e.g. comparing portfolio performance
    # to stock performance.
    if (
        len(first.columns) == len(second.columns)
        and first_table_type == second_table_type
        and first_table_type in (TableType.STRING_ONLY, TableType.STRING_TIMESERIES)
    ):
        first_group_col = first.get_first_col_of_type(TableColumnType.STRING)
        second_group_col = second.get_first_col_of_type(TableColumnType.STRING)
        first_date_col = first.get_date_column()
        second_date_col = second.get_date_column()
        first_indexes = {
            (date, group)
            for date, group in zip_longest(
                first_date_col.data if first_date_col else [],
                first_group_col.data if first_group_col else [],
            )
        }
        second_indexes = {
            (date, group)
            for date, group in zip_longest(
                second_date_col.data if second_date_col else [],
                second_group_col.data if second_group_col else [],
            )
        }
        if (
            len(first_indexes.intersection(second_indexes)) == 0
            and first.columns[-1].metadata.col_type == second.columns[-1].metadata.col_type
            and first_group_col
            and second_group_col
            and first_group_col.metadata.label == second_group_col.metadata.label
            and first_group_col.metadata.unit == second_group_col.metadata.unit
        ):
            # Rename data columns so that they can be joined later
            old_first_data_label = first.columns[-1].metadata.label
            old_second_data_label = second.columns[-1].metadata.label
            first.columns[-1].metadata.label = DEFAULT_VALUE_COL_NAME
            second.columns[-1].metadata.label = DEFAULT_VALUE_COL_NAME

            # Rename the group cols to be completely clear
            first_group_col.data = [
                f"{group} - {old_first_data_label}" if old_first_data_label not in group else group  # type: ignore
                for group in first_group_col.data
            ]
            second_group_col.data = [
                f"{group} - {old_second_data_label}"
                if old_second_data_label not in group  # type: ignore
                else group
                for group in second_group_col.data
            ]

    return first, second


def check_for_index_overlap(first: Table, second: Table) -> bool:
    """
    If we're calling this, we know 100% that the tables have the same column
    metadatas. We just need to check for overlap in index column values.
    """
    first_group_col = first.get_stock_column() or first.get_first_col_of_type(
        TableColumnType.STRING
    )
    second_group_col = second.get_stock_column() or second.get_first_col_of_type(
        TableColumnType.STRING
    )
    if not first_group_col or not second_group_col:
        return False

    if len(set(first_group_col.data).intersection(set(second_group_col.data))) > 0:
        return True

    return False


def add_extra_group_cols(table: Table, table_name: str) -> Table:
    group_col = table.get_stock_column() or table.get_first_col_of_type(TableColumnType.STRING)
    date_col = table.get_date_column()

    # This is VERY hacky, and is really just a last resort to prevent weirdness
    new_cols = []
    new_col_added = False
    for col in table.columns:
        if col in (group_col, date_col):
            new_cols.append(col)
            continue
        if not new_col_added:
            new_cols.append(
                TableColumn(
                    data=[table_name] * len(col.data),
                    metadata=TableColumnMetadata(
                        label=DEFAULT_GROUP_COL_NAME, col_type=TableColumnType.STRING
                    ),
                )
            )
            new_col_added = True
        new_cols.append(col)

    table.columns = new_cols

    return table
