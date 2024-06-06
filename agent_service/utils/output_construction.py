from typing import List

from agent_service.io_type_utils import ComplexIOBase, IOType
from agent_service.io_types.output import Output
from agent_service.io_types.stock import StockID
from agent_service.io_types.table import (
    StockTableColumn,
    Table,
    TableColumn,
    TableColumnMetadata,
)
from agent_service.io_types.text import Text
from agent_service.utils.async_utils import gather_with_concurrency
from agent_service.utils.boosted_pg import BoostedPG


def convert_list_of_stocks_to_table(stocks: List[StockID]) -> Table:
    # Each stock in the list has a history associated with it (e.g. "reasons"
    # for the text being present in this output). Every stock should have the
    # same number of these entries, so we can construct a table. Where each
    # stock is a row and each entry is a column. For example, the query "find
    # companies with mcap > X and Y in their earnings call" might return a table like this:
    # STOCK, MARKET CAP, CONNECTION TO 'Y'
    # AAPL,  123,        blah blah
    # GOOGL, 234,        blah blah

    entry_title_to_col_map = {}
    # First column contains all stocks
    for stock in stocks:
        for entry in stock.history:
            # Hack for backwards compat, TODO will remove
            if not entry.title:
                continue
            if entry.title not in entry_title_to_col_map:
                # create the column
                col = TableColumn(
                    metadata=TableColumnMetadata(
                        label=entry.title, col_type=entry.entry_type, unit=entry.unit
                    ),
                    data=[entry.explanation],
                )
                entry_title_to_col_map[entry.title] = col
            else:
                entry_title_to_col_map[entry.title].data.append(entry.explanation)

    columns: List[TableColumn] = [StockTableColumn(data=stocks)] + list(entry_title_to_col_map.values())  # type: ignore
    return Table(columns=columns)


async def get_output_from_io_type(val: IOType, pg: BoostedPG) -> Output:
    """
    This function accepts any IOType and returns a 'nice' output for the
    frontend. There are special cases that are handled specifically, otherwise
    we do our best.
    """
    if isinstance(val, list):
        if not val:
            # TODO probably improve this
            val = Text(val="No values found.")
        elif isinstance(val[0], StockID):
            val = convert_list_of_stocks_to_table(stocks=val)
        else:
            val = await gather_with_concurrency([get_output_from_io_type(v, pg=pg) for v in val])
    elif isinstance(val, dict):
        # Ideally we shouldn't ever hit this, tools should not output raw dicts.
        val = {key: await get_output_from_io_type(v, pg=pg) for key, v in val.items()}
    if not isinstance(val, ComplexIOBase):
        val = Text.from_io_type(val)
    val = await val.to_rich_output(pg)
    return val
