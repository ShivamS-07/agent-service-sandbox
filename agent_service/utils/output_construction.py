from collections import defaultdict
from typing import Dict, List

from agent_service.io_type_utils import ComplexIOBase, IOType, io_type
from agent_service.io_types.output import Output
from agent_service.io_types.stock import StockID
from agent_service.io_types.table import StockTableColumn, Table, TableColumn
from agent_service.io_types.text import StockText, Text
from agent_service.utils.async_utils import gather_with_concurrency
from agent_service.utils.boosted_pg import BoostedPG


def prepare_list_of_stocks(stocks: List[StockID]) -> Table:
    columns: List[TableColumn] = [StockTableColumn(data=stocks)]
    return Table(columns=columns)


def prepare_list_of_stock_texts(texts: List[StockText]) -> Text:
    # Maps stocks to strings
    stock_text_map: Dict[StockID, List[str]] = defaultdict(list)
    text_strs = StockText.get_all_strs(texts)
    for text, text_str in zip(texts, text_strs):
        if text.stock_id:
            stock_text_map[text.stock_id].append(text_str)

    # Now for each stock, construct a markdown string and stick it in a list
    stock_strings = []
    for stock, text_list in stock_text_map.items():
        # Add the symbol
        mdown_strs = [f"{stock.to_markdown_string()}"]
        # Add the history info
        mdown_strs.append(stock.history_to_str())
        # Add the text info
        mdown_strs.extend((f"- {t}" for t in text_list))
        stock_strings.append("\n".join(mdown_strs))

    # Merge the markdown strings together
    return Text(val="\n\n".join(stock_strings))


async def get_output_from_io_type(val: IOType, pg: BoostedPG, title: str = "") -> Output:
    """
    This function accepts any IOType and returns a 'nice' output for the
    frontend. There are special cases that are handled specifically, otherwise
    we do our best.
    """
    # Be a bit sneaky, check if there's a title attached to the object
    if not title and hasattr(val, "title"):
        title = str(val.title)

    if isinstance(val, list):
        if not val:
            # TODO probably improve this
            val = Text(val="No values found.")
        elif isinstance(val[0], StockID):
            val = prepare_list_of_stocks(stocks=val)
        elif isinstance(val[0], StockText):
            val = prepare_list_of_stock_texts(texts=val)
        else:
            val = await gather_with_concurrency([get_output_from_io_type(v, pg=pg) for v in val])
    elif isinstance(val, dict):
        # Ideally we shouldn't ever hit this, tools should not output raw dicts.
        val = {key: await get_output_from_io_type(v, pg=pg) for key, v in val.items()}
    if not isinstance(val, ComplexIOBase):
        val = Text.from_io_type(val)
    val = await val.to_rich_output(pg, title=title)
    return val


@io_type
class PreparedOutput(ComplexIOBase):
    """Wrapper type around ANY IO type that includes a title."""

    val: IOType
    title: str = ""

    async def to_rich_output(self, pg: BoostedPG, title: str = "") -> Output:
        return await get_output_from_io_type(val=self.val, pg=pg, title=self.title)


# TODO remove me, for backwards compat
@io_type
class TitledIOType(PreparedOutput):
    pass
