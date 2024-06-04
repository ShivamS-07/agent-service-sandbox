from typing import List

from agent_service.io_type_utils import ComplexIOBase, IOType
from agent_service.io_types.output import Output
from agent_service.io_types.stock import StockID
from agent_service.io_types.text import Text
from agent_service.utils.async_utils import gather_with_concurrency
from agent_service.utils.boosted_pg import BoostedPG


def convert_list_of_stocks_to_text(stocks: List[StockID]) -> Text:
    data_str = "\n".join(
        [f"- **{stock.symbol or stock.isin}** - {stock.get_history_string()}" for stock in stocks]
    )
    return Text(val=data_str)


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
            val = convert_list_of_stocks_to_text(stocks=val)
        else:
            val = await gather_with_concurrency([get_output_from_io_type(v, pg=pg) for v in val])
    elif isinstance(val, dict):
        # Ideally we shouldn't ever hit this, tools should not output raw dicts.
        val = {key: await get_output_from_io_type(v, pg=pg) for key, v in val.items()}
    if not isinstance(val, ComplexIOBase):
        val = Text.from_io_type(val)
    val = await val.to_rich_output(pg)
    return val
