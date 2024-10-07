from typing import List

from agent_service.io_type_utils import ComplexIOBase, IOType, io_type
from agent_service.io_types.stock import StockID
from agent_service.utils.output_utils.output_construction import PreparedOutput


@io_type
class StockGroup(ComplexIOBase):
    name: str
    stocks: List[StockID]


@io_type
class StockGroups(ComplexIOBase):
    stock_groups: List[StockGroup]

    async def split_into_components(self) -> List[IOType]:
        return [
            PreparedOutput(val=stock_group.stocks, title=stock_group.name)
            for stock_group in self.stock_groups
        ]
