from typing import List, Optional

from agent_service.io_type_utils import ComplexIOBase, io_type
from agent_service.io_types.output import Output
from agent_service.io_types.text import TextOutput
from agent_service.utils.boosted_pg import BoostedPG
from agent_service.utils.stock_metadata import get_stock_metadata


@io_type
class StockID(ComplexIOBase):
    gbi_id: int
    symbol: Optional[str]
    isin: str

    @staticmethod
    async def from_gbi_id_list(gbi_ids: List[int]) -> List["StockID"]:
        meta_dict = await get_stock_metadata(gbi_ids=gbi_ids)
        return [
            StockID(gbi_id=meta.gbi_id, symbol=meta.symbol, isin=meta.isin)
            for meta in meta_dict.values()
        ]

    async def to_rich_output(self, pg: BoostedPG) -> Output:
        return TextOutput(val=self.symbol or self.isin)
