from __future__ import annotations

from typing import Any, List, Optional

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

    def __hash__(self) -> int:
        return self.gbi_id

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, StockID):
            return self.gbi_id == other.gbi_id
        return False

    @staticmethod
    async def from_gbi_id_list(gbi_ids: List[int]) -> List["StockID"]:
        meta_dict = await get_stock_metadata(gbi_ids=gbi_ids)
        return [
            StockID(gbi_id=meta.gbi_id, symbol=meta.symbol, isin=meta.isin)
            for meta in meta_dict.values()
        ]

    async def to_rich_output(self, pg: BoostedPG) -> Output:
        return TextOutput(val=self.symbol or self.isin)
