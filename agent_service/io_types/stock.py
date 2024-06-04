from __future__ import annotations

from typing import Any, Dict, List, Optional

from agent_service.io_type_utils import ComplexIOBase, io_type
from agent_service.io_types.output import Output
from agent_service.io_types.text import TextGroup, TextOutput
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


@io_type
class StockAlignedTextGroups(ComplexIOBase):
    val: Dict[StockID, TextGroup]

    @staticmethod
    def join(
        stock_to_texts_1: StockAlignedTextGroups, stock_to_texts_2: StockAlignedTextGroups
    ) -> StockAlignedTextGroups:
        from agent_service.io_types.stock import StockID

        output_dict = {}
        all_stocks = StockID.union_sets(
            set(stock_to_texts_1.val.keys()), set(stock_to_texts_2.val.keys())
        )
        for stock in all_stocks:
            if stock in stock_to_texts_1.val:
                if stock in stock_to_texts_2.val:
                    output_dict[stock] = TextGroup.join(
                        stock_to_texts_1.val[stock], stock_to_texts_2.val[stock]
                    )
                else:
                    output_dict[stock] = stock_to_texts_1.val[stock]
            else:
                output_dict[stock] = stock_to_texts_2.val[stock]

        return StockAlignedTextGroups(val=output_dict)
