from __future__ import annotations

from typing import Any, List, Optional

from agent_service.io_type_utils import ComplexIOBase, io_type
from agent_service.io_types.output import Output
from agent_service.utils.boosted_pg import BoostedPG
from agent_service.utils.stock_metadata import StockMetadata, get_stock_metadata


@io_type
class StockID(ComplexIOBase):
    gbi_id: int
    symbol: Optional[str]
    # Default for backwards compat
    isin: str = ""
    company_name: str = ""

    def __hash__(self) -> int:
        return self.gbi_id

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, StockID):
            return (
                self.gbi_id == other.gbi_id and self.symbol == other.symbol
                if self.symbol is not None and other.symbol is not None
                else True
            )
        return False

    def __lt__(self, other: Any) -> bool:
        if isinstance(other, StockID):
            return self.gbi_id < other.gbi_id
        return NotImplemented

    def __str__(self) -> str:
        return self.symbol or self.company_name

    async def to_gpt_input(self, use_abbreviated_output: bool = True) -> str:
        return f"<StockID: {self.company_name} ({self.symbol or self.isin})>"

    @staticmethod
    async def from_gbi_id_list(gbi_ids: List[int]) -> List["StockID"]:
        # make sure to return the StockIds in the exact same order as requested
        meta_dict = await get_stock_metadata(gbi_ids=gbi_ids)
        input_order_metas = [
            meta_dict.get(
                gbi_id,
                # This should never happen but... who knows?
                StockMetadata(
                    gbi_id=gbi_id,
                    symbol="UNKNOWN_" + str(gbi_id),
                    company_name="UNKNOWN_" + str(gbi_id),
                    isin="UNKNOWN_" + str(gbi_id),
                ),
            )
            for gbi_id in gbi_ids
        ]
        return [
            StockID(
                gbi_id=meta.gbi_id,
                symbol=meta.symbol,
                isin=meta.isin,
                company_name=meta.company_name,
            )
            for meta in input_order_metas
        ]

    @staticmethod
    async def to_gbi_id_list(stock_ids: List["StockID"]) -> List[int]:
        return [s.gbi_id for s in stock_ids]

    def to_markdown_string(self) -> str:
        return f"**{self.company_name} ({self.symbol or self.isin})**"

    async def to_rich_output(self, pg: BoostedPG, title: str = "") -> Output:
        # convert the stock to a rich text format showing its history if present
        from agent_service.io_types.text import Text

        strings = [self.to_markdown_string()]
        for entry in self.history:
            if entry.title:
                strings.append(f"- **{entry.title}**: {entry.explanation}")
            else:
                strings.append(f"- {entry.explanation}")
        string_val = "\n".join(strings)
        text: Text = Text(val=string_val)
        return await text.to_rich_output(pg=pg)

    def to_hashable(self) -> str:
        return self.model_dump_json()

    @staticmethod
    def from_hashable(val: str) -> StockID:
        return StockID.model_validate_json(val)
