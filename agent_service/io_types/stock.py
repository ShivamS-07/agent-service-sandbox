from __future__ import annotations

from typing import Any, List, Optional

from agent_service.io_type_utils import ComplexIOBase, io_type
from agent_service.io_types.output import Output
from agent_service.utils.boosted_pg import BoostedPG
from agent_service.utils.stock_metadata import get_stock_metadata


@io_type
class StockID(ComplexIOBase):
    gbi_id: int
    symbol: Optional[str]
    isin: str
    # Default for backwards compat
    company_name: str = ""

    def __hash__(self) -> int:
        return self.gbi_id

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, StockID):
            return self.gbi_id == other.gbi_id
        return False

    def __lt__(self, other: Any) -> bool:
        if isinstance(other, StockID):
            return self.gbi_id < other.gbi_id
        return NotImplemented

    async def to_gpt_input(self, use_abbreviated_output: bool = True) -> str:
        return f"<StockID: {self.company_name} ({self.symbol or self.isin})>"

    @staticmethod
    async def from_gbi_id_list(gbi_ids: List[int]) -> List["StockID"]:
        meta_dict = await get_stock_metadata(gbi_ids=gbi_ids)
        return [
            StockID(
                gbi_id=meta.gbi_id,
                symbol=meta.symbol,
                isin=meta.isin,
                company_name=meta.company_name,
            )
            for meta in meta_dict.values()
        ]

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
        text = Text(val=string_val)
        return await text.to_rich_output(pg=pg)

    def to_hashable(self) -> str:
        return self.model_dump_json()

    @staticmethod
    def from_hashable(val: str) -> StockID:
        return StockID.model_validate_json(val)
