from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional

from pydantic.functional_serializers import field_serializer
from pydantic.functional_validators import field_validator

from agent_service.io_type_utils import ComplexIOBase, io_type
from agent_service.io_types.output import Output
from agent_service.io_types.text import Text, TextGroup
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

    async def to_rich_output(self, pg: BoostedPG, title: str = "") -> Output:
        # convert the stock to a rich text format showing its history if present
        strings = [f"**{self.company_name} ({self.symbol or self.isin})**"]
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

    @staticmethod
    def from_stocks_and_text(stocks: List[StockID], texts: List[Text]) -> StockAlignedTextGroups:
        temp_dict = defaultdict(list)
        for text in texts:
            if hasattr(text, "gbi_id"):  # might not be the right kind of text
                temp_dict[text.gbi_id].append(text)

        final_dict = {}
        for stock in stocks:
            if stock.gbi_id in temp_dict:
                final_dict[stock] = TextGroup(val=temp_dict[stock.gbi_id])

        return StockAlignedTextGroups(val=final_dict)

    # Need to do this for types with complex keys, since json keys can only be strings
    @field_validator("val", mode="before")
    @classmethod
    def _deserializer(cls, data: Any) -> Any:
        if isinstance(data, dict):
            # convert the stocks to back to StockID's
            data = {
                StockID.model_validate_json(stock) if isinstance(stock, str) else stock: text
                for stock, text in data.items()
            }
        return data

    @field_serializer("val", mode="wrap")
    @classmethod
    def _field_serializer(cls, data: Any, dumper: Callable) -> Any:
        if isinstance(data, dict):
            data = {stock.model_dump_json(): text for stock, text in data.items()}
        return dumper(data)
