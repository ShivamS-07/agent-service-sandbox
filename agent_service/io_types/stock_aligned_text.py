from collections import defaultdict
from typing import Any, Callable, Dict, List

from pydantic.functional_serializers import field_serializer
from pydantic.functional_validators import field_validator

from agent_service.io_type_utils import ComplexIOBase, io_type
from agent_service.io_types.output import Output
from agent_service.io_types.stock import StockID
from agent_service.io_types.text import StockText, TextGroup
from agent_service.utils.boosted_pg import BoostedPG


@io_type
class StockAlignedTextGroups(ComplexIOBase):
    val: Dict[StockID, TextGroup]

    @staticmethod
    def join(
        stock_to_texts_1: "StockAlignedTextGroups", stock_to_texts_2: "StockAlignedTextGroups"
    ) -> "StockAlignedTextGroups":
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
    def from_stocks_and_text(
        stocks: List[StockID], texts: List[StockText]
    ) -> "StockAlignedTextGroups":
        temp_dict = defaultdict(list)
        for text in texts:
            if hasattr(text, "stock_id"):  # might not be the right kind of text
                temp_dict[text.stock_id].append(text)

        final_dict = {}
        for stock in stocks:
            if stock in temp_dict:
                final_dict[stock] = TextGroup(val=temp_dict[stock])  # type: ignore

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

    async def to_rich_output(self, pg: BoostedPG, title: str = "") -> Output:
        from agent_service.utils.output_construction import get_output_from_io_type

        return await get_output_from_io_type(val=self.val, pg=pg, title=title)
