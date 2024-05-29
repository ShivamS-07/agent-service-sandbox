import enum
from abc import ABC
from typing import List, Literal, Optional, Union, cast

from pydantic.fields import Field

from agent_service.io_type_utils import ComplexIOBase, PrimitiveType, io_type
from agent_service.io_types.output import Output, OutputType
from agent_service.io_types.table import TableColumnType
from agent_service.utils.boosted_pg import BoostedPG
from agent_service.utils.stock_metadata import get_stock_metadata


class GraphType(str, enum.Enum):
    LINE = "line"
    PIE = "pie"


@io_type
class Graph(ComplexIOBase, ABC):
    graph_type: GraphType


@io_type
class DataPoint(ComplexIOBase):
    x_val: Optional[PrimitiveType]
    y_val: Optional[PrimitiveType]


@io_type
class GraphDataset(ComplexIOBase):
    dataset_id: PrimitiveType
    dataset_id_type: TableColumnType
    points: List[DataPoint]


@io_type
class LineGraph(Graph):
    graph_type: Literal[GraphType.LINE] = GraphType.LINE
    x_axis_type: TableColumnType
    x_unit: Optional[str] = None
    y_axis_type: TableColumnType
    y_unit: Optional[str] = None
    data: List[GraphDataset]

    async def to_rich_output(self, pg: BoostedPG) -> Output:
        stocks_to_map = cast(
            List[int],
            [
                dataset.dataset_id
                for dataset in self.data
                if dataset.dataset_id_type == TableColumnType.STOCK
            ],
        )
        if not stocks_to_map:
            return GraphOutput(graph=self)

        metadata_map = await get_stock_metadata(pg, gbi_ids=stocks_to_map)
        for dataset in self.data:
            gbi = dataset.dataset_id
            if gbi in metadata_map:
                dataset.dataset_id = metadata_map[gbi].symbol  # type: ignore
        return GraphOutput(graph=self)

    def to_gpt_input(self) -> str:
        return (
            f"Line Graph with X axis type: {self.x_axis_type.value},"
            f" Y axis type: {self.y_axis_type.value}."
        )


@io_type
class PieSection(ComplexIOBase):
    label: PrimitiveType
    value: PrimitiveType


@io_type
class PieGraph(Graph):
    graph_type: Literal[GraphType.PIE] = GraphType.PIE
    label_type: TableColumnType
    data_type: TableColumnType
    unit: Optional[str] = None
    data: List[PieSection]

    async def to_rich_output(self, pg: BoostedPG) -> Output:
        if not self.label_type == TableColumnType.STOCK:
            return GraphOutput(graph=self)
        stocks_to_map = cast(List[int], [section.label for section in self.data])
        metadata_map = await get_stock_metadata(pg, gbi_ids=stocks_to_map)
        for section in self.data:
            section.label = metadata_map[section.label].symbol  # type: ignore
        return GraphOutput(graph=self)

    def to_gpt_input(self) -> str:
        return f"Pie Chart with sections: {self.data}"


class GraphOutput(Output):
    output_type: Literal[OutputType.GRAPH] = OutputType.GRAPH
    graph: Union[PieGraph, LineGraph] = Field(discriminator="graph_type")
