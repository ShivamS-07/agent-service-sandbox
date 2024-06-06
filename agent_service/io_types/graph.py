import enum
from abc import ABC
from typing import List, Literal, Optional, Union

from pydantic.fields import Field

from agent_service.io_type_utils import (
    ComplexIOBase,
    PrimitiveType,
    TableColumnType,
    io_type,
)
from agent_service.io_types.output import Output, OutputType
from agent_service.io_types.stock import StockID
from agent_service.utils.boosted_pg import BoostedPG


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
    dataset_id: Union[PrimitiveType, StockID]
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
        for dataset in self.data:
            if isinstance(dataset.dataset_id, StockID):
                dataset.dataset_id = dataset.dataset_id.symbol or dataset.dataset_id.isin
        return GraphOutput(graph=self)

    def to_gpt_input(self) -> str:
        return (
            f"Line Graph with X axis type: {self.x_axis_type.value},"
            f" Y axis type: {self.y_axis_type.value}."
        )


@io_type
class PieSection(ComplexIOBase):
    label: Union[PrimitiveType, StockID]
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
        for section in self.data:
            if isinstance(section.label, StockID):
                section.label = section.label.symbol or section.label.isin
        return GraphOutput(graph=self)

    def to_gpt_input(self) -> str:
        return f"Pie Chart with sections: {self.data}"


class GraphOutput(Output):
    output_type: Literal[OutputType.GRAPH] = OutputType.GRAPH
    graph: Union[PieGraph, LineGraph] = Field(discriminator="graph_type")
