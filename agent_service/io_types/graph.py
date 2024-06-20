import enum
from abc import ABC
from itertools import islice
from typing import List, Literal, Optional, Union

from pydantic.fields import Field

from agent_service.io_type_utils import (
    Citation,
    ComplexIOBase,
    PrimitiveType,
    TableColumnType,
    io_type,
)
from agent_service.io_types.output import Output, OutputType
from agent_service.io_types.stock import StockID
from agent_service.utils.async_utils import gather_with_concurrency
from agent_service.utils.boosted_pg import BoostedPG
from agent_service.utils.output_utils.utils import io_type_to_gpt_input

MAX_DATAPOINTS_FOR_GPT = 10


class GraphType(str, enum.Enum):
    LINE = "line"
    PIE = "pie"
    BAR = "bar"


@io_type
class Graph(ComplexIOBase, ABC):
    graph_type: GraphType


@io_type
class DataPoint(ComplexIOBase):
    x_val: Optional[PrimitiveType]
    y_val: Optional[PrimitiveType]

    async def to_gpt_input(self, use_abbreviated_output: bool = True) -> str:
        return str((self.x_val, self.y_val))


@io_type
class GraphDataset(ComplexIOBase):
    dataset_id: Union[PrimitiveType, StockID]
    dataset_id_type: TableColumnType
    points: List[DataPoint]

    async def to_gpt_input(self, use_abbreviated_output: bool = True) -> str:
        # Make sure the points are sorted
        latest_N_datapoints = list(
            islice(
                sorted(self.points, key=lambda point: (point.x_val, point.y_val), reverse=True),
                MAX_DATAPOINTS_FOR_GPT,
            )
        )
        dataset_name = await io_type_to_gpt_input(self.dataset_id)
        datapoints = await gather_with_concurrency(
            [point.to_gpt_input() for point in latest_N_datapoints]
        )
        return f"<{dataset_name}: {list(datapoints)}>"


@io_type
class LineGraph(Graph):
    graph_type: Literal[GraphType.LINE] = GraphType.LINE
    x_axis_type: TableColumnType
    x_unit: Optional[str] = None
    y_axis_type: TableColumnType
    y_unit: Optional[str] = None
    data: List[GraphDataset]

    async def to_rich_output(self, pg: BoostedPG, title: str = "") -> Output:
        for dataset in self.data:
            if isinstance(dataset.dataset_id, StockID):
                dataset.dataset_id = dataset.dataset_id.symbol or dataset.dataset_id.isin
        return GraphOutput(graph=self, title=title)

    async def to_gpt_input(self, use_abbreviated_output: bool = True) -> str:
        if use_abbreviated_output or len(self.data) > 10:
            return (
                f"<Line Graph with X axis type: {self.x_axis_type.value},"
                f" Y axis type: {self.y_axis_type.value}>"
            )
        datasets = await gather_with_concurrency([dataset.to_gpt_input() for dataset in self.data])
        return (
            f"<Line Graph with X axis type: {self.x_axis_type.value},"
            f" Y axis type: {self.y_axis_type.value}\n"
            f"Datasets: {list(datasets)}\n>"
        )


@io_type
class PieSection(ComplexIOBase):
    label: Union[PrimitiveType, StockID]
    value: PrimitiveType

    async def to_gpt_input(self, use_abbreviated_output: bool = True) -> str:
        label = self.label
        if isinstance(self.label, ComplexIOBase):
            label = await self.label.to_gpt_input()
        return f"{label}: {self.value}"


@io_type
class PieGraph(Graph):
    graph_type: Literal[GraphType.PIE] = GraphType.PIE
    label_type: TableColumnType
    data_type: TableColumnType
    unit: Optional[str] = None
    data: List[PieSection]

    async def to_rich_output(self, pg: BoostedPG, title: str = "") -> Output:
        if not self.label_type == TableColumnType.STOCK:
            return GraphOutput(graph=self)
        for section in self.data:
            if isinstance(section.label, StockID):
                section.label = section.label.symbol or section.label.isin
        citations = await Citation.resolve_all_citations(self.get_all_citations(), db=pg)
        return GraphOutput(graph=self, title=title, citations=citations)

    async def to_gpt_input(self, use_abbreviated_output: bool = True) -> str:
        section_strs = await gather_with_concurrency(
            [
                section.to_gpt_input(use_abbreviated_output=use_abbreviated_output)
                for section in self.data
            ]
        )
        sections = ", ".join(section_strs)
        return f"<Pie Chart with sections: {sections}>"


@io_type
class BarDataPoint(ComplexIOBase):
    label: Optional[Union[PrimitiveType, StockID]]
    value: Optional[PrimitiveType]


@io_type
class BarData(ComplexIOBase):
    index: Union[PrimitiveType, StockID]
    # values are a list of tuples (dataset_id, value)
    values: List[BarDataPoint]


@io_type
class BarGraph(Graph):
    graph_type: Literal[GraphType.BAR] = GraphType.BAR
    data_type: TableColumnType  # values type
    data_unit: Optional[str] = None  # currency valued unit data
    data: List[BarData]

    async def to_rich_output(self, pg: BoostedPG, title: str = "") -> Output:
        for bar in self.data:
            if isinstance(bar.index, StockID):
                bar.index = bar.index.symbol or bar.index.isin
            for bar_point in bar.values:
                if isinstance(bar_point.label, StockID):
                    bar_point.label = bar_point.label.symbol or bar_point.label.isin
        return GraphOutput(graph=self, title=title)

    async def to_gpt_input(self, use_abbreviated_output: bool = True) -> str:
        return f"Bar Chart: {self.data}"


class GraphOutput(Output):
    output_type: Literal[OutputType.GRAPH] = OutputType.GRAPH
    graph: Union[PieGraph, LineGraph, BarGraph] = Field(discriminator="graph_type")
