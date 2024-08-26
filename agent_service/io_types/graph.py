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
from agent_service.utils.stock_metadata import StockMetadata, get_stock_metadata

MAX_DATAPOINTS_FOR_GPT = 10


class GraphType(enum.StrEnum):
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
    dataset_id: Union[PrimitiveType, StockID, StockMetadata]
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
        dataset_name = await io_type_to_gpt_input(
            self.dataset_id.symbol or self.dataset_id.company_name
            if isinstance(self.dataset_id, StockMetadata)
            else self.dataset_id
        )
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
        gbi_ids = [
            dataset.dataset_id.gbi_id
            for dataset in self.data
            if isinstance(dataset.dataset_id, StockID)
        ]
        metadata = await get_stock_metadata(gbi_ids=gbi_ids, pg=pg)
        for dataset in self.data:
            if isinstance(dataset.dataset_id, StockID):
                dataset.dataset_id = (
                    metadata.get(dataset.dataset_id.gbi_id)
                    or dataset.dataset_id.symbol
                    or dataset.dataset_id.isin
                )
        return GraphOutput(
            graph=LineGraph(
                x_axis_type=self.x_axis_type,
                x_unit=self.x_unit,
                y_axis_type=self.y_axis_type,
                y_unit=self.y_unit,
                data=self.data,
            ),
            title=title,
        )

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
    label: Union[PrimitiveType, StockID, StockMetadata]
    value: PrimitiveType
    citation_refs: List[str] = []

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
        gbi_ids = [
            section.label.gbi_id for section in self.data if isinstance(section.label, StockID)
        ]
        metadata = await get_stock_metadata(gbi_ids=gbi_ids, pg=pg)
        for section in self.data:
            if isinstance(section.label, StockID):
                section.label = (
                    metadata.get(section.label.gbi_id) or section.label.symbol or section.label.isin
                )
        citations = await Citation.resolve_all_citations(self.get_all_citations(), db=pg)
        return GraphOutput(
            graph=PieGraph(
                label_type=self.label_type, data_type=self.data_type, unit=self.unit, data=self.data
            ),
            title=title,
            citations=citations,
        )

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
    label: Optional[Union[PrimitiveType, StockID, StockMetadata]]
    value: Optional[PrimitiveType]


@io_type
class BarData(ComplexIOBase):
    index: Union[PrimitiveType, StockID, StockMetadata]
    # values are a list of tuples (dataset_id, value)
    values: List[BarDataPoint]


@io_type
class BarGraph(Graph):
    graph_type: Literal[GraphType.BAR] = GraphType.BAR
    data_type: TableColumnType  # values type
    data_unit: Optional[str] = None  # currency valued unit data
    data: List[BarData]
    # For backwards compatibility
    index_type: TableColumnType = TableColumnType.STRING
    label_type: TableColumnType = TableColumnType.STRING

    async def to_rich_output(self, pg: BoostedPG, title: str = "") -> Output:
        gbi_ids = [bar.index.gbi_id for bar in self.data if isinstance(bar.index, StockID)]
        gbi_ids.extend(
            [
                point.label.gbi_id
                for bar in self.data
                for point in bar.values
                if isinstance(point.label, StockID)
            ]
        )
        metadata = await get_stock_metadata(gbi_ids=gbi_ids, pg=pg)
        for bar in self.data:
            if isinstance(bar.index, StockID):
                # Make sure this is set, for backwards compat
                self.index_type = TableColumnType.STOCK
                bar.index = metadata.get(bar.index.gbi_id) or bar.index.symbol or bar.index.isin
            for bar_point in bar.values:
                if isinstance(bar_point.label, StockID):
                    # Make sure this is set, for backwards compat
                    self.label_type = TableColumnType.STOCK
                    bar_point.label = (
                        metadata.get(bar_point.label.gbi_id)
                        or bar_point.label.symbol
                        or bar_point.label.isin
                    )
        return GraphOutput(
            graph=BarGraph(
                data_type=self.data_type,
                data_unit=self.data_unit,
                data=self.data,
                index_type=self.index_type,
                label_type=self.label_type,
            ),
            title=title,
        )

    async def to_gpt_input(self, use_abbreviated_output: bool = True) -> str:
        return f"Bar Chart: {self.data}"


class GraphOutput(Output):
    output_type: Literal[OutputType.GRAPH] = OutputType.GRAPH
    graph: Union[PieGraph, LineGraph, BarGraph] = Field(discriminator="graph_type")
