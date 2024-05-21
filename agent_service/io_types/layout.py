from typing import List, Union

from agent_service.io_type_utils import ComplexIOBase, io_type
from agent_service.io_types.graph import Graph
from agent_service.io_types.table import Table
from agent_service.io_types.text import Text

Widget = Union[Table, Graph, Text]


@io_type
class LayoutComponent(ComplexIOBase):
    value: Widget
    # Columns are from 1 - 4
    columns: int = 4


@io_type
class Layout(ComplexIOBase):
    layout: List[LayoutComponent]
