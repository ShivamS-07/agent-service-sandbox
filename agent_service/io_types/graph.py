from typing import Literal

from agent_service.io_type_utils import ComplexIOBase, io_type
from agent_service.io_types.output import Output, OutputType


@io_type
class Graph(ComplexIOBase):
    pass


class GraphOutput(Output):
    output_type: Literal[OutputType.GRAPH] = OutputType.GRAPH
