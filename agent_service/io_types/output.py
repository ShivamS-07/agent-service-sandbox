import enum
from abc import ABC

from pydantic import BaseModel


class OutputType(str, enum.Enum):
    TABLE = "table"
    TEXT = "text"
    GRAPH = "graph"
    LAYOUT = "layout"


class Output(BaseModel, ABC):
    output_type: OutputType
