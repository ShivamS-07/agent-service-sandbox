import enum
from abc import ABC
from typing import List

from pydantic import BaseModel

from agent_service.io_types.citations import CitationOutput


class OutputType(str, enum.Enum):
    TABLE = "table"
    TEXT = "text"
    GRAPH = "graph"
    LAYOUT = "layout"


class Output(BaseModel, ABC):
    output_type: OutputType
    title: str = ""
    citations: List[CitationOutput] = []
