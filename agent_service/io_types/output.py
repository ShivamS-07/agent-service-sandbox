import enum
from abc import ABC
from typing import List, Union

from pydantic import BaseModel

from agent_service.io_types.citations import (
    CitationOutput,
    CompanyFilingCitationOutput,
    CustomDocumentCitationOutput,
    DocumentCitationOutput,
    KPICitationOutput,
    LinkCitationOutput,
    NewsArticleCitationOutput,
    NewsDevelopmentCitationOutput,
    TextCitationOutput,
    ThemeCitationOutput,
)


class OutputType(enum.StrEnum):
    TABLE = "table"
    TEXT = "text"
    GRAPH = "graph"
    LAYOUT = "layout"


OutputCitationTypes = Union[
    ThemeCitationOutput,
    CustomDocumentCitationOutput,
    CompanyFilingCitationOutput,
    NewsDevelopmentCitationOutput,
    NewsArticleCitationOutput,
    DocumentCitationOutput,
    TextCitationOutput,
    LinkCitationOutput,
    CitationOutput,
    KPICitationOutput,
]


class Output(BaseModel, ABC):
    output_type: OutputType
    title: str = ""
    citations: List[OutputCitationTypes] = []
