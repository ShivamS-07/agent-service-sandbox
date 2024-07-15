import datetime
import enum
from abc import ABC
from typing import List, Optional
from uuid import uuid4

from pydantic import BaseModel
from pydantic.fields import Field


class CitationType(str, enum.Enum):
    LINK = "link"
    TEXT = "text"


CitationID = str


class CitationOutput(BaseModel):
    id: CitationID = Field(default_factory=lambda: str(uuid4()))
    citation_type: CitationType
    name: str
    link: Optional[str] = None
    summary: Optional[str] = None
    # Used to track when the underlying text (article, summary, etc.) was created
    published_at: Optional[datetime.datetime] = None
    article_id: Optional[str] = None


class OutputType(str, enum.Enum):
    TABLE = "table"
    TEXT = "text"
    GRAPH = "graph"
    LAYOUT = "layout"


class Output(BaseModel, ABC):
    output_type: OutputType
    title: str = ""
    citations: List[CitationOutput] = []
