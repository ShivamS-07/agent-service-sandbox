import datetime
import enum
from typing import Optional
from uuid import uuid4

from pydantic import BaseModel, Field

from agent_service.planner.planner_types import ExecutionPlan


class OutputType(enum.StrEnum):
    TEXT = "text"
    TABLE = "table"
    LINE_GRAPH = "line_graph"
    PIE_GRAPH = "pie_graph"
    BAR_GRAPH = "bar_graph"


class PromptTemplate(BaseModel):
    template_id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    description: str
    prompt: str
    category: str
    created_at: datetime.datetime
    plan: ExecutionPlan
    output_types: Optional[list[OutputType]] = None
