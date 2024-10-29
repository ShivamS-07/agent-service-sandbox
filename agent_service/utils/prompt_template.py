import datetime
import enum
from typing import List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field

from agent_service.io_types.graph import GraphType
from agent_service.planner.planner_types import ExecutionPlan


class OutputType(enum.StrEnum):
    TEXT = "text"
    TABLE = "table"
    LINE_GRAPH = GraphType.LINE
    PIE_GRAPH = GraphType.PIE
    BAR_GRAPH = GraphType.BAR


class OutputPreview(BaseModel):
    title: str
    output_type: OutputType


class PromptTemplate(BaseModel):
    template_id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    description: str
    prompt: str
    category: str
    created_at: datetime.datetime
    plan: ExecutionPlan
    is_visible: bool = False
    organization_ids: Optional[List[str]] = None
    preview: Optional[List[OutputPreview]] = None


class UserOrganization(BaseModel):
    organization_id: str
    organization_name: str
