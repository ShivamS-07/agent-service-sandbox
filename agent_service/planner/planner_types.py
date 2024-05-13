from dataclasses import dataclass
from typing import Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field

from agent_service.tool import PartialToolArgs


@dataclass(frozen=True)
class ParsedStep:
    output_var: str
    function: str
    arguments: Dict[str, str]
    description: str


class ToolExecutionNode(BaseModel):
    tool_name: str
    # For all executions of the plan, nodes use consistent ID's.
    tool_task_id: str = Field(default_factory=lambda: str(uuid4()))
    args: PartialToolArgs
    output_variable_name: Optional[str] = None
    is_output_node: bool = False


class ExecutionPlan(BaseModel):
    nodes: List[ToolExecutionNode]


class ExecutionPlanParsingError(RuntimeError):
    pass
