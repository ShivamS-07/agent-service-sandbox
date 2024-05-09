from dataclasses import dataclass
from typing import Dict, List, Optional

from pydantic import BaseModel

from agent_service.tool import PartialToolArgs


@dataclass(frozen=True)
class ParsedStep:
    output_var: str
    function: str
    arguments: Dict[str, str]
    description: str


class ToolExecutionNode(BaseModel):
    tool_name: str
    args: PartialToolArgs
    output_variable_name: Optional[str] = None
    is_output_node: bool = False


class ExecutionPlan(BaseModel):
    nodes: List[ToolExecutionNode]
    current_node: Optional[ToolExecutionNode] = None


class ExecutionPlanParsingError(RuntimeError):
    pass
