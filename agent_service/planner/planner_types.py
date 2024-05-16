from dataclasses import dataclass
from typing import Dict, List, Optional, Union
from uuid import uuid4

from pydantic import BaseModel, Field

from agent_service.io_type_utils import IOType


@dataclass(frozen=True, eq=True)
class Variable:
    var_name: str


# Represents a tool's arguments that have had the literals resolved. Variable
# arguments cannot be resolved until the execution plan is run.
PartialToolArgs = Dict[str, Union[IOType, Variable, List[Union[IOType, Variable]]]]


@dataclass(frozen=True)
class ParsedStep:
    output_var: str
    function: str
    arguments: Dict[str, str]
    description: str


class ToolExecutionNode(BaseModel):
    tool_name: str  # The name of the tool to be executed, for GPT
    # For all executions of the plan, nodes use consistent ID's.
    tool_task_id: str = Field(default_factory=lambda: str(uuid4()))
    args: PartialToolArgs
    description: str  # A human-readable description of the node's purpose.
    output_variable_name: Optional[str] = None
    is_output_node: bool = False


class ExecutionPlan(BaseModel):
    nodes: List[ToolExecutionNode]

    def get_plan_steps_for_gpt(self) -> str:
        output = []
        for i, node in enumerate(self.nodes, start=1):
            output.append(f"{i}. {node.description}")
        return "\n".join(output)

    def get_formatted_plan(self) -> str:
        str_list = []
        for node in self.nodes:
            str_list.append(f"{node.output_variable_name} = {node.tool_name}({node.args})")
        return "\n".join(str_list)


class ExecutionPlanParsingError(RuntimeError):
    pass
