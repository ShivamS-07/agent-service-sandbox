import enum
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Union
from uuid import uuid4

from pydantic import BaseModel, Field
from pydantic.functional_validators import field_validator

from agent_service.io_type_utils import ComplexIOBase, IOType, io_type
from agent_service.io_types.text import Text


class Variable(BaseModel):
    var_name: str
    # For variables that are lists, allow a constant index
    index: Optional[int] = None

    # Just used for type checking, otherwise not important
    var_type: Optional[Any] = Field(exclude=True, default=None)

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, Variable):
            return False
        return (self.var_name, self.index) == (value.var_name, value.index)


# Represents a tool's arguments that have had the literals resolved. Variable
# arguments cannot be resolved until the execution plan is run.
PartialToolArgs = Dict[str, Union[Variable, IOType, List[Union[Variable, IOType]]]]  # type: ignore


@dataclass(frozen=True)
class ParsedStep:
    output_var: str
    function: str
    arguments: Dict[str, str]
    description: str


def convert_arg(arg: Union[IOType, Variable, List[Union[IOType, Variable]]]) -> str:
    if isinstance(arg, list):
        return f"[{', '.join(convert_arg(item) for item in arg)}]"
    elif isinstance(arg, Variable):
        return arg.var_name
    elif isinstance(arg, str):
        return f'"{arg}"'
    else:
        return str(arg)


class ToolExecutionNode(BaseModel):
    tool_name: str  # The name of the tool to be executed, for GPT
    # For all executions of the plan, nodes use consistent ID's.
    tool_task_id: str = Field(default_factory=lambda: str(uuid4()))
    args: PartialToolArgs
    description: str  # A human-readable description of the node's purpose.
    output_variable_name: Optional[str] = None
    is_output_node: bool = False
    store_output: bool = True

    def convert_args(self) -> str:
        return ", ".join(f"{key}={convert_arg(value)}" for key, value in self.args.items())

    def __hash__(self) -> int:
        # Each node is unique, even with an identical tool name and arguments.
        return hash(self.tool_task_id)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, ToolExecutionNode):
            return False
        return other.tool_task_id == self.tool_task_id

    @field_validator("args", mode="before")
    @classmethod
    def _deserialize_args(cls, args: Any) -> Any:
        # TODO clean this up, it's a bit hacky and annoying right now
        if not isinstance(args, dict):
            return args
        # Make sure to load variables into their proper class
        for key, val in args.items():
            if isinstance(val, dict) and "var_name" in val:
                args[key] = Variable(var_name=val["var_name"], index=val.get("index"))
            elif isinstance(val, list):
                for i, elem in enumerate(val):
                    if isinstance(elem, dict) and "var_name" in elem:
                        args[key][i] = Variable(var_name=elem["var_name"], index=elem.get("index"))

        return args

    def get_plan_step_str(self) -> str:
        return f"{self.output_variable_name} = {self.tool_name}({self.convert_args()})  # {self.description}"

    @staticmethod
    def _resolve_single_arg(
        val: Union[Variable, IOType], variable_lookup: Dict[str, IOType]
    ) -> IOType:
        if isinstance(val, Variable):
            output_val = variable_lookup[val.var_name]
            if val.index is not None and isinstance(output_val, list):
                # Handle the case of indexing into a variable
                output_val = output_val[val.index]
        else:
            output_val = val
        return output_val

    def resolve_arguments(self, variable_lookup: Dict[str, IOType]) -> Dict[str, IOType]:
        resolved_args: Dict[str, IOType] = {}
        for arg, val in self.args.items():
            if isinstance(val, Variable):
                resolved_args[arg] = self._resolve_single_arg(val, variable_lookup)
            elif isinstance(val, list):
                actual_list = []
                for item in val:
                    actual_list.append(self._resolve_single_arg(item, variable_lookup))
                resolved_args[arg] = actual_list
            else:
                resolved_args[arg] = val

        return resolved_args

    def get_variables_in_arguments(self) -> List[Variable]:
        variables = []
        for val in self.args.values():
            if isinstance(val, Variable):
                variables.append(val)
            elif isinstance(val, list):
                variables.extend((list_val for list_val in val if isinstance(list_val, Variable)))

        return variables


class PlanStatus(enum.StrEnum):
    CREATING = "CREATING"
    READY = "READY"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class ExecutionPlan(BaseModel):
    nodes: List[ToolExecutionNode]

    def __hash__(self) -> int:
        return self.get_formatted_plan().__hash__()

    def __eq__(self, value: object) -> bool:
        if isinstance(value, ExecutionPlan):
            return self.get_formatted_plan() == value.get_formatted_plan()
        return False

    def get_plan_steps_for_gpt(self) -> str:
        output = []
        for i, node in enumerate(self.nodes, start=1):
            output.append(f"{i}. {node.description}")
        return "\n".join(output)

    def get_formatted_plan(self, numbered: bool = False) -> str:
        str_list = []
        for i, node in enumerate(self.nodes, start=1):
            prefix = ""
            if numbered:
                prefix = f"{i}. "
            str_list.append(f"{prefix}{node.get_plan_step_str()}")
        return "\n\n".join(str_list)

    def get_node_dependency_map(self) -> Dict[ToolExecutionNode, Set[ToolExecutionNode]]:
        """
        Given an execution plan (i.e. a list of tool execution nodes), we can
        construct a dependency tree mapping each node to its dependent
        ("children") nodes. This is useful for pruning an execution plan by
        removing any non-output nodes with no dependents.
        """

        # Maps each node to its children
        dependency_map: Dict[ToolExecutionNode, Set[ToolExecutionNode]] = {
            node: set() for node in self.nodes
        }

        # Maps a variable name to the node that created it
        variable_node_map = {
            node.output_variable_name: node for node in self.nodes if node.output_variable_name
        }
        for node in reversed(self.nodes):
            variable_args = node.get_variables_in_arguments()
            for var in variable_args:
                parent = variable_node_map.get(var.var_name)
                if not parent:
                    continue
                dependency_map[parent].add(node)

        return dependency_map

    def get_pruned_plan(self, task_ids_to_remove: Set[str]) -> "ExecutionPlan":
        """
        Given a set of task IDs that represent OUPTUTS ONLY, return a new execution
        plan with the specified task ID's removed, and unused plan nodes (that
        aren't outputs) pruned. NOTE: THIS DOES NOT UPDATE IN PLACE.
        """

        nodes_to_remove = [node for node in self.nodes if node.tool_task_id in task_ids_to_remove]
        if not all((node.is_output_node for node in nodes_to_remove)):
            raise RuntimeError("Only may remove output nodes!!")

        plan_with_nodes_removed = ExecutionPlan(
            nodes=[node for node in self.nodes if node.tool_task_id not in task_ids_to_remove]
        )
        node_to_children = plan_with_nodes_removed.get_node_dependency_map()
        node_to_parents = defaultdict(list)
        for parent, children in node_to_children.items():
            for child in children:
                node_to_parents[child].append(parent)

        nodes_to_remove = deque(
            [
                node
                for node, children in node_to_children.items()
                if not children and not node.is_output_node
            ]
        )

        while nodes_to_remove:
            node = nodes_to_remove.popleft()
            for parent in node_to_parents[node]:
                # Remove the leaf from its parent's children list
                node_to_children[parent].remove(node)

                # If the parent becomes a leaf, add it to the queue
                if not node_to_children[parent] and not parent.is_output_node:
                    nodes_to_remove.append(parent)

            # Remove the node
            node_to_children.pop(node)

        # Make sure we preserve order
        remaining_nodes = [node for node in self.nodes if node in node_to_children]
        return ExecutionPlan(nodes=remaining_nodes)


class ExecutionPlanParsingError(RuntimeError):
    pass


class ErrorInfo(BaseModel):
    error: str
    step: ToolExecutionNode
    change: str


class SamplePlan(BaseModel):
    id: str
    input: str
    plan: str

    def get_formatted_plan(self) -> str:
        return f"Input: {self.input}\nPlan:\n{self.plan}"


@io_type
class RunMetadata(ComplexIOBase):
    run_summary_long: Optional[Union[str, Text]] = None
    run_summary_short: Optional[str] = None
    updated_output_ids: Optional[List[str]] = None


@dataclass(frozen=True)
class OutputWithID:
    output: IOType
    output_id: str
    task_id: Optional[str] = None  # None for backwards compatibility
