import enum
import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Union
from uuid import uuid4

from pydantic import BaseModel, Field
from pydantic.functional_validators import field_validator

from agent_service.io_type_utils import ComplexIOBase, IOType, io_type
from agent_service.io_types.text import Text

logger = logging.getLogger(__name__)


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
PartialToolArgs = Dict[  # type: ignore
    str,
    Union[Variable, IOType, List[Union[Variable, IOType]], Dict[IOType, Union[Variable, IOType]]],  # type: ignore
]


@dataclass(frozen=True)
class ParsedStep:
    output_var: str
    function: str
    arguments: Dict[str, str]
    description: str


def convert_arg(
    arg: Union[
        IOType, Variable, List[Union[IOType, Variable]], Dict[IOType, Union[Variable, IOType]]
    ],
) -> str:
    if isinstance(arg, list):
        return f"[{', '.join(convert_arg(item) for item in arg)}]"
    elif isinstance(arg, dict):
        dict_str = ", ".join(
            (f"{convert_arg(key)}: {convert_arg(val)}" for key, val in arg.items())
        )
        return f"{{{dict_str}}}"
    elif isinstance(arg, Variable):
        if arg.index is None:
            return arg.var_name
        else:
            return f"{arg.var_name}[{arg.index}]"
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
            elif isinstance(val, dict):
                for elem_key, elem in val.items():
                    if isinstance(elem, dict) and "var_name" in elem:
                        args[key][elem_key] = Variable(
                            var_name=elem["var_name"], index=elem.get("index")
                        )

        return args

    def get_plan_step_str(
        self, include_task_id: bool = False, include_description: bool = True
    ) -> str:
        if include_description:
            description_str = f"  # {self.description}"
        else:
            description_str = ""
        if include_task_id:
            task_id_str = f" (Task ID: {self.tool_task_id})"
        else:
            task_id_str = ""
        return f"{self.output_variable_name} = {self.tool_name}({self.convert_args()}){description_str}{task_id_str}"  # noqa

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
            elif isinstance(val, dict):
                actual_dict = {}
                for key, elem in val.items():
                    actual_dict[key] = self._resolve_single_arg(elem, variable_lookup)
                resolved_args[arg] = actual_dict
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
            elif isinstance(val, dict):
                variables.extend(
                    (dict_val for dict_val in val.values() if isinstance(dict_val, Variable))
                )

        return variables


class PlanStatus(enum.StrEnum):
    CREATING = "CREATING"
    READY = "READY"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class ExecutionPlan(BaseModel):
    nodes: List[ToolExecutionNode]
    locked_task_ids: List[str] = []
    deleted_task_ids: List[str] = []

    def __hash__(self) -> int:
        return self.get_formatted_plan().__hash__()

    def __eq__(self, value: object) -> bool:
        if isinstance(value, ExecutionPlan):
            return self.get_formatted_plan() == value.get_formatted_plan()
        return False

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
        prune_deleted_tasks: bool = True,
    ) -> "ExecutionPlan":
        plan = cls.model_validate(data)
        if prune_deleted_tasks and plan.deleted_task_ids:
            plan = plan.get_pruned_plan(task_ids_to_remove=set(plan.deleted_task_ids))
        return plan

    def get_plan_steps_for_gpt(self) -> str:
        output = []
        for i, node in enumerate(self.nodes, start=1):
            output.append(f"{i}. {node.description}")
        return "\n".join(output)

    def get_formatted_plan(
        self,
        numbered: bool = False,
        include_task_ids: bool = False,
        include_descriptions: bool = True,
    ) -> str:
        str_list = []
        for i, node in enumerate(self.nodes, start=1):
            prefix = ""
            if numbered:
                prefix = f"{i}. "
            str_list.append(
                f"{prefix}{node.get_plan_step_str(include_task_id=include_task_ids, include_description=include_descriptions)}"
            )
        return "\n\n".join(str_list)

    def get_output_nodes(self) -> List[ToolExecutionNode]:
        """
        Returns a list of nodes that are output nodes. Order is preserved.
        """
        return [node for node in self.nodes if node.is_output_node]

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

    def get_node_parent_map(self) -> Dict[ToolExecutionNode, Set[ToolExecutionNode]]:
        """
        Given an execution plan (i.e. a list of tool execution nodes), we can
        construct a reverse dependency tree mapping each node to its parent
        nodes.
        """
        node_to_children = self.get_node_dependency_map()
        node_to_parents = defaultdict(set)
        for parent, children in node_to_children.items():
            for child in children:
                node_to_parents[child].add(parent)

        return node_to_parents

    def inherit_locked_task_ids_from(self, old_plan: "ExecutionPlan") -> None:
        task_id_set = {node.tool_task_id for node in self.nodes}
        self.locked_task_ids = [
            task_id for task_id in old_plan.locked_task_ids if task_id in task_id_set
        ]

    def remove_non_locked_output_nodes(self) -> "ExecutionPlan":
        """
        Returns a NEW plan with non-locked output nodes removed, and pruned to
        remove unused tasks.
        """
        locked_task_set = set(self.locked_task_ids)
        logger.info(f"Replanning while preserving locked tasks: {locked_task_set}")
        node_dep_map = self.get_node_dependency_map()
        non_locked_output_task_ids = {
            node.tool_task_id
            for node in self.nodes
            if node.is_output_node
            and node.tool_task_id not in locked_task_set
            # don't remove it if it's depended on, e.g. analyze_output
            and not node_dep_map.get(node)
        }
        return self.get_pruned_plan(task_ids_to_remove=non_locked_output_task_ids)

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
        node_to_parents = plan_with_nodes_removed.get_node_parent_map()

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
        remaining_task_ids = {node.tool_task_id for node in remaining_nodes}
        return ExecutionPlan(
            nodes=remaining_nodes,
            # Preserve locked and deleted tasks that are still in the new plan
            locked_task_ids=[
                task_id for task_id in self.locked_task_ids if task_id in remaining_task_ids
            ],
            deleted_task_ids=[
                task_id for task_id in self.deleted_task_ids if task_id in remaining_task_ids
            ],
        )

    def reorder_plan_with_output_task_ordering(
        self, output_task_ordering: List[ToolExecutionNode]
    ) -> "ExecutionPlan":
        """
        Given an ordered list of output nodes, return a NEW execution plan with
        that will produce outputs in the order given. NOTE: THIS DOES NOT UPDATE
        IN PLACE.
        """
        output_nodes = self.get_output_nodes()
        output_nodes_set = set(output_nodes)

        # Add any unspecified output nodes to the end
        output_nodes_not_specified = output_nodes_set - set(output_task_ordering)
        output_task_ordering.extend(output_nodes_not_specified)

        if not all((node in output_nodes_set for node in output_task_ordering)):
            raise RuntimeError(
                f"Some nodes in the ordering lists were not output nodes! {output_task_ordering}"
            )

        nodes_included = set()
        final_nodes = []

        # This method is probably inefficient in terms of time complexity, but
        # in the scheme of things the size of the inputs will be so small that
        # it won't matter.
        for output_node in output_task_ordering:
            # Get the execution plan of ONLY this node by removing all
            # the others. We could make this more efficient.
            task_ids_to_remove = {node.tool_task_id for node in output_nodes if node != output_node}
            isolated_plan_nodes = self.get_pruned_plan(task_ids_to_remove=task_ids_to_remove).nodes
            for node in isolated_plan_nodes:
                if node not in nodes_included:
                    final_nodes.append(node)
                    nodes_included.add(node)

        return ExecutionPlan(nodes=final_nodes, locked_task_ids=self.locked_task_ids)

    def get_plan_after_task(self, task_id: str) -> "ExecutionPlan":
        nodes_to_keep = []
        seen_task_id = False
        for node in self.nodes:
            if seen_task_id:
                nodes_to_keep.append(node)
            elif node.tool_task_id == task_id:
                seen_task_id = True

        return ExecutionPlan(nodes=nodes_to_keep, locked_task_ids=self.locked_task_ids)


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
    relevance: Optional[float] = None
    enabled: str = ""
    category: Optional[str] = None
    author: Optional[str] = None
    note: Optional[str] = None
    changelog: Optional[str] = None
    last_updated_author: Optional[str] = None

    def get_formatted_plan(self) -> str:
        return f"Input: {self.input}\nPlan:\n{self.plan}"

    def get_formatted_plan_internal(self, show_changelog: bool = False) -> str:
        res = f"""
Input: {self.input}
Plan:
{self.plan}
Relevance: {self.relevance if self.relevance else "N/A"}
Enabled: {self.enabled if self.enabled else "N/A"}
Category: {self.category if self.category else "N/A"}
Author: {self.author if self.author else "N/A"}
Last Updated Author: {self.last_updated_author if self.last_updated_author else "N/A"}
Note:
{self.note if self.note else ""}"""
        if show_changelog:
            res += f"\nChangelog:\n{self.changelog if self.changelog else ""}"
        return res


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
    # Tasks which analyze or read this output are dependent on it.
    dependent_task_ids: List[str] = field(default_factory=list)
    parent_task_ids: List[str] = field(default_factory=list)
