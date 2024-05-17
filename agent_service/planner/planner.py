from logging import getLogger
from typing import Any, Dict, List, Optional, Tuple, Type, Union, get_args, get_origin
from uuid import uuid4

from agent_service.GPT.constants import DEFAULT_SMART_MODEL
from agent_service.GPT.requests import GPT
from agent_service.io_type_utils import IOType, PrimitiveType, check_type_is_valid
from agent_service.planner.constants import ARGUMENT_RE, ASSIGNMENT_RE
from agent_service.planner.planner_types import (
    ExecutionPlan,
    ExecutionPlanParsingError,
    ParsedStep,
    PartialToolArgs,
    ToolExecutionNode,
    Variable,
)
from agent_service.planner.prompts import (
    PLAN_EXAMPLE,
    PLAN_GUIDELINES,
    PLAN_RULES,
    PLANNER_MAIN_PROMPT,
    PLANNER_SYS_PROMPT,
)
from agent_service.tool import Tool, ToolRegistry

# Make sure all tools are imported for the planner
from agent_service.tools import *  # noqa
from agent_service.types import ChatContext
from agent_service.utils.gpt_logging import GptJobIdType, GptJobType, create_gpt_context
from agent_service.utils.logs import async_perf_logger


def get_arg_dict(arg_str: str) -> Dict[str, str]:
    if len(arg_str) == 0:
        return {}
    arg_indicies = [0, *[match.start() for match in ARGUMENT_RE.finditer(arg_str)], len(arg_str)]
    arg_dict = {}
    for i in range(len(arg_indicies) - 1):
        key, value = arg_str[arg_indicies[i] : arg_indicies[i + 1]].strip(" ,").split("=")
        arg_dict[key.strip()] = value.strip()
    return arg_dict


logger = getLogger(__name__)


class Planner:
    def __init__(
        self,
        agent_id: str,
        model: str = DEFAULT_SMART_MODEL,
        tool_registry: Type[ToolRegistry] = ToolRegistry,
    ) -> None:
        self.agent_id = agent_id
        context = create_gpt_context(GptJobType.AGENT_PLANNER, agent_id, GptJobIdType.AGENT_ID)
        self.llm = GPT(context, model)
        self.tool_registry = tool_registry
        self.tool_string = tool_registry.get_tool_str()

    @async_perf_logger
    async def create_initial_plan(self, chat_context: ChatContext) -> ExecutionPlan:
        # TODO: put this in a loop where if the parse fails, we query GPT with
        # two additional messages: the GPT response and the parse exception

        plan_str = await self._query_GPT_for_initial_plan(chat_context.get_gpt_input())

        steps = self._parse_plan_str(plan_str)

        try:
            plan = self._validate_and_construct_plan(steps)
        except Exception:
            logger.warning(f"Failed to validate plan with steps: {steps}")
            raise

        return plan

    async def _query_GPT_for_initial_plan(self, user_input: str) -> str:
        sys_prompt = PLANNER_SYS_PROMPT.format(
            rules=PLAN_RULES,
            guidelines=PLAN_GUIDELINES,
            example=PLAN_EXAMPLE,
            tools=self.tool_string,
        )

        main_prompt = PLANNER_MAIN_PROMPT.format(message=user_input)
        return await self.llm.do_chat_w_sys_prompt(main_prompt, sys_prompt, no_cache=True)

    def _try_parse_str_literal(self, val: str) -> Optional[str]:
        if (val.startswith('"') and val.endswith('"')) or (
            val.startswith("'") and val.endswith("'")
        ):
            # Strip off the quotes, return a string
            return val[1:-1]
        return None

    def _try_parse_bool_literal(self, val: str) -> Optional[bool]:
        if val == "True":
            return True
        if val == "False":
            return False
        return None

    def _try_parse_int_literal(self, val: str) -> Optional[int]:
        try:
            return int(val)
        except ValueError:
            return None

    def _try_parse_float_literal(self, val: str) -> Optional[float]:
        try:
            return float(val)
        except ValueError:
            return None

    def _try_parse_variable(
        self, var_name: str, variable_lookup: Dict[str, Type[IOType]]
    ) -> Optional[Type[IOType]]:
        return variable_lookup.get(var_name)

    def _try_parse_primitive_literal(
        self, val: str, expected_type: Optional[Type]
    ) -> Optional[IOType]:
        parsed_val: Optional[PrimitiveType] = None
        if expected_type is float:
            parsed_val = self._try_parse_float_literal(val)
        elif expected_type is int:
            parsed_val = self._try_parse_int_literal(val)
        elif expected_type is str:
            parsed_val = self._try_parse_str_literal(val)
        elif expected_type is bool:
            parsed_val = self._try_parse_bool_literal(val)
        else:
            # Try them all!
            parsed_val = self._try_parse_int_literal(val)
            if parsed_val is None:
                parsed_val = self._try_parse_float_literal(val)
            if parsed_val is None:
                parsed_val = self._try_parse_str_literal(val)
            if parsed_val is None:
                parsed_val = self._try_parse_bool_literal(val)

        return parsed_val

    def _try_parse_variable_or_literal(
        self,
        val_or_name: str,
        expected_type: Optional[Type],
        variable_lookup: Dict[str, Type[IOType]],
    ) -> Optional[Union[IOType, Variable]]:
        # First try to parse as a literal
        output = self._try_parse_primitive_literal(val=val_or_name, expected_type=expected_type)
        if output is None:
            var_type = self._try_parse_variable(
                var_name=val_or_name, variable_lookup=variable_lookup
            )
            if var_type is None:
                return None
            if not check_type_is_valid(var_type, expected_type):
                raise ExecutionPlanParsingError(
                    f"Variable {val_or_name} is invalid type, got {var_type}, expecting {expected_type}."
                )
            output = Variable(var_name=val_or_name)

        return output

    def _try_parse_list_literal(
        self,
        val: str,
        expected_type: Optional[Type],
        variable_lookup: Dict[str, Type[IOType]],
    ) -> Optional[List[Union[IOType, Variable]]]:
        if (not val.startswith("[")) or (not val.endswith("]")):
            return None
        contents_str = val[1:-1]
        list_type_tup: Tuple[Type] = get_args(expected_type)
        list_type: Type
        if not list_type_tup:
            list_type = Any
        else:
            list_type = list_type_tup[0]
        elements = contents_str.split(",")
        output = []
        for i, elem in enumerate(elements):
            elem = elem.strip()
            parsed = self._try_parse_variable_or_literal(
                elem, expected_type=list_type, variable_lookup=variable_lookup
            )
            if parsed is None:
                raise ExecutionPlanParsingError(
                    f"Element {i} of list {val} is invalid type, expecting {expected_type}"
                )
            output.append(parsed)

        return output

    def _validate_tool_arguments(
        self, tool: Tool, args: Dict[str, str], variable_lookup: Dict[str, Type[IOType]]
    ) -> PartialToolArgs:
        """
        Given a tool, a set of arguments, and a variable lookup table, validate
        all the arguments for the tool given by GPT. If validation is
        successful, return a PartialToolArgs, which represents both literals and
        variables that must be bound to values at execution time.
        """
        expected_args = tool.input_type.model_fields
        parsed_args: PartialToolArgs = {}

        for arg, val in args.items():
            if not val:
                raise ExecutionPlanParsingError(
                    f"Tool '{tool.name} has invalid empty argument for '{arg}'"
                )
            if arg not in expected_args:
                raise ExecutionPlanParsingError(f"Tool '{tool.name}' has invalid argument '{arg}'")

            arg_info = expected_args[arg]

            # For literals, we can parse out an actual value at "compile" time
            parsed_val = self._try_parse_primitive_literal(val, expected_type=arg_info.annotation)
            if parsed_val is None:
                parsed_val = self._try_parse_list_literal(
                    val, arg_info.annotation, variable_lookup=variable_lookup
                )

            if parsed_val is not None:
                literal_typ: Type = type(parsed_val)
                expected = get_origin(arg_info.annotation) or arg_info.annotation
                if not check_type_is_valid(actual=literal_typ, expected=expected):
                    raise ExecutionPlanParsingError(
                        (
                            f"Tool '{tool.name}' has incorrectly typed literal argument for '{arg}'. "
                            f"Expected '{arg_info.annotation}' but found '{val}' ({literal_typ})"
                        )
                    )
                # We have found a literal. Add it to the map,
                # and move on to the next argument.
                parsed_args[arg] = parsed_val
                continue

            parsed_variable_type = self._try_parse_variable(val, variable_lookup=variable_lookup)
            # First check for an undefined variable
            if not parsed_variable_type:
                raise ExecutionPlanParsingError(
                    f"Tool '{tool.name}' has undefined variable argument '{arg}'."
                )
            # Next, check if the variable's type matches the expected type for the argument
            if not check_type_is_valid(actual=parsed_variable_type, expected=arg_info.annotation):
                raise ExecutionPlanParsingError(
                    (
                        f"Tool '{tool.name}' has a type mismatch for arg '{arg}',"
                        f" expected '{arg_info.annotation}' but found '{val}: {parsed_variable_type}'"
                    )
                )
            parsed_args[arg] = Variable(var_name=val)

        return parsed_args

    def _validate_and_construct_plan(self, steps: List[ParsedStep]) -> ExecutionPlan:
        variable_lookup: Dict[str, Type[IOType]] = {}
        plan_nodes: List[ToolExecutionNode] = []
        for step in steps:
            if not self.tool_registry.is_tool_registered(step.function):
                raise ExecutionPlanParsingError(f"Invalid function '{step.function}'")

            tool = self.tool_registry.get_tool(step.function)
            partial_args = self._validate_tool_arguments(
                tool, args=step.arguments, variable_lookup=variable_lookup
            )
            variable_lookup[step.output_var] = tool.return_type

            node = ToolExecutionNode(
                tool_name=step.function,
                args=partial_args,
                description=step.description,
                output_variable_name=step.output_var,
                tool_task_id=str(uuid4()),
            )
            plan_nodes.append(node)
        if plan_nodes:
            plan_nodes[-1].is_output_node = True

        return ExecutionPlan(nodes=plan_nodes)

    def _parse_plan_str(self, plan_str: str) -> List[ParsedStep]:
        plan_steps: List[ParsedStep] = []
        for line in plan_str.split("\n"):
            if line.startswith("#"):  # allow initial comment line
                continue
            if line.startswith("```"):  # GPT likes to add this to Python code
                continue
            if not line.strip():
                continue

            match = ASSIGNMENT_RE.match(line)

            if match is None:
                raise ExecutionPlanParsingError(f"Failed to parse plan line: {line}")

            output_var, function, arguments, description = match.groups()
            arg_dict = get_arg_dict(arguments)
            plan_steps.append(
                ParsedStep(
                    output_var=output_var,
                    function=function,
                    arguments=arg_dict,
                    description=description,
                )
            )

        return plan_steps
