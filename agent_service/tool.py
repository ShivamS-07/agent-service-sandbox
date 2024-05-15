"""
A tool represents a 'function' that can be included in an execution plan by an
LLM. Tools are essentially python functions wrapped in a decorator. For example:

class MyToolInput(ToolArgs):
    arg1: StockTimeseriesTable
    arg2: int = 1
    arg3: List[int] = [1, 2, 3]

@tool(description="My tool does XYZ")
def my_tool(args: MyToolInput, context: PlanRunContext) -> int:
    ...
"""

import enum
import functools
import inspect
from abc import ABC
from collections import defaultdict
from dataclasses import dataclass
from typing import (
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Type,
    TypeVar,
    Union,
    get_args,
    get_origin,
)

from prefect.tasks import Task
from pydantic import BaseModel
from pydantic_core import PydanticUndefined

from agent_service.io_type_utils import (
    IOType,
    check_type_is_io_type,
    get_clean_type_name,
)
from agent_service.types import PlanRunContext
from agent_service.utils.prefect import get_task_run_name

CacheKeyType = str


class ToolArgs(BaseModel, ABC):
    """
    Generic class for tool args. Note that ALL member variables of this class
    MUST be variants of IOType. This will be checked by the @tool decorator.
    """

    @classmethod
    def validate_types(cls) -> None:
        """
        This iterates through the model's fields and checks their types. If any
        field's type is not a primitive type or in the IOType enum, raise an
        error.
        """
        for var, info in cls.model_fields.items():
            var_type = info.annotation
            if get_origin(var_type) is Union:
                # Unwrap optionals to type check
                var_type = get_args(var_type)[0]

            if not check_type_is_io_type(var_type):
                raise ValueError(
                    (
                        f"Property '{var}' in class '{cls.__name__}'"
                        f" must be PrimitiveType or IOType, not {str(var_type)}."
                    )
                )


# See https://mypy.readthedocs.io/en/stable/generics.html#variance-of-generic-types
# Note that this is actually slightly incorrect, this time is actually NOT
# contravariant since not all subclasses of ToolArgs are interchangeable with
# BaseModel itself. However, for our cases, this should work fine, since the only
# methods we care about will always be present in both.
T = TypeVar("T", bound=ToolArgs, contravariant=True)


class ToolFunc(Protocol[T]):
    """
    Represents the type of a function that looks like this:

    def func(args: ToolArgsSubclass, context: PlanRunContext) -> IOType

    Names of arguments must be the same. Note that args can be any subclass of
    BaseModel.
    """

    __name__: str = ""

    async def __call__(self, args: T, context: PlanRunContext) -> IOType:
        raise NotImplementedError()


@dataclass(frozen=True)
class Tool:
    """
    Represents a tool as an input to an LLM. After an execution plan is
    validated, a ToolCallTask is created to represent the actual 'call' of the
    tool. Note that this class is only for in memory use, and cannot be serialized.
    """

    name: str
    func: ToolFunc
    input_type: Type[ToolArgs]
    return_type: Type[IOType]
    description: str

    def to_function_header(self) -> str:
        """
        Returns the tool as a function header for use by an LLM. E.g.

        'def my_func(t: StockTimeseriesTable, x: int, y: int = 2)'

        """
        args = []
        for var, info in self.input_type.model_fields.items():
            clean_type_name = get_clean_type_name(info.annotation)
            if info.default is PydanticUndefined:
                args.append(f"{var}: {clean_type_name}")
            else:
                args.append(f"{var}: {clean_type_name} = {info.default}")

        args_str = ", ".join(args)
        return f"def {self.name}({args_str}) -> {get_clean_type_name(self.return_type)}"


class ToolCategory(str, enum.Enum):
    MISC = "misc"
    NEWS = "news"
    STOCK = "stocks"
    LIST = "lists"
    DATES = "dates"
    LLM_ANALYSIS = "LLM analysis"
    USER = "user"
    EARNINGS = "earnings"
    STATISTICS = "statistics"
    TABLE = "table"
    OUTPUT = "output"
    THEME = "theme"

    def get_description(self) -> str:
        if self == ToolCategory.MISC:
            return "Other tools"

        if self == ToolCategory.STOCK:
            return "Tools for stock lookup"

        if self == ToolCategory.LIST:
            return "Tools for manipulating lists"

        if self == ToolCategory.DATES:
            return "Tools related to dates"

        if self == ToolCategory.LLM_ANALYSIS:
            return "Tools that use LLMs to analyze data"

        if self == ToolCategory.USER:
            return "Tools that get information about the user"

        if self == ToolCategory.EARNINGS:
            return "Tools that involve earnings calls"

        if self == ToolCategory.STATISTICS:
            return "Tools that work with the database of stock statistic"

        if self == ToolCategory.TABLE:
            return "Tools that do operations over tables"

        if self == ToolCategory.OUTPUT:
            return "Tools that prepare final outputs for visualization"

        if self == ToolCategory.NEWS:
            return "Tools that involve news"

        if self == ToolCategory.THEME:
            return "Tools that involve macroeconomic themes"

        return ""


class ToolRegistry:
    """
    Stores all tools using a mapping from tool name to tool. Contains a map per
    tool category.
    """

    _REGISTRY_CATEGORY_MAP: Dict[ToolCategory, Dict[str, Tool]] = defaultdict(dict)
    _REGISTRY_ALL_TOOLS_MAP: Dict[str, Tool] = {}

    @classmethod
    def register_tool(cls, tool: Tool, category: ToolCategory = ToolCategory.MISC) -> None:
        cls._REGISTRY_CATEGORY_MAP[category][tool.name] = tool
        cls._REGISTRY_ALL_TOOLS_MAP[tool.name] = tool

    @classmethod
    def get_tool_in_category(
        cls, tool_name: str, category: ToolCategory = ToolCategory.MISC
    ) -> Tool:
        return cls._REGISTRY_CATEGORY_MAP[category][tool_name]

    @classmethod
    def get_all_tools_in_category(cls, category: ToolCategory = ToolCategory.MISC) -> List[Tool]:
        return list(cls._REGISTRY_CATEGORY_MAP[category].values())

    @classmethod
    def get_tool(cls, tool_name: str) -> Tool:
        return cls._REGISTRY_ALL_TOOLS_MAP[tool_name]

    @classmethod
    def is_tool_registered(cls, tool_name: str) -> bool:
        return tool_name in cls._REGISTRY_ALL_TOOLS_MAP

    @classmethod
    def get_tool_str(cls) -> str:
        output = []
        for tool_category, tool_dict in cls._REGISTRY_CATEGORY_MAP.items():
            output.append(f"## {tool_category.get_description()}")
            for tool in tool_dict.values():
                output.append(tool.to_function_header())
                output.append(f"# {tool.description}")
        return "\n".join(output)


def tool(
    description: str,
    category: ToolCategory = ToolCategory.MISC,
    use_cache: bool = True,
    use_cache_fn: Optional[Callable[[T, PlanRunContext], bool]] = None,  # TODO default
    cache_key_fn: Optional[Callable[[T, PlanRunContext], CacheKeyType]] = None,  # TODO default
    retries: int = 0,  # TODO default
    timeout_seconds: int = 6000,  # TODO default
    create_prefect_task: bool = True,
    is_visible: bool = True,
    enabled: bool = True,
    tool_registry: Type[ToolRegistry] = ToolRegistry,
) -> Callable[[ToolFunc], ToolFunc]:
    """
    Decorator to register a function as a Tool usable by GPT. This can only decorate a function of the format:
        def func(args: ToolArgsSubclass, context: PlanRunContext) -> IOType

    description: A description of the tool. This is required, as it is used in
      addition to the function's signature to explain to GPT what the tool does.

    category: A way to group tools into categories, useful for GPT.

    use_cache: If true, tool output is cached.

    use_cache_fn: Function of the inputs. If it evaluates to true, tool output is cached.

    cache_key_fn: Function of the inputs. Evaluates to a string cache key to
      potentially retrieve cached values. If caching is disabled, this is ignored.

    cache_expiration: Timedelta representing amount of time the output should
      exist in the cache.

    retries: An integer number of retries in case the task fails. NOTE: Only
      respected when `create_prefect_task` is also True.

    timeout_seconds: An integer number of seconds, after which the tool run
      times out. NOTE: only respected when `create_prefect_task` is True.

    create_prefect_task: If true, wraps the tool run in a prefect
      task. Otherwise run locally without prefect.

    is_visible: If true, the task is visible to the end user.

    enabled: If false, the tool is not registered for use.

    tool_registry: A class type for the registry. Useful for testing or tiered registries.
    """

    def tool_deco(func: ToolFunc) -> ToolFunc:
        # Inspect the function's arguments and return type to ensure they are
        # all valid.
        sig = inspect.signature(func)
        if not check_type_is_io_type(sig.return_annotation):
            raise ValueError(
                (
                    f"Tool function '{func.__name__}' has invalid return type "
                    f"{str(sig.return_annotation)}. Return type must be an 'IOType'."
                )
            )
        tool_args_type = None
        for param in sig.parameters.values():
            typ = param.annotation
            # If the parameter is a subclass of ToolArgs, make sure all
            # properties are IOType using the validate_types class method.
            if issubclass(typ, ToolArgs):
                typ.validate_types()
                tool_args_type = typ
            elif not issubclass(typ, PlanRunContext):
                raise ValueError(
                    (
                        f"Tool function '{func.__name__}' has argument "
                        f"'{param.name}' with invalid type {str(typ)}. "
                        "Tool functions must accept only a 'ToolArgs'"
                        " argument and a 'PlanRunContext' argument."
                    )
                )

        if tool_args_type is None:
            raise ValueError(
                f"Tool function '{func.__name__}' is missing argument of type 'ToolArgs'."
            )
        # Add the tool to the registry
        if enabled:
            tool_registry.register_tool(
                Tool(
                    name=func.__name__,
                    func=func,
                    input_type=tool_args_type,
                    return_type=sig.return_annotation,
                    description=description,
                ),
                category=category,
            )

        @functools.wraps(func)
        async def wrapper(args: T, context: PlanRunContext) -> IOType:
            # Wrap any logic in another function. This will ensure e.g. caching
            # is executed as part of the prefect task, and not before the task
            # runs.
            async def main_func(args: T, context: PlanRunContext) -> IOType:
                if (
                    use_cache or (use_cache_fn and use_cache_fn(args, context))
                ) and not context.skip_task_cache:
                    # TODO: HANDLE CACHING
                    return await func(args, context)
                return await func(args, context)

            if create_prefect_task and not context.run_tasks_without_prefect:
                # Create a prefect task that wraps the function with its caching
                # logic. This will ensure that retried tasks will include caching.
                tags = None
                if not is_visible:
                    tags = ["hidden", "minitool"]
                task = Task(
                    name=func.__name__,
                    task_run_name=get_task_run_name(ctx=context),
                    fn=main_func,
                    description=description,
                    retries=retries,
                    timeout_seconds=timeout_seconds,
                    tags=tags,
                )

                value = await task(args, context)
            else:
                # Otherwise, run locally without prefect running at all
                value = await main_func(args, context)
            return value

        return wrapper

    return tool_deco
