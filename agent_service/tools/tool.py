"""
A tool represents a 'function' that can be included in an execution plan by an
LLM. Tools are essentially python functions wrapped in a decorator. For example:

class MyToolInput(ToolArgs):
    arg1: IntIO = 1  # For primitives, you can assign directly instead of IntIO(val=1)
    arg2: ListIO[int] = ListIO[int](vals=[1, 2, 3])

@tool(description="My tool does XYZ")
def my_tool(args: MyToolInput, context: PlanRunContext) -> IntIO:
    ...


"""

import datetime
import functools
import inspect
from abc import ABC
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Protocol, Type, TypeVar, get_args

from prefect.tasks import Task
from pydantic import BaseModel
from pydantic_core import PydanticUndefined

from agent_service.tools.io_types import IOType, PrimitiveType
from agent_service.types import PlanRunContext

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
            if var_type in get_args(PrimitiveType):
                continue

            if not var_type in get_args(IOType):
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

    __name__ = ""

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
    readable_name: str
    func: ToolFunc
    input_type: Type[ToolArgs]
    return_type: Type

    def to_function_header(self) -> str:
        """
        Returns the tool as a function header for use by an LLM. E.g.

        'def my_func(x: int, y: int = 2)'

        """
        args = []
        for var, info in self.input_type.model_fields.items():
            if not info.default or info.default is PydanticUndefined:
                args.append(f"{var}: {info.annotation.gpt_type_name()}")  # type: ignore
            else:
                args.append(f"{var}: {info.annotation.gpt_type_name()} = {info.default}")  # type: ignore

        args_str = ", ".join(args)
        return f"def {self.name}({args_str}) -> {self.return_type.gpt_type_name()}"


class ToolRegistry:
    """
    Stores all tools using a mapping from tool name to tool.
    """

    _REGISTRY_MAP: Dict[str, Tool] = {}

    @classmethod
    def register_tool(cls, tool: Tool) -> None:
        cls._REGISTRY_MAP[tool.name] = tool

    @classmethod
    def get_tool(cls, tool_name: str) -> Tool:
        return cls._REGISTRY_MAP[tool_name]

    @classmethod
    def get_all_tools(cls) -> List[Tool]:
        return list(cls._REGISTRY_MAP.values())


class ToolTask(BaseModel):
    """
    Represents a tool run in an execution plan. Execution plans that are
    serialized are composed of a graph of ToolTasks.

    Note: any additional info about the tool may be fetched from the
    ToolRegistry using the tool_name.
    """

    tool_name: str
    input_val: ToolArgs

    async def execute(
        self, context: PlanRunContext, registry: Type[ToolRegistry] = ToolRegistry
    ) -> IOType:
        tool = registry.get_tool(self.tool_name)
        return await tool.func(args=self.input_val, context=context)


def tool(
    description: str,
    readable_name: Optional[str] = None,
    use_cache: bool = True,
    use_cache_fn: Optional[Callable[[T, PlanRunContext], bool]] = None,  # TODO default
    cache_key_fn: Optional[Callable[[T, PlanRunContext], CacheKeyType]] = None,  # TODO default
    cache_expiration: Optional[
        datetime.timedelta
    ] = None,  # TODO default + how to handle with postgres?
    retries: int = 0,  # TODO default
    timeout_seconds: int = 6000,  # TODO default
    create_prefect_task: bool = True,
    is_visible: bool = True,
    enabled: bool = True,
) -> Callable[[ToolFunc], ToolFunc]:
    """
    Decorator to register a function as a Tool usable by GPT. This can only decorate a function of the format:
        def func(args: ToolArgsSubclass, context: PlanRunContext) -> IOType

    description: A description of the tool. This is required, as it is used in
      addition to the function's signature to explain to GPT what the tool does.

    readable_name: A readable name that can be presented to the end user
      explaining what the task does. If absent, defaults to the function's name.

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
    """

    def tool_deco(func: ToolFunc) -> ToolFunc:
        # Inspect the function's arguments and return type to ensure they are
        # all valid.
        sig = inspect.signature(func)
        if sig.return_annotation not in get_args(IOType):
            raise ValueError(
                (
                    f"Tool function f'{func.__name__}' has invalid return type "
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
                        f"Tool function f'{func.__name__}' has argument "
                        f"'{param.name}' with invalid type {str(typ)}. "
                        "Tool functions must accept only a 'ToolArgs'"
                        " argument and a 'PlanRunContext' argument."
                    )
                )

        if tool_args_type is None:
            raise ValueError(
                f"Tool function f'{func.__name__}' is missing argument of type 'ToolArgs'."
            )
        # Add the tool to the registry
        if enabled:
            ToolRegistry.register_tool(
                Tool(
                    name=func.__name__,
                    readable_name=readable_name or func.__name__,
                    func=func,
                    input_type=tool_args_type,
                    return_type=sig.return_annotation,
                )
            )

        @functools.wraps(func)
        async def wrapper(args: ToolArgs, context: PlanRunContext) -> IOType:
            # Wrap any logic in another function. This will ensure e.g. caching
            # is executed as part of the prefect task, and not before the task
            # runs.
            async def main_func(args: ToolArgs, context: PlanRunContext) -> IOType:
                if use_cache or (use_cache_fn and use_cache_fn(args, context)):
                    # TODO: HANDLE CACHING
                    return await func(args, context)
                return await func(args, context)

            if create_prefect_task:
                # Create a prefect task that wraps the function with its caching
                # logic. This will ensure that retried tasks will include caching.
                tags = None
                if not is_visible:
                    tags = ["hidden", "minitool"]
                task = Task(
                    name=func.__name__,
                    task_run_name=context.task_id,
                    fn=main_func,
                    description=description,
                    retries=retries,
                    timeout_seconds=timeout_seconds,
                    tags=tags,
                )

                value = await task(args, context)
            else:
                value = await main_func(args, context)
            return value

        return wrapper

    return tool_deco
