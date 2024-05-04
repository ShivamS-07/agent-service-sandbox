import datetime
import functools
import inspect
from dataclasses import dataclass
from typing import Callable, Optional, Protocol, Type, TypeVar, get_args

from prefect.tasks import Task
from pydantic.main import BaseModel

from agent_service.tools.io_types import IOType, IntIO, ListIO
from agent_service.types import PlanRunContext

"""
A Tool:
    knows its input types
    knows its output types
    knows its caching strategies
    knows its retries
    knows if visible

A ToolCallTask:
    knows its INPUTS
    knows its TOOL
"""

CacheKeyType = str


class ToolArgs(BaseModel):
    """
    Generic class for tool args. Note that ALL member variables of this class
    MUST be variants of IOType. This will be checked by the @tool decorator.
    """

    @classmethod
    def validate_types(cls) -> None:
        """
        This iterates through the model's fields and checks their types. If any
        field's type is not in the IOType enum, raise an error.
        """
        for var, info in cls.model_fields.items():
            var_type = info.annotation
            if not var_type in get_args(IOType):
                raise ValueError(
                    (
                        f"Property '{var}' in class '{cls.__name__}'"
                        f" must be IOType, not {str(var_type)}."
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

    def func(args: BaseModelSubclass, context: PlanRunContext) -> IOType

    Names of arguments must be the same. Note that args can be any subclass of
    BaseModel.
    """

    __name__ = ""

    async def __call__(self, args: T, context: PlanRunContext) -> IOType:
        raise NotImplementedError()


@dataclass(frozen=True)
class Tool:
    func: ToolFunc
    input_type: Type[ToolArgs]


class ToolRegistry:
    _REGISTRY_MAP = {}

    @classmethod
    def register_tool(cls, tool: ToolFunc) -> None:
        pass


def tool(
    readable_name: Optional[str] = None,
    description: Optional[str] = None,
    use_cache: bool = False,
    use_cache_fn: Optional[Callable[[T, PlanRunContext], bool]] = None,
    cache_key_fn: Optional[Callable[[T, PlanRunContext], CacheKeyType]] = None,
    cache_expiration: Optional[datetime.timedelta] = None,
    retries: int = 0,
    timeout_seconds: int = 6000,  # or whatever
    create_prefect_task: bool = True,
) -> Callable[[ToolFunc], ToolFunc]:
    def tool_deco(func: ToolFunc) -> ToolFunc:
        # Inspect the function's arguments at runtime, to ensure the ToolArgs
        # parameter is valid.
        sig = inspect.signature(func)
        for param in sig.parameters.values():
            typ = param.annotation
            # If the parameter is a subclass of ToolArgs, make sure all
            # properties are IOType using the validate_types class method.
            if issubclass(typ, ToolArgs):
                typ.validate_types()

        @functools.wraps(func)
        async def wrapper(args: ToolArgs, context: PlanRunContext) -> IOType:
            # Wrap any logic in another function. This will ensure e.g. caching
            # is executed as part of the prefect task, and not before the task
            # runs.
            async def sub_wrapper(args: ToolArgs, context: PlanRunContext) -> IOType:
                if use_cache or (use_cache_fn and use_cache_fn(args, context)):
                    # TODO: HANDLE CACHING
                    return await func(args, context)
                return await func(args, context)

            # Create a prefect task that wraps the function with its caching
            # logic. This will ensure that retried tasks will include caching.
            if create_prefect_task:
                task = Task(
                    name=readable_name or func.__name__,  # TODO
                    fn=sub_wrapper,
                    description=description or "",
                    retries=retries,
                    timeout_seconds=timeout_seconds,
                )

                value = await task(args, context)
            else:
                value = await sub_wrapper(args, context)
            return value

        return wrapper

    return tool_deco


class MyToolInput(ToolArgs):
    x: int
    y: str


class MyToolInput2(ToolArgs):
    x: IntIO
    y: ListIO


@tool()
async def my_tool_2(args: MyToolInput2, context: PlanRunContext) -> IOType:
    return args  # Placeholder logic for demonstration purposes
