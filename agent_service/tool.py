"""
A tool represents a 'function' that can be included in an execution plan by an
LLM. Tools are essentially python functions wrapped in a decorator. For example:

class MyToolInput(ToolArgs):
    arg1: Table
    arg2: int = 1
    arg3: List[int] = [1, 2, 3]

@tool(description="My tool does XYZ")
def my_tool(args: MyToolInput, context: PlanRunContext) -> int:
    ...
"""

import datetime
import enum
import functools
import inspect
import logging
import traceback
from abc import ABC
from collections import defaultdict
from dataclasses import dataclass
from typing import (
    Any,
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

from gbi_common_py_utils.utils.event_logging import log_event
from prefect.tasks import Task
from pydantic import BaseModel
from pydantic_core import PydanticUndefined

from agent_service.GPT.constants import CLIENT_NAMESPACE
from agent_service.io_type_utils import (
    IOType,
    check_type_is_io_type,
    get_clean_type_name,
)
from agent_service.types import PlanRunContext
from agent_service.utils.cache_utils import (
    DEFAULT_CACHE_TTL,
    CacheBackend,
    RedisCacheBackend,
)
from agent_service.utils.prefect import get_task_run_name

CacheKeyType = str

logger = logging.getLogger(__name__)


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
    reads_chat: bool

    def to_function_header(self) -> str:
        """
        Returns the tool as a function header for use by an LLM. E.g.

        'def my_func(t: Table, x: int, y: int = 2)'

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
    COMMENTARY = "commentary"
    GRAPH = "graph"
    TEXT = "text"
    AUTOMATION = "automation"
    KPI = "key performance indicators"

    def get_description(self) -> str:
        if self == ToolCategory.MISC:
            return "Other tools"

        if self == ToolCategory.STOCK:
            return "Tools for stock lookup"

        if self == ToolCategory.TEXT:
            return "Other tools for retrieving texts"

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

        if self == ToolCategory.GRAPH:
            return "Tools that create or deal with graphs"

        if self == ToolCategory.OUTPUT:
            return "Tools that prepare final outputs for visualization"

        if self == ToolCategory.NEWS:
            return "Tools that involve news"

        if self == ToolCategory.THEME:
            return "Tools that involve macroeconomic themes"

        if self == ToolCategory.COMMENTARY:
            return "Tools that involve writing commentary"

        if self == ToolCategory.AUTOMATION:
            return "Tools that involve automating tasks and notifying users"

        if self == ToolCategory.KPI:
            return (
                "Tools that involve metrics that are company, market, industry, "
                "segment, or product specific. There are not general or broad line items "
                "applicable to all companies such as 'Revenue' or 'EPS', these are found "
                "under the 'statistics' category."
            )

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
    def does_tool_read_chat(cls, tool_name: str) -> bool:
        return cls._REGISTRY_ALL_TOOLS_MAP[tool_name].reads_chat

    @classmethod
    def get_tool_str(cls) -> str:
        output = []
        for tool_category, tool_dict in cls._REGISTRY_CATEGORY_MAP.items():
            output.append(f"## {tool_category.get_description()}")
            for tool in tool_dict.values():
                output.append(tool.to_function_header())
                output.append(f"# {tool.description}")
        return "\n".join(output)


def default_cache_key_func(tool_name: str, args: ToolArgs, _context: PlanRunContext) -> str:
    args_str = args.model_dump_json()
    return f"{tool_name}-{args_str}"


def tool(
    description: str,
    category: ToolCategory = ToolCategory.MISC,
    use_cache: bool = False,
    use_cache_fn: Optional[Callable[[T, PlanRunContext], bool]] = None,
    cache_key_fn: Callable[[str, T, PlanRunContext], CacheKeyType] = default_cache_key_func,
    cache_backend: Optional[CacheBackend] = None,
    cache_ttl: int = DEFAULT_CACHE_TTL,
    retries: int = 0,
    timeout_seconds: int = 6000,
    create_prefect_task: bool = True,
    is_visible: bool = True,
    enabled: bool = True,
    reads_chat: bool = False,
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

    cache_key_fn: Function of the tool's name and its two inputs. Evaluates to a
      string cache key to potentially retrieve cached values. If caching is
      disabled, this is ignored.

    cache_backend: The backend for caching, defaults to redis if None. If caching is
      disabled, this is ignored.

    cache_ttl: Integer number of seconds for the cached value's TTL. NOTE:
      if postgres is is used as the cache, this value is NOT USED.

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

        @functools.wraps(func)
        async def wrapper(args: T, context: PlanRunContext) -> IOType:
            # Wrap any logic in another function. This will ensure e.g. caching
            # is executed as part of the prefect task, and not before the task
            # runs.
            tool_name = func.__name__

            async def main_func(args: T, context: PlanRunContext) -> IOType:
                start = datetime.datetime.utcnow().isoformat()
                event_data: Dict[str, Any] = {
                    "tool_name": tool_name,
                    "agent_id": context.agent_id,
                    "user_id": context.user_id,
                    "task_id": context.task_id,
                    "plan_id": context.plan_id,
                    "plan_run_id": context.plan_run_id,
                    "start_time_utc": start,
                }

                async def call_func() -> IOType:
                    try:
                        result = await func(args, context)
                        event_data["end_time_utc"] = datetime.datetime.utcnow().isoformat()
                        if CLIENT_NAMESPACE != "LOCAL":
                            log_event(event_name="agent-service-tool-call", event_data=event_data)

                        return result
                    except Exception as e:
                        event_data["end_time_utc"] = datetime.datetime.utcnow().isoformat()
                        event_data["error_msg"] = traceback.format_exc()
                        if CLIENT_NAMESPACE != "LOCAL":
                            log_event(event_name="agent-service-tool-call", event_data=event_data)
                        raise e

                if (
                    use_cache or (use_cache_fn and use_cache_fn(args, context))
                ) and not context.skip_task_cache:
                    new_val = None
                    try:
                        cache_client = cache_backend if cache_backend else RedisCacheBackend()
                        key = cache_key_fn(tool_name, args, context)
                        cached_val = await cache_client.get(key)
                        event_data["cache_key"] = key
                        if cached_val:
                            event_data["cache_hit"] = True
                            event_data["end_time_utc"] = datetime.datetime.utcnow().isoformat()
                            if CLIENT_NAMESPACE != "LOCAL":
                                log_event(
                                    event_name="agent-service-tool-call", event_data=event_data
                                )
                            return cached_val

                        new_val = await func(args, context)
                        await cache_client.set(key=key, val=new_val, ttl=cache_ttl)
                        event_data["end_time_utc"] = datetime.datetime.utcnow().isoformat()
                        if CLIENT_NAMESPACE != "LOCAL":
                            log_event(event_name="agent-service-tool-call", event_data=event_data)
                        return new_val
                    except Exception:
                        event_data["end_time_utc"] = datetime.datetime.utcnow().isoformat()
                        event_data["error_msg"] = traceback.format_exc()
                        if CLIENT_NAMESPACE != "LOCAL":
                            log_event(event_name="agent-service-tool-call", event_data=event_data)
                        logger.exception(f"Cache check failed for {(tool_name, args, context)}")
                        if new_val is not None:
                            log_event(event_name="agent-service-tool-call", event_data=event_data)
                            return new_val

                        return await call_func()
                else:
                    return await call_func()

            if create_prefect_task and not context.run_tasks_without_prefect:
                # Create a prefect task that wraps the function with its caching
                # logic. This will ensure that retried tasks will include caching.
                tags = None
                if not is_visible:
                    tags = ["hidden", "minitool"]
                task = Task(
                    name=tool_name,
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

        # Add the tool to the registry
        if enabled:
            tool_registry.register_tool(
                Tool(
                    name=func.__name__,
                    func=wrapper,
                    input_type=tool_args_type,
                    return_type=sig.return_annotation,
                    description=description,
                    reads_chat=reads_chat,
                ),
                category=category,
            )
        return wrapper

    return tool_deco
