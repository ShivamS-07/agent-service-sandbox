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

import contextvars
import datetime
import enum
import functools
import inspect
import logging
import traceback
import uuid
from abc import ABC
from collections import defaultdict
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    ClassVar,
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

import backoff
from pydantic import BaseModel
from pydantic_core import PydanticUndefined

from agent_service.endpoints.models import Status
from agent_service.io_type_utils import (
    IOType,
    check_type_is_io_type,
    dump_io_type,
    get_clean_type_name,
    load_io_type,
)
from agent_service.io_types.stock import StockID
from agent_service.planner.errors import AgentExecutionError
from agent_service.types import AgentUserSettings, PlanRunContext
from agent_service.utils.cache_utils import (
    DEFAULT_CACHE_TTL,
    CacheBackend,
    RedisCacheBackend,
)
from agent_service.utils.date_utils import get_now_utc
from agent_service.utils.event_logging import log_event

CacheKeyType = str

logger = logging.getLogger(__name__)

TOOL_DEBUG_INFO: contextvars.ContextVar = contextvars.ContextVar("debug_info", default={})


async def log_tool_call_event(context: PlanRunContext, event_data: Dict[str, Any]) -> None:
    # Prevent circular imports
    from agent_service.utils.async_db import AsyncDB, SyncBoostedPG

    async_db = AsyncDB(pg=SyncBoostedPG(skip_commit=context.skip_db_commit))
    event_data["replay_id"] = str(uuid.uuid4())
    log_event(event_data=event_data, event_name="agent-service-tool-call")
    if context.skip_db_commit:
        return
    try:
        debug_info = event_data.get("debug_info")
        args = event_data.get("args")
        tool_name = event_data.get("tool_name")
        output = event_data.get("result")
        if args and tool_name:
            start_time_utc: datetime.datetime = event_data.get("start_time_utc")  # type: ignore
            end_time_utc: datetime.datetime = event_data.get("end_time_utc")  # type: ignore
            await async_db.insert_task_run_info(
                context=context,
                args=args,
                debug_info=debug_info,
                tool_name=tool_name,
                output=output,
                error_msg=event_data.get("error_msg"),
                start_time_utc=start_time_utc,
                end_time_utc=end_time_utc,
                replay_id=event_data["replay_id"],
            )
    except Exception:
        logger.exception("Failed to store tool call in agent.task_run_info table!")


UNDEFINED = "UNDEFINED_DEFAULT_VALUE"


class ToolArgMetadata(BaseModel):
    """
    Class to track additional metadata for arguments to tools.
    """

    # Hide the argument from the planner. Arguments that use this MUST have a
    # default value.
    hidden_from_planner: bool = False
    # These are used for showing the planner one thing while accepting something
    # else. Most generally applicable when e.g. a new argument is added that you
    # want to be required, but can't be for backwards compatibility reasons.
    planner_type_override: Optional[str] = None
    # We need two values here, since otherwise it's impossible to tell if the
    # default override is None or if it should be ignored.
    use_default_override_for_planner: bool = False
    planner_default_override: Optional[Any] = None


class ToolArgs(BaseModel, ABC):
    """
    Generic class for tool args. Note that ALL member variables of this class
    MUST be variants of IOType. This will be checked by the @tool decorator.
    """

    # Stores a mapping from the field name as a string to its metadata. Fields
    # not present will use the defaults for all metadata options.
    arg_metadata: ClassVar[Dict[str, ToolArgMetadata]] = {}

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
    update_instructions: Optional[str]
    is_output_tool: bool = False
    store_output: bool = True
    enabled: bool = True
    enabled_checker_func: Optional[Callable[[Optional[str], Optional[AgentUserSettings]], bool]] = (
        None
    )
    enabled_for_subplanner: bool = True

    def to_function_header(self) -> str:
        """
        Returns the tool as a function header for use by an LLM. E.g.

        'def my_func(t: Table, x: int, y: int = 2)'

        """
        args = []
        for var, info in self.input_type.model_fields.items():
            arg_metadata = self.input_type.arg_metadata.get(var, ToolArgMetadata())
            if arg_metadata.hidden_from_planner:
                continue
            if arg_metadata.planner_type_override is not None:
                clean_type_name = arg_metadata.planner_type_override
            else:
                clean_type_name = get_clean_type_name(info.annotation)

            if arg_metadata.use_default_override_for_planner:
                default = arg_metadata.planner_default_override
            else:
                default = info.default

            if default is PydanticUndefined or default == UNDEFINED:
                args.append(f"{var}: {clean_type_name}")
            else:
                args.append(f"{var}: {clean_type_name} = {default}")

        args_str = ", ".join(args)
        return f"def {self.name}({args_str}) -> {get_clean_type_name(self.return_type)}"


class ToolCategory(enum.StrEnum):
    NEWS = "news"
    STOCK = "stocks"
    STOCK_GROUPS = "stock groups"
    STOCK_FILTERS = "stock filters"
    STOCK_SENTIMENT = "stock sentiment"
    PEERS = "peers"
    LIST = "lists"
    DATES = "dates"
    TEXT_WRITER = "text writer"
    EARNINGS = "earnings"
    STATISTICS = "statistics"
    TABLE = "table"
    OUTPUT = "output"
    THEME = "theme"
    COMMENTARY = "commentary"
    PORTFOLIO = "portfolio"
    GRAPH = "graph"
    TEXT_RETRIEVAL = "text"
    CUSTOM_DOCS = "custom documents"
    AUTOMATION = "automation"
    KPI = "KPIs"
    SEC_FILINGS = "SEC Filings"
    COMPETITIVE_ANALYSIS = "competitive analysis"
    IDEAS = "brainstorm ideas"
    WEB = "web search"
    STRATEGY = "strategy"

    def get_description(self) -> str:
        if self == ToolCategory.GRAPH:
            return "Tools for doing graphs and charts"

        elif self == ToolCategory.STOCK:
            return "Tools for basic stock and ETF lookup"

        elif self == ToolCategory.STOCK_GROUPS:
            return (
                "Tools for related to grouping stocks, including sectors, industries, and countries"
            )

        elif self == ToolCategory.STOCK_FILTERS:
            return "Tools for filtering stocks"

        elif self == ToolCategory.STOCK_SENTIMENT:
            return "Tools for identifying the sentiment associated with stocks, including recommendations"

        elif self == ToolCategory.PEERS:
            return "Tools that involve company peers and competitors"

        elif self == ToolCategory.TEXT_RETRIEVAL:
            return "Other tools for general text retrieval, when specific type is not specified"

        elif self == ToolCategory.CUSTOM_DOCS:
            return (
                "Tools for retrieving custom (uploadied) user documents, including analyst reports"
            )

        elif self == ToolCategory.LIST:
            return "Tools for manipulating lists (of stocks, texts, etc.)"

        elif self == ToolCategory.DATES:
            return "Tools related to dates"

        elif self == ToolCategory.TEXT_WRITER:
            return "Tools that use LLMs to analyze text data and produce written texts"

        elif self == ToolCategory.EARNINGS:
            return "Tools that involve earnings calls"

        elif self == ToolCategory.STATISTICS:
            return (
                "Tools that work with the database of statistics, potentially relevant to "
                "any quantitative analysis"
            )

        elif self == ToolCategory.TABLE:
            return (
                "Tools that do operations over tables, required for most calculations"
                " and filtering and rankings of stocks based on statistics"
            )

        elif self == ToolCategory.OUTPUT:
            return "Tools that prepare final outputs for visualization"

        elif self == ToolCategory.NEWS:
            return "Tools that involve news"

        elif self == ToolCategory.THEME:
            return "Tools that involve macroeconomic themes"

        elif self == ToolCategory.COMMENTARY:
            return "Tools that involve writing commentary"

        elif self == ToolCategory.PORTFOLIO:
            return "Tools that involve user portfolios and watchlists"

        elif self == ToolCategory.AUTOMATION:
            return "Tools that involve automating tasks and notifying users"

        elif self == ToolCategory.KPI:
            return (
                "Tools that involve metrics that are company, market, industry, "
                "segment, or product specific. There are not general or broad line items "
                "applicable to all companies such as 'Revenue' or 'EPS', these are found "
                "under the 'statistics' category."
            )

        elif self == ToolCategory.SEC_FILINGS:
            return "Tools that involve SEC filings"

        elif self == ToolCategory.COMPETITIVE_ANALYSIS:
            return "Tools used in determining the relative ranking of companies in particular product markets"

        elif self == ToolCategory.IDEAS:
            return "Tools that involve brainstormed ideas"

        elif self == ToolCategory.WEB:
            return "Tools that involve web searching"

        elif self == ToolCategory.STRATEGY:
            return "Tools that relate to quantitative strategies/model from Boosted 1"

        raise ValueError(f"ToolCategory is set to an Unsupported or Unimplemented type: {self}")
        return ""


class ToolRegistry:
    """
    Stores all tools using a mapping from tool name to tool. Contains a map per
    tool category.
    """

    def __init__(self) -> None:
        self._REGISTRY_CATEGORY_MAP: Dict[ToolCategory, Dict[str, Tool]] = defaultdict(dict)
        self._REGISTRY_ALL_TOOLS_MAP: Dict[str, Tool] = {}
        self._TOOL_NAME_TO_CATEGORY: Dict[str, ToolCategory] = {}
        self._TOOL_NAME_TO_SOURCE_FILE: Dict[str, str] = {}

    def register_tool(self, tool: Tool, category: ToolCategory) -> None:
        this_tool_source = inspect.getfile(inspect.unwrap(tool.func))
        orig_tool_source = self._TOOL_NAME_TO_SOURCE_FILE.get(tool.name)
        if orig_tool_source:
            if orig_tool_source == this_tool_source:
                # We import in at least 2 ways so we trigger this more than once per tool
                return
            raise Exception(f"{tool.name=} is already in registry from: {orig_tool_source}")

        self._TOOL_NAME_TO_SOURCE_FILE[tool.name] = this_tool_source
        self._REGISTRY_CATEGORY_MAP[category][tool.name] = tool
        self._REGISTRY_ALL_TOOLS_MAP[tool.name] = tool
        self._TOOL_NAME_TO_CATEGORY[tool.name] = category

    def get_tool_in_category(self, tool_name: str, category: ToolCategory) -> Tool:
        return self._REGISTRY_CATEGORY_MAP[category][tool_name]

    def get_all_tools_in_category(self, category: ToolCategory) -> List[Tool]:
        return list(self._REGISTRY_CATEGORY_MAP[category].values())

    def get_tool(self, tool_name: str) -> Tool:
        return self._REGISTRY_ALL_TOOLS_MAP[tool_name]

    def is_tool_registered(self, tool_name: str) -> bool:
        return tool_name in self._REGISTRY_ALL_TOOLS_MAP

    def get_tool_str(
        self,
        user_id: Optional[str] = None,
        filter_input: bool = False,
        skip_list: Optional[List[ToolCategory]] = None,
        user_settings: Optional[AgentUserSettings] = None,
        using_subplanner: bool = False,
    ) -> str:
        """Returns a string representation of the tool library. The user_id is used to control which
        tools are visible to individual users. If filter_input is true, the string is going to be input
        to the tool filter and so we do not include the full description of each tool but do include the
        category names, so GPT can choose them. The skip list is a list of tool categories which will
        not be included, also used for tool filtering."""
        output = []
        for tool_category, tool_dict in self._REGISTRY_CATEGORY_MAP.items():
            if skip_list and tool_category in skip_list:
                continue
            tool_descriptions = []
            for tool in tool_dict.values():
                if not tool.enabled and not using_subplanner:
                    continue
                elif tool.enabled_checker_func and not (
                    tool.enabled_checker_func(user_id, user_settings)
                ):
                    continue
                    # If there is a checker, only continue if there is a user ID which results in a true checker
                elif using_subplanner and not tool.enabled_for_subplanner:
                    continue
                tool_descriptions.append(tool.to_function_header())
                if not filter_input:
                    tool_descriptions.append(f"# {tool.description}")
            if tool_descriptions:
                if filter_input:
                    output.append(f"## {tool_category}: {tool_category.get_description()}")
                else:
                    output.append(f"## {tool_category.get_description()}")

                output.extend(tool_descriptions)

        return "\n".join(output)


_DEFAULT_TOOL_REGISTRY = ToolRegistry()


def default_tool_registry() -> ToolRegistry:
    return _DEFAULT_TOOL_REGISTRY


def default_cache_key_func(tool_name: str, args: ToolArgs, _context: PlanRunContext) -> str:
    args_str = args.model_dump_json(serialize_as_any=True)
    return f"{tool_name}-{args_str}"


def _handle_tool_result(val: IOType, context: PlanRunContext) -> None:
    # We want to identify stocks in the output later, so store any stocks that
    # are produced by a tool for later matching.
    if isinstance(val, StockID):
        context.add_stocks_to_context([val])
    elif isinstance(val, list):
        stocks = [v for v in val if isinstance(v, StockID)]
        context.add_stocks_to_context(stocks)


def tool(
    description: str,
    category: ToolCategory = ToolCategory.STOCK,
    use_cache: bool = False,
    use_cache_fn: Optional[Callable[[T, PlanRunContext], bool]] = None,
    cache_key_fn: Callable[[str, T, PlanRunContext], CacheKeyType] = default_cache_key_func,
    cache_backend: Optional[CacheBackend] = None,
    cache_ttl: int = DEFAULT_CACHE_TTL,
    retries: int = 0,
    timeout_seconds: int = 6000,
    is_visible: bool = True,
    enabled: bool = True,
    enabled_checker_func: Optional[
        Callable[[Optional[str], Optional[AgentUserSettings]], bool]
    ] = None,
    reads_chat: bool = False,
    update_instructions: Optional[str] = None,
    tool_registry: Optional[ToolRegistry] = None,
    is_output_tool: bool = False,
    store_output: bool = True,
    enabled_for_subplanner: Optional[bool] = None,
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

    retries: An integer number of retries in case the task fails.

    timeout_seconds: An integer number of seconds, after which the tool run
      times out.

    is_visible: If true, the task is visible to the end user.

    enabled: If false, the tool is not registered for use.

    enabled_checker_func: A function which resolves to a boolean, will be resolved at planner time alongside context
      to determine if a tool should be enabled for specific user. If None, then it resolved as should be enabled

    tool_registry: A class type for the registry. Useful for testing or tiered registries.

    store_output: If true, stores the tool's output in a table for later lookups.

    reads_chat: If true, indicates that the tool reads the chat and so updates can occur
      without a replan

    update_instructions: Should be included for any tool which reads the chat or otherwise
      requires special consideration inside the action decider
    """

    tool_registry = tool_registry or default_tool_registry()

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
                start = get_now_utc().isoformat()
                event_data: Dict[str, Any] = {
                    "tool_name": tool_name,
                    "start_time_utc": start,
                    "args": args.model_dump_json(serialize_as_any=True),
                    "context": context.model_dump_json(),
                }

                @backoff.on_exception(
                    backoff.constant, exception=Exception, max_tries=retries + 1, logger=logger
                )
                async def call_func() -> IOType:
                    previous_debug_info = TOOL_DEBUG_INFO.get()
                    try:
                        result = await func(args, context)
                        event_data["end_time_utc"] = get_now_utc().isoformat()
                        event_data["result"] = dump_io_type(result)
                        debug_info = TOOL_DEBUG_INFO.get()
                        if debug_info:
                            event_data["debug_info"] = dump_io_type(debug_info)
                        if not previous_debug_info:
                            TOOL_DEBUG_INFO.set({})
                        else:
                            TOOL_DEBUG_INFO.set(previous_debug_info)
                        await log_tool_call_event(context=context, event_data=event_data)

                        return result
                    except Exception as e:
                        event_data["end_time_utc"] = get_now_utc().isoformat()
                        event_data["error_msg"] = traceback.format_exc()
                        event_data["status"] = (
                            e.result_status.value
                            if isinstance(e, AgentExecutionError)
                            else Status.ERROR.value
                        )
                        debug_info = TOOL_DEBUG_INFO.get()
                        if debug_info:
                            event_data["debug_info"] = dump_io_type(debug_info)
                        if not previous_debug_info:
                            TOOL_DEBUG_INFO.set({})
                        else:
                            TOOL_DEBUG_INFO.set(previous_debug_info)
                        await log_tool_call_event(context=context, event_data=event_data)
                        raise e

                if (
                    use_cache or (use_cache_fn and use_cache_fn(args, context))
                ) and not context.skip_task_cache:
                    new_val = None
                    try:
                        cache_client = (
                            cache_backend
                            if cache_backend
                            else RedisCacheBackend(
                                namespace="agent-tool-cache",
                                serialize_func=dump_io_type,
                                deserialize_func=load_io_type,
                                auto_close_connection=True,
                            )
                        )
                        key = cache_key_fn(tool_name, args, context)
                        cached_val = await cache_client.get(key, ttl=cache_ttl)
                        event_data["cache_key"] = key
                        if cached_val:
                            event_data["cache_hit"] = True
                            event_data["end_time_utc"] = get_now_utc().isoformat()
                            event_data["result"] = dump_io_type(cached_val)
                            await log_tool_call_event(context=context, event_data=event_data)
                            return cached_val

                        new_val = await func(args, context)
                        await cache_client.set(key=key, val=new_val, ttl=cache_ttl)
                        event_data["end_time_utc"] = get_now_utc().isoformat()
                        event_data["result"] = dump_io_type(new_val)
                        await log_tool_call_event(context=context, event_data=event_data)
                        return new_val
                    except Exception as e:
                        event_data["end_time_utc"] = get_now_utc().isoformat()
                        event_data["error_msg"] = traceback.format_exc()
                        event_data["status"] = (
                            e.result_status.value
                            if isinstance(e, AgentExecutionError)
                            else Status.ERROR.value
                        )
                        await log_tool_call_event(context=context, event_data=event_data)
                        logger.exception(f"Cache check failed for {(tool_name, args, context)}")
                        if new_val is not None:
                            await log_tool_call_event(context=context, event_data=event_data)
                            return new_val

                        return await call_func()
                else:
                    return await call_func()

            # RUN THE TOOL
            value = await main_func(args, context)
            _handle_tool_result(value, context)
            return value

        (tool_registry or default_tool_registry()).register_tool(
            Tool(
                name=func.__name__,
                func=wrapper,
                input_type=tool_args_type,
                return_type=sig.return_annotation,
                description=description,
                reads_chat=reads_chat,
                update_instructions=update_instructions,
                is_output_tool=is_output_tool,
                store_output=store_output,
                enabled=enabled,
                enabled_checker_func=enabled_checker_func,
                enabled_for_subplanner=enabled
                if enabled_for_subplanner is None
                else enabled_for_subplanner,
            ),
            category=category,
        )
        return wrapper

    return tool_deco
