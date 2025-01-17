import asyncio
import contextvars
import copy
import json
import logging
import os
import traceback
import uuid
from typing import Any, Dict, List, Optional, Tuple, Union

from gbi_common_py_utils.utils.environment import (
    DEV_TAG,
    LOCAL_TAG,
    get_environment_tag,
)
from google.protobuf.struct_pb2 import Struct
from gpt_service_proto_v1.service_grpc import GPTServiceStub
from gpt_service_proto_v1.service_pb2 import (
    EmbedTextRequest,
    EmbedTextResponse,
    QueryGPTRequest,
    QueryGPTResponse,
)
from grpclib.client import Channel

from agent_service.external.grpc_utils import dont_retry, grpc_retry
from agent_service.GPT.constants import (
    CLIENT_NAME,
    CLIENT_NAMESPACE,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_SMART_MODEL,
    DEFAULT_TEMPERATURE,
    JSON_RESPONSE_FORMAT,
    MAX_GPT_WORKER_TIMEOUT,
    TEXT_RESPONSE_FORMAT,
    TIMEOUTS,
)
from agent_service.types import PlanRunContext
from agent_service.unit_test_util import RUNNING_IN_UNIT_TEST
from agent_service.utils.date_utils import get_now_utc
from agent_service.utils.event_logging import log_event
from agent_service.utils.gpt_logging import (
    GPT_TASK_TYPE,
    MAIN_PROMPT_NAME,
    MAIN_PROMPT_TEMPLATE_ARGS,
    MAIN_PROMPT_TEMPLATE_NAME,
    SYS_PROMPT_NAME,
    SYS_PROMPT_TEMPLATE_ARGS,
    SYS_PROMPT_TEMPLATE_NAME,
    get_gpt_task_type,
)
from agent_service.utils.prompt_utils import FilledPrompt

logger = logging.getLogger(__name__)

DEFAULT_PRIORITY = os.getenv("GPT_SERVICE_DEFAULT_PRIORITY", "LOW")
STUB = None
CHANNEL = None
PLAN_RUN_CONTEXT: contextvars.ContextVar[Dict[str, Union[bool, str, None]]] = (
    contextvars.ContextVar("PLAN_RUN_CONTEXT", default={})
)
# unit tests that inherit from IsolatedAsyncioTestCase broken by this stub cache because
# IsolatedAsyncioTestCase creates a new event loop for each test case and thus we need a new
# connection for each test case
USE_GLOBAL_STUB = os.getenv("USE_GLOBAL_STUB", "") != ""


def use_global_stub() -> bool:
    return USE_GLOBAL_STUB


def set_use_global_stub(val: bool) -> None:
    global USE_GLOBAL_STUB
    USE_GLOBAL_STUB = val


def set_plan_run_context(context: PlanRunContext, scheduled_by_automation: bool) -> None:
    # ContextVar is context safe/local and not shared between different threads or concurrencies
    plan_run_context = {
        "plan_id": context.plan_id,
        "plan_run_id": context.plan_run_id,
        "task_id": context.task_id,
        "user_id": context.user_id,
        "tool_name": context.tool_name,
        "agent_id": context.agent_id,
        "scheduled_by_automation": scheduled_by_automation,
    }
    PLAN_RUN_CONTEXT.set(plan_run_context)


def _get_gpt_service_stub(context: Optional[Dict] = None) -> Tuple[GPTServiceStub, Channel]:
    global STUB
    global CHANNEL
    if STUB and use_global_stub():
        return STUB, CHANNEL
    url = os.getenv("GPT_SERVICE_URL", "gpt-service-2.boosted.ai:50051")
    host, port = url.split(":")
    CHANNEL = Channel(host=host, port=int(port))
    STUB = GPTServiceStub(CHANNEL)
    stack_trace = traceback.format_stack()
    stack_trace_str = "".join(stack_trace)
    if not context:
        context = {}
    log_event(
        event_name="agent_service_gpt_service_connection_created",
        event_data={"stack_trace": stack_trace_str, "context": context},
    )
    return STUB, CHANNEL


async def query_gpt_worker(
    model: str,
    main_prompt: str,
    sys_prompt: str,
    temperature: float,
    context: Optional[Dict[str, str]] = None,
    max_tokens: Optional[int] = None,
    output_json: bool = False,
    retry_num: int = 1,
    max_retries: int = 6,
    request_id: Optional[str] = None,
    no_cache: bool = False,
    gpt_service_stub: Optional[GPTServiceStub] = None,
) -> str:
    if not request_id:
        request_id = str(uuid.uuid4())
    client_timestamp = get_now_utc().isoformat()
    try:
        return await _query_gpt_worker(
            model=model,
            main_prompt=main_prompt,
            sys_prompt=sys_prompt,
            temperature=temperature,
            context=context,
            max_tokens=max_tokens,
            output_json=output_json,
            request_id=request_id,
            client_timestamp=client_timestamp,
            no_cache=no_cache,
            gpt_service_stub=gpt_service_stub,
        )
    except Exception as e:
        exception_text = traceback.format_exc()
        log_event(
            event_name="GPTService-Response-Error",
            event_data={
                "error_msg": exception_text,
                "send_timestamp": client_timestamp,
                "retry_number": retry_num,
                "request_id": request_id,
                "give_up_timestamp": get_now_utc().isoformat(),
                "model_id": model,
            },
        )

        if retry_num == max_retries or dont_retry(e):
            # Immediately raise, no further retries
            raise e
        else:
            await asyncio.sleep(1)
            return await query_gpt_worker(
                model=model,
                main_prompt=main_prompt,
                sys_prompt=sys_prompt,
                temperature=temperature,
                context=context,
                max_tokens=max_tokens,
                output_json=output_json,
                retry_num=retry_num + 1,
                max_retries=max_retries,
                request_id=request_id,
                no_cache=no_cache,
            )


@grpc_retry
async def _query_gpt_worker(
    model: str,
    main_prompt: str,
    sys_prompt: str,
    temperature: float,
    context: Optional[Dict[str, str]] = None,
    max_tokens: Optional[int] = None,
    output_json: bool = False,
    request_id: str = "",
    client_timestamp: Optional[str] = None,
    no_cache: bool = False,
    gpt_service_stub: Optional[GPTServiceStub] = None,
) -> str:
    plan_run_context = PLAN_RUN_CONTEXT.get()
    if not client_timestamp:
        client_timestamp = get_now_utc().isoformat()
    priority = DEFAULT_PRIORITY
    context_struct = Struct()
    if context is not None:
        context_struct.update(context)
    if plan_run_context:
        context_struct.update(plan_run_context)
    extra_params = Struct()
    env = get_environment_tag()
    is_dev = env in (DEV_TAG, LOCAL_TAG)
    extra_params_dict: Dict[str, Any] = {
        "context": context_struct,
        "is_dev": is_dev,
        "gpt_params": {"response_format": TEXT_RESPONSE_FORMAT},
    }
    if output_json:
        extra_params_dict["gpt_params"]["response_format"] = JSON_RESPONSE_FORMAT
    extra_params.update(extra_params_dict)
    request = QueryGPTRequest(
        model=model,
        main_prompt=main_prompt,
        sys_prompt=sys_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        request_priority=f"GPT_SVC_PRIORITY_{priority}",
        extra_params=extra_params,
    )
    stub, channel = _get_gpt_service_stub() if not gpt_service_stub else (gpt_service_stub, None)
    metadata = [
        ("clienttimestamp", client_timestamp),
        ("clientname", CLIENT_NAME),
        ("clientnamespace", CLIENT_NAMESPACE),
        ("clientrequestid", request_id),
    ]

    if no_cache or os.environ.get("NO_GPT_CACHE", "0") == "1":
        metadata.append(("nocache", "true"))

    result: QueryGPTResponse = await stub.QueryGPT(
        request, timeout=MAX_GPT_WORKER_TIMEOUT, metadata=metadata
    )

    if not use_global_stub() and channel:
        # explicitly close the channel after each call during unittests
        channel.close()

    if result.status.code != 0:
        raise RuntimeError(f"Error response from GPT service: {result.status.message}")

    return result.response


async def get_embedding(
    model: str,
    text: str,
    retry_num: int = 1,
    max_retries: int = 3,
    request_id: Optional[str] = None,
    no_cache: bool = False,
    gpt_service_stub: Optional[GPTServiceStub] = None,
) -> List[float]:
    if not request_id:
        request_id = str(uuid.uuid4())
    client_timestamp = get_now_utc().isoformat()
    try:
        return await _get_embedding(
            model=model,
            text=text,
            request_id=request_id,
            client_timestamp=client_timestamp,
            no_cache=no_cache,
            gpt_service_stub=gpt_service_stub,
        )
    except Exception as e:
        exception_text = traceback.format_exc()
        log_event(
            event_name="GPTService-Embed-Error",
            event_data={
                "error_msg": exception_text,
                "send_timestamp": client_timestamp,
                "retry_number": retry_num,
                "request_id": request_id,
                "give_up_timestamp": get_now_utc().isoformat(),
                "model_id": model,
            },
        )
        if retry_num == max_retries:
            raise e
        else:
            await asyncio.sleep(1)
            return await get_embedding(
                model=model,
                text=text,
                retry_num=retry_num + 1,
                max_retries=max_retries,
                request_id=request_id,
                no_cache=no_cache,
            )


@grpc_retry
async def _get_embedding(
    model: str,
    text: str,
    request_id: str = "",
    client_timestamp: Optional[str] = None,
    no_cache: bool = False,
    gpt_service_stub: Optional[GPTServiceStub] = None,
) -> List[float]:
    if not client_timestamp:
        client_timestamp = get_now_utc().isoformat()

    request = EmbedTextRequest(
        model=model,
        text=text,
    )
    stub = _get_gpt_service_stub() if not gpt_service_stub else gpt_service_stub
    metadata = [
        ("clienttimestamp", client_timestamp),
        ("clientname", CLIENT_NAME),
        ("clientnamespace", CLIENT_NAMESPACE),
        ("clientrequestid", request_id),
    ]

    if no_cache or os.environ.get("NO_GPT_CACHE", "0") == "1":
        metadata.append(("nocache", "true"))

    stub, channel = _get_gpt_service_stub()
    result: EmbedTextResponse = await stub.EmbedText(
        request, timeout=TIMEOUTS.get(model, MAX_GPT_WORKER_TIMEOUT), metadata=metadata
    )

    if not use_global_stub():
        # explicitly close the channel after each call during unittests
        channel.close()

    if result.status.code != 0:
        raise RuntimeError(f"Error response from GPT service: {result.status.message}")

    return [float(i) for i in result.embedding]


class GPT:
    def __init__(
        self,
        context: Optional[Dict[str, str]] = None,
        model: str = DEFAULT_SMART_MODEL,
        gpt_service_stub: Optional[GPTServiceStub] = None,
    ) -> None:
        self.model = model
        self.context = context
        should_create_stub = not gpt_service_stub and not RUNNING_IN_UNIT_TEST
        self.gpt_service_stub = (
            _get_gpt_service_stub(context=context)[0] if should_create_stub else gpt_service_stub
        )

    async def do_chat_w_sys_prompt(
        self,
        main_prompt: FilledPrompt,
        sys_prompt: FilledPrompt,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: Optional[int] = None,
        task_type: Optional[str] = None,
        additional_context: Optional[Dict[str, Any]] = None,
        output_json: bool = False,
        no_cache: bool = False,
    ) -> str:
        context_with_task_type = copy.deepcopy(self.context) if self.context is not None else None
        if context_with_task_type is not None:
            if task_type is not None:
                context_with_task_type[GPT_TASK_TYPE] = task_type
            else:
                context_with_task_type[GPT_TASK_TYPE] = get_gpt_task_type()

        if additional_context is not None:
            if context_with_task_type is not None:
                context_with_task_type = {**context_with_task_type, **additional_context}
            else:
                context_with_task_type = additional_context

        try:
            context_with_task_type = context_with_task_type or {}
            if main_prompt.name:
                context_with_task_type[MAIN_PROMPT_NAME] = main_prompt.name
            if main_prompt.template_args:
                template_keys_str = json.dumps(main_prompt.template_args, default=str)
                context_with_task_type[MAIN_PROMPT_TEMPLATE_ARGS] = template_keys_str
            if main_prompt.template:
                context_with_task_type[MAIN_PROMPT_TEMPLATE_NAME] = main_prompt.template

            if sys_prompt.name:
                context_with_task_type[SYS_PROMPT_NAME] = sys_prompt.name
            if sys_prompt.template_args:
                template_keys_str = json.dumps(sys_prompt.template_args, default=str)
                context_with_task_type[SYS_PROMPT_TEMPLATE_ARGS] = template_keys_str
            if sys_prompt.template:
                context_with_task_type[SYS_PROMPT_TEMPLATE_NAME] = sys_prompt.template
        except Exception:
            logger.exception("Failed to store extra prompt args")

        # Revert to prior behavior, with None instead of empty dict
        context_with_task_type = context_with_task_type if context_with_task_type else None

        return await query_gpt_worker(
            model=self.model,
            main_prompt=main_prompt.filled_prompt,
            sys_prompt=sys_prompt.filled_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            context=context_with_task_type,
            output_json=output_json,
            no_cache=no_cache,
            gpt_service_stub=self.gpt_service_stub,
        )

    async def embed_text(
        self, text: str, no_cache: bool = False, embedding_model: str = DEFAULT_EMBEDDING_MODEL
    ) -> List[float]:
        return await get_embedding(
            model=embedding_model,
            text=text,
            no_cache=no_cache,
            gpt_service_stub=self.gpt_service_stub,
        )
