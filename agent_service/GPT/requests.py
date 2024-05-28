import asyncio
import copy
import json
import logging
import os
import traceback
import uuid
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

from gbi_common_py_utils.utils.environment import (
    DEV_TAG,
    LOCAL_TAG,
    get_environment_tag,
)
from gbi_common_py_utils.utils.event_logging import log_event
from google.protobuf.struct_pb2 import Struct
from gpt_service_proto_v1.service_grpc import GPTServiceStub
from gpt_service_proto_v1.service_pb2 import QueryGPTRequest, QueryGPTResponse
from grpclib.client import Channel

from agent_service.GPT.constants import (
    CLIENT_NAME,
    CLIENT_NAMESPACE,
    DEFAULT_SMART_MODEL,
    DEFAULT_TEMPERATURE,
    JSON_RESPONSE_FORMAT,
    MAX_GPT_WORKER_TIMEOUT,
    TEXT_RESPONSE_FORMAT,
)
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

# unit tests that inherit from IsolatedAsyncioTestCase broken by this stub cache because
# IsolatedAsyncioTestCase creates a new event loop for each test case and thus we need a new
# connection for each test case
USE_GLOBAL_STUB = False


def use_global_stub() -> bool:
    return USE_GLOBAL_STUB


def set_use_global_stub(val: bool) -> None:
    global USE_GLOBAL_STUB
    USE_GLOBAL_STUB = val


def _get_gpt_service_stub() -> Tuple[GPTServiceStub, Channel]:
    global STUB
    if STUB and use_global_stub():
        return STUB
    url = os.getenv("GPT_SERVICE_URL", "gpt-service-2.boosted.ai:50051")
    host, port = url.split(":")
    channel = Channel(host=host, port=int(port))
    STUB = GPTServiceStub(channel)
    return STUB, channel


async def query_gpt_worker(
    model: str,
    main_prompt: str,
    sys_prompt: str,
    temperature: float,
    context: Optional[Dict[str, str]] = None,
    max_tokens: Optional[int] = None,
    output_json: bool = False,
    retry_num: int = 1,
    max_retries: int = 3,
    request_id: Optional[str] = None,
    no_cache: bool = False,
    gpt_service_stub: Optional[GPTServiceStub] = None,
) -> str:
    if not request_id:
        request_id = str(uuid.uuid4())
    client_timestamp = datetime.utcnow().isoformat()
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
                "give_up_timestamp": datetime.utcnow().isoformat(),
                "model_id": model,
            },
        )
        if retry_num == max_retries:
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

    if not client_timestamp:
        client_timestamp = datetime.utcnow().isoformat()
    priority = DEFAULT_PRIORITY
    context_struct = Struct()
    if context is not None:
        context_struct.update(context)
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
    result: QueryGPTResponse = await stub.QueryGPT(
        request, timeout=MAX_GPT_WORKER_TIMEOUT, metadata=metadata
    )

    if not use_global_stub():
        # explicitly close the channel after each call during unittests
        channel.close()

    if result.status.code != 0:
        raise RuntimeError(f"Error response from GPT service: {result.status.message}")

    return result.response


class GPT:
    def __init__(
        self,
        context: Optional[Dict[str, str]] = None,
        model: str = DEFAULT_SMART_MODEL,
        gpt_service_stub: Optional[GPTServiceStub] = None,
    ) -> None:
        self.model = model
        self.context = context
        self.gpt_service_stub = gpt_service_stub

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
