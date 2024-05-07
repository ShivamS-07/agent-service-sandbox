import asyncio
import copy
import hashlib
import json
import logging
import os
import traceback
import uuid
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, Generator, List, Optional, Tuple, cast

import aiohttp
import backoff
import openai
from gbi_common_py_utils.utils.environment import (
    DEV_TAG,
    LOCAL_TAG,
    get_environment_tag,
)
from gbi_common_py_utils.utils.event_logging import log_event
from gbi_common_py_utils.utils.ssm import get_param
from google.protobuf.struct_pb2 import Struct
from gpt_service_proto_v1.service_grpc import GPTServiceStub
from gpt_service_proto_v1.service_pb2 import QueryGPTRequest, QueryGPTResponse
from grpclib import GRPCError
from grpclib.client import Channel
from grpclib.exceptions import StreamTerminatedError

from agent_service.GPT.constants import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_SMART_MODEL,
    DEFAULT_TEMPERATURE,
    JSON_RESPONSE_FORMAT,
    MAX_GPT_WORKER_TIMEOUT,
    OPENAI_API_PARAM,
    OPENAI_ORG_PARAM,
    TEXT_RESPONSE_FORMAT,
)
from agent_service.GPT.utils import GPTQueryClient
from agent_service.utils.environment import EnvironmentUtils
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


def get_client_name() -> str:
    try:
        with open("/etc/hostname", "r") as f:
            return f.read().strip()
    except Exception:
        return "LOCAL"


def get_client_namespace() -> str:
    try:
        with open("/var/run/secrets/kubernetes.io/serviceaccount/namespace", "r") as f:
            return f.read().strip()
    except Exception:
        return "LOCAL"


CLIENT_NAME = get_client_name()
CLIENT_NAMESPACE = get_client_namespace()
DEFAULT_PRIORITY = os.getenv("GPT_SERVICE_DEFAULT_PRIORITY", "LOW")


@contextmanager
def _get_llm_service_stub() -> Generator[GPTServiceStub, None, None]:
    try:
        url = os.getenv("LLM_SERVICE_URL", "localhost:50051")
        host, port = url.split(":")
        channel = Channel(host=host, port=int(port))
        yield GPTServiceStub(channel)
    finally:
        channel.close()


@backoff.on_exception(
    backoff.constant, (GRPCError, StreamTerminatedError, RuntimeError), interval=3, max_tries=3
)
async def query_llm_service(
    model: str,
    main_prompt: str,
    sys_prompt: str,
    temperature: float,
    context: Optional[Dict[str, str]] = None,
    max_tokens: Optional[int] = None,
    output_json: bool = False,
) -> str:
    priority = os.getenv("GPT_SERVICE_DEFAULT_PRIORITY", "LOW")
    context_struct = Struct()
    if context is not None:
        context_struct.update(context)
    extra_params = Struct()
    extra_params.update({"context": context_struct})
    request = QueryGPTRequest(
        model=model,
        main_prompt=main_prompt,
        sys_prompt=sys_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        request_priority=f"GPT_SVC_PRIORITY_{priority}",
        extra_params=extra_params,
        # output_json= output_json
    )
    with _get_llm_service_stub() as stub:
        result: QueryGPTResponse = await stub.QueryGPT(request, timeout=MAX_GPT_WORKER_TIMEOUT)
    if result.status.code != 0:
        raise RuntimeError(f"Error response from LLM service: {result.status.message}")
    return result.response


@contextmanager
def _get_gpt_service_stub() -> Generator[GPTServiceStub, None, None]:
    try:
        url = os.getenv("GPT_SERVICE_URL", "gpt-service.boosted.ai:50051")
        host, port = url.split(":")
        channel = Channel(host=host, port=int(port))
        yield GPTServiceStub(channel)
    finally:
        channel.close()


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
    with _get_gpt_service_stub() as stub:
        result: QueryGPTResponse = await stub.QueryGPT(
            request,
            timeout=MAX_GPT_WORKER_TIMEOUT,
            metadata=[
                ("clienttimestamp", client_timestamp),
                ("clientname", CLIENT_NAME),
                ("clientnamespace", CLIENT_NAMESPACE),
                ("clientrequestid", request_id),
            ],
        )
    if result.status.code != 0:
        raise RuntimeError(f"Error response from GPT service: {result.status.message}")
    return result.response


class GPT:
    def __init__(
        self,
        context: Optional[Dict[str, str]] = None,
        model: str = DEFAULT_SMART_MODEL,
        log_request_response: bool = True,
        use_gpt_worker: bool = True,
    ) -> None:
        openai.organization = get_param(OPENAI_ORG_PARAM)
        openai.api_key = get_param(OPENAI_API_PARAM)
        self.model = model
        self.gpt_query_client = GPTQueryClient(log_request_response=log_request_response)
        self.use_gpt_worker = use_gpt_worker
        self.context = context
        self.cache_data: Optional[Dict[str, Any]] = None

    def check_cache(self, main_prompt: str, sys_prompt: str) -> Optional[str]:
        if self.cache_data is None:
            return None
        cache_key = hashlib.md5(f"{sys_prompt}{main_prompt}".encode()).hexdigest()
        get = cast(str, self.cache_data.get(cache_key, None))
        return get

    async def do_chat_w_sys_prompt(
        self,
        main_prompt: FilledPrompt,
        sys_prompt: FilledPrompt,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: Optional[int] = None,
        task_type: Optional[str] = None,
        additional_context: Optional[Dict[str, Any]] = None,
        output_json: bool = False,
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

        cache_result = self.check_cache(
            main_prompt=main_prompt.filled_prompt,
            sys_prompt=sys_prompt.filled_prompt,
        )
        if cache_result:
            return cache_result
        prompt_id = (
            f'{(context_with_task_type or {}).get("job_type", "")}'
            f'{(context_with_task_type or {}).get("task_type", "")}'
        )
        override_model = EnvironmentUtils.llm_config.get(prompt_id)
        self.model = override_model or self.model
        if override_model:
            result = await query_llm_service(
                model=self.model,
                main_prompt=main_prompt.filled_prompt,
                sys_prompt=sys_prompt.filled_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                context=context_with_task_type,
                # output_json = output_json
            )
        elif self.use_gpt_worker:
            result = await query_gpt_worker(
                model=self.model,
                main_prompt=main_prompt.filled_prompt,
                sys_prompt=sys_prompt.filled_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                context=context_with_task_type,
                output_json=output_json,
            )
        else:
            result = await self.gpt_query_client.send_prompt_to_gpt_with_retries(
                main_prompt=main_prompt.filled_prompt,
                sys_prompt=sys_prompt.filled_prompt,
                model=self.model,
                temperature=temperature,
                max_tokens=max_tokens,
                context=context_with_task_type,
                output_json=output_json,
            )
        if self.cache_data is not None:
            cache_key = hashlib.md5(f"{sys_prompt}{main_prompt}".encode()).hexdigest()
            self.cache_data[cache_key] = result
        return result

    @backoff.on_exception(
        backoff.expo,
        (
            openai.error.RateLimitError,
            openai.error.Timeout,
            openai.error.APIError,
            openai.error.TryAgain,
            openai.error.APIConnectionError,
            openai.error.ServiceUnavailableError,
            aiohttp.ClientPayloadError,
            aiohttp.ClientOSError,
            asyncio.TimeoutError,
        ),
        max_time=600,  # 10 minutes in total
    )
    async def get_embedding(
        self, text: str, embedding_model: str = DEFAULT_EMBEDDING_MODEL
    ) -> List[float]:
        """
        Creates a GPT-based embedding (1536D) for the provided text
        """

        if self.cache_data is not None:
            cache_key = hashlib.md5(text.encode()).hexdigest()
            get = cast(List[float], self.cache_data.get(cache_key, None))
            if get is not None:
                return get
        coroutine = openai.Embedding.acreate(
            input=text,
            model=embedding_model,
        )
        # Should be very quick, timeout set to low
        response = await asyncio.wait_for(coroutine, timeout=90)
        embedding: List[float] = response["data"][0]["embedding"]
        if self.cache_data is not None:
            cache_key = hashlib.md5(text.encode()).hexdigest()
            self.cache_data[cache_key] = embedding
        return embedding

    @backoff.on_exception(
        backoff.expo,
        (
            openai.error.RateLimitError,
            openai.error.Timeout,
            openai.error.APIError,
            openai.error.TryAgain,
            openai.error.APIConnectionError,
            openai.error.ServiceUnavailableError,
            aiohttp.ClientPayloadError,
            aiohttp.ClientOSError,
            asyncio.TimeoutError,
        ),
        max_time=600,  # 10 minutes in total
    )
    async def get_embedding_with_metadata(
        self, text: str, embedding_model: str = DEFAULT_EMBEDDING_MODEL
    ) -> Tuple[List[float], str, int, int]:
        """
        Creates a GPT-based embedding (1536D) for the provided text
        """

        if self.cache_data is not None:
            cache_key = hashlib.md5(text.encode()).hexdigest()
            get = cast(Tuple[List[float], str, int, int], self.cache_data.get(cache_key, None))
            if get is not None:
                return get
        coroutine = openai.Embedding.acreate(
            input=text,
            model=embedding_model,
        )
        # Should be very quick, timeout set to low
        response = await asyncio.wait_for(coroutine, timeout=90)
        embedding = response["data"][0]["embedding"]
        num_tokens = response["usage"]["total_tokens"]
        latency_ms = response.response_ms
        if self.cache_data is not None:
            cache_key = hashlib.md5(text.encode()).hexdigest()
            self.cache_data[cache_key] = embedding
        return embedding, embedding_model, num_tokens, latency_ms
