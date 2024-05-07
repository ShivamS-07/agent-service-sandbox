import asyncio
import logging
import os
import threading
import time
from typing import Dict, Optional

import aiohttp
import backoff
import openai
from gbi_common_py_utils.utils.ssm import get_param
from openai import util

from agent_service.GPT.constants import (
    DEFAULT_TEMPERATURE,
    JSON_RESPONSE_FORMAT,
    OPENAI_API_PARAM,
    OPENAI_ORG_PARAM,
    TEXT_RESPONSE_FORMAT,
    TIMEOUTS,
)
from agent_service.utils.gpt_input_output_logger import GPTInputOutputLogger

logger = logging.getLogger(__name__)

openai.api_key = get_param(OPENAI_API_PARAM)

OPENAI_RETRIABLE_EXCEPTION_TYPES = (
    openai.error.RateLimitError,
    openai.error.Timeout,
    openai.error.APIError,
    openai.error.TryAgain,
    openai.error.ServiceUnavailableError,
    openai.error.APIConnectionError,
    asyncio.TimeoutError,
    aiohttp.http.HttpProcessingError,
    aiohttp.client.ClientError,
)

OPENAI_RATE_LIMIT_EXCEPTION_TYPES = (openai.error.RateLimitError,)

SERVICE_VERSION = os.getenv("SERVICE_VERSION", "")

util.logger.setLevel(logging.WARNING)  # filter out the annoying INFO logs


class GPTQueryClient:
    logging_lock = threading.Lock()

    def __init__(self, log_request_response: bool = True) -> None:
        openai.organization = get_param(OPENAI_ORG_PARAM)
        openai.api_key = get_param(OPENAI_API_PARAM)
        self.log_request_response = log_request_response
        if os.getenv("NO_GPT_LOG") == "1":
            self.log_request_response = False
        self.publish_metrics = log_request_response

    @backoff.on_exception(
        backoff.constant,
        OPENAI_RETRIABLE_EXCEPTION_TYPES,
        interval=1,
        jitter=None,  # type: ignore
        max_time=6000,  # 100 minutes in total
    )
    async def send_prompt_to_gpt_with_retries(
        self,
        main_prompt: str,
        sys_prompt: str,
        model: str,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: Optional[int] = None,
        context: Optional[Dict[str, str]] = None,
        output_json: bool = False,
    ):
        return await self.send_prompt_to_gpt(
            main_prompt=main_prompt,
            sys_prompt=sys_prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            context=context,
            output_json=output_json,
        )

    async def send_prompt_to_gpt(
        self,
        main_prompt: str,
        sys_prompt: str,
        model: str,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: Optional[int] = None,
        context: Optional[Dict[str, str]] = None,
        output_json: bool = False,
    ) -> str:
        """
        Runs GPT chat completion with both a main prompt as well as a system prompt

        main_prompt (str): the main (user) prompt, specific instructions and input
        sys_prompt (str): the system prompt, general instructions
        temperature (str): the temperature to be used, controls randomness of GPT output

        Returns the GPT response (str)
        """
        # Note that the backoff interval won't really exponentially increase, but more like a random
        # number within the `max_value` range. But as long as it meets the max_value and max_time
        # constraints it should be fine
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": main_prompt},
        ]
        timeout = TIMEOUTS[model]
        start_counter = time.perf_counter()
        if output_json:
            response_format = JSON_RESPONSE_FORMAT
        else:
            response_format = TEXT_RESPONSE_FORMAT

        if max_tokens is None:
            coroutine = openai.ChatCompletion.acreate(
                model=model,
                messages=messages,
                temperature=temperature,
                response_format=response_format,
            )
            response = await asyncio.wait_for(coroutine, timeout=timeout)
        else:  # need to do this because no way to get the right max_token default
            coroutine = openai.ChatCompletion.acreate(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=response_format,
            )
            response = await asyncio.wait_for(coroutine, timeout=timeout)
        latency_seconds = time.perf_counter() - start_counter
        num_input_tokens = response["usage"]["prompt_tokens"]
        num_output_tokens = response["usage"]["completion_tokens"]
        result: str = response["choices"][0]["message"]["content"]
        context = context or {}
        context["service_name"] = "Agent"
        context["service_version"] = SERVICE_VERSION

        try:
            # contains token info
            extra_args = dict(response["usage"])
            extra_args["model"] = model
            extra_args["temperature"] = str(temperature)
            if max_tokens:
                extra_args["max_tokens"] = str(max_tokens)
        except Exception:
            logger.exception("Error publishing GPT query metrics")

        if self.log_request_response:
            GPTInputOutputLogger.log_request_response(
                model,
                sys_prompt,
                main_prompt,
                result,
                latency_seconds,
                num_input_tokens,
                num_output_tokens,
                context,
            )
        return result
