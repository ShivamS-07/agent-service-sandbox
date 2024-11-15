import contextvars
import copy
import json
import logging
import os
from typing import Any, Dict, List, Optional, Union

from llm_client.datamodels import DoChatArgs, DoEmbedArgs
from llm_client.llm_client import LLMClient
from pydantic import BaseModel
from pydantic.v1.main import ModelMetaclass

from agent_service.GPT.constants import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_SMART_MODEL,
    DEFAULT_TEMPERATURE,
)
from agent_service.types import PlanRunContext
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


def set_plan_run_context(context: PlanRunContext, scheduled_by_automation: bool) -> None:
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


class GPT:
    def __init__(
        self,
        context: Optional[Dict[str, str]] = None,
        model: str = DEFAULT_SMART_MODEL,
    ) -> None:
        self.model = model
        self.context = context
        self.llm_client = LLMClient(max_retries=20)

    async def do_chat_w_sys_prompt_impl(
        self,
        main_prompt: FilledPrompt,
        sys_prompt: FilledPrompt,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: Optional[int] = None,
        task_type: Optional[str] = None,
        additional_context: Optional[Dict[str, Any]] = None,
        output_json: bool = False,
        no_cache: bool = False,
        base_model: Optional[ModelMetaclass] = None,
    ) -> Union[str, BaseModel]:
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
        plan_run_context = PLAN_RUN_CONTEXT.get()
        context_with_task_type = context_with_task_type if context_with_task_type else {}
        context_to_use = {**context_with_task_type, **plan_run_context}
        do_chat_args = DoChatArgs(
            model_id=self.model,
            max_tokens=max_tokens,
            main_prompt=main_prompt.filled_prompt,
            sys_prompt=sys_prompt.filled_prompt,
            temperature=temperature,
            context=context_to_use,
            base_model=base_model,
        )
        return await self.llm_client.do_chat(do_chat_args=do_chat_args, no_cache=no_cache)

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
        result: str = await self.do_chat_w_sys_prompt_impl(
            main_prompt=main_prompt,
            sys_prompt=sys_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            task_type=task_type,
            additional_context=additional_context,
            output_json=output_json,
            no_cache=no_cache,
        )  # type: ignore
        return result

    async def embed_text(
        self, text: str, no_cache: bool = False, embedding_model: str = DEFAULT_EMBEDDING_MODEL
    ) -> List[float]:
        do_embed_args = DoEmbedArgs(model_id=embedding_model, text=text)
        return await self.llm_client.do_embed(do_embed_args=do_embed_args)
