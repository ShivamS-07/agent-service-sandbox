from typing import List, Optional

from gpt_service_proto_v1.service_grpc import GPTServiceStub

from agent_service.chatbot.prompts import AGENT_DESCRIPTION
from agent_service.GPT.constants import DEFAULT_SMART_MODEL
from agent_service.GPT.requests import GPT
from agent_service.types import ChatContext
from agent_service.utils.gpt_logging import chatbot_context
from agent_service.utils.logs import async_perf_logger
from agent_service.utils.prompt_utils import Prompt

NAME_AGENT_SYS_PROMPT_STR = "{agent_description}"

NAME_AGENT_MAIN_PROMPT_STR = """
    Given the following query from a client, generate a short name that is appropriate for this chat.
    Here is the client request: {chat_context}.
    Try to use names that user can easily distinguish from the following: {existing_names}.
    Limit your response to only the name. Be formal, very specific and short.
    The words don't need to be glued together. Don't use quotes. Use less than 4 words.
"""

NAME_AGENT_SYS_PROMPT = Prompt(NAME_AGENT_SYS_PROMPT_STR, "NAME_AGENT_SYS_PROMPT")
NAME_AGENT_MAIN_PROMPT = Prompt(NAME_AGENT_MAIN_PROMPT_STR, "NAME_AGENT_MAIN_PROMPT")


@async_perf_logger
async def generate_name_for_agent(
    agent_id: str,
    chat_context: ChatContext,
    existing_names: List[str],
    model: str = DEFAULT_SMART_MODEL,
    gpt_service_stub: Optional[GPTServiceStub] = None,
    user_id: Optional[str] = None,
) -> str:
    main_prompt = NAME_AGENT_MAIN_PROMPT.format(
        chat_context=chat_context.get_gpt_input(), existing_names=existing_names
    )
    sys_prompt = NAME_AGENT_SYS_PROMPT.format(agent_description=AGENT_DESCRIPTION)
    context = chatbot_context(agent_id, user_id)
    llm = GPT(context=context, model=model, gpt_service_stub=gpt_service_stub)
    result = await llm.do_chat_w_sys_prompt(main_prompt, sys_prompt, max_tokens=10)
    return result
