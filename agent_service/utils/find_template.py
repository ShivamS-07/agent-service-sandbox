import logging
from typing import List, Union

from agent_service.endpoints.models import PromptTemplate
from agent_service.GPT.constants import GPT4_O_MINI
from agent_service.GPT.requests import GPT
from agent_service.utils.async_utils import gather_with_concurrency
from agent_service.utils.logs import async_perf_logger
from agent_service.utils.prompt_utils import Prompt

LOGGER = logging.getLogger(__name__)

FIND_TEMPLATE_SYS_PROMPT = Prompt(
    name="FIND_TEMPLATE_SYS_PROMPT",
    template=(
        "Your goal is evaluate if the following template is a good match "
        "or highly related to a given client query. "
        "Each template is a plan for doing a task. "
        "You will be given the template description and template prompt. "
        "Here are some guidelines to help you evaluate the template: "
        "\n- Your response must be only one word: 'yes' or 'no'. "
    ),
)


FIND_TEMPLATE_MAIN_PROMPT = Prompt(
    name="FIND_TEMPLATE_MAIN_PROMPT",
    template=(
        "Given the client query: '{query}', evaluate if the following template is a good match "
        "or related to the client query. "
        "\n\nThe template description is: '{description}'. "
        "\n\nThe template prompt is: '{prompt}'. "
        "Now write 'yes' or 'no' to indicate if the template is a good match or related to the client query."
    ),
)


@async_perf_logger
async def get_matched_templates(
    query: str,
    prompt_templates: List[PromptTemplate],
    model: str = GPT4_O_MINI,
) -> Union[List[PromptTemplate], None]:
    # get all templates
    sys_prompt = FIND_TEMPLATE_SYS_PROMPT.format()

    llm = GPT(model=model)
    tasks = [
        llm.do_chat_w_sys_prompt(
            FIND_TEMPLATE_MAIN_PROMPT.format(
                query=query, description=template.description, prompt=template.prompt
            ),
            sys_prompt,
        )
        for template in prompt_templates
    ]
    llm_responses = await gather_with_concurrency(tasks)
    matched_templates = []
    for res, template in zip(llm_responses, prompt_templates):
        if "yes" in res.lower():
            LOGGER.info(f"Template matched: {template.name}")
            matched_templates.append(template)
    return matched_templates if matched_templates else None
