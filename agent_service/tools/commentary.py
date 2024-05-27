# Author(s): Mohammad Zarei

from typing import List

from agent_service.GPT.constants import DEFAULT_SMART_MODEL
from agent_service.GPT.requests import GPT
from agent_service.io_types.text import (
    Text,
    ThemeNewsDevelopmentArticlesText,
    ThemeNewsDevelopmentText,
    ThemeText,
)
from agent_service.tool import ToolArgs, ToolCategory, tool
from agent_service.types import PlanRunContext
from agent_service.utils.gpt_logging import GptJobIdType, GptJobType, create_gpt_context
from agent_service.utils.prompt_utils import FilledPrompt, Prompt

# Constants
MAX_ARTICLES_PER_DEVELOPMENT = 10
MAX_DEVELOPMENTS_PER_THEME = 20


# PROMPTS
COMMENTARY_SYS_PROMPT = Prompt(
    name="LLM_COMMENTARY_SYS_PROMPT",
    template=(
        "You are a financial analyst tasked with writing a macroeconomic commentary according "
        "to the instructions of an important client. You will be provided with the texts "
        "as well as transcript of your conversation with the client. If the client has "
        "provided you with any specifics about the format or content of the summary, you "
        "must follow those instructions. If a specific topic is mentioned, you must only "
        "include information about that topic. Otherwise, you should write a normal prose "
        "summary that touches on what you be to be the most important points that you see "
        "across all the text you have been provided with. The most important points are those "
        "which are highlighted, repeated, or otherwise appear most relevant to the user's "
        "expressed interest, if any. If none of these criteria seem to apply, use your best "
        "judgment on what seems to be important. Unless the user says otherwise, your output "
        "should be must smaller (a small fraction) of all the text provided. Please be concise "
        "in your writing, and do not include any fluff. NEVER include ANY information about your "
        "portfolio that is not explicitly provided to you. NEVER PUT SOURCES INLINE. For example, if "
        "the input involves several news summaries, a single sentence or two would be "
        "appropriate. Individual texts in your collection are delimited by ***. "
        "You can use markdown formatting in your final output to highlight some points."
    ),
)

COMMENTARY_PROMPT_MAIN = Prompt(
    name="LLM_COMMENTARY_MAIN_PROMPT",
    template=(
        "Summarize the following text(s) based on the needs of the client. "
        "Here are the themes descriptions along with their related developments and articles, delimited by ----:\n"
        "----\n"
        "{sections}\n"
        "----\n"
        "Here is the transcript of your interaction with the client, delimited by ----:\n"
        "----\n"
        "{chat_context}\n"
        "----\n"
        "Now write your commentary."
    ),
)


class WriteCommentaryInput(ToolArgs):
    themes: List[ThemeText]
    developments: List[List[ThemeNewsDevelopmentText]]
    articles: List[List[List[ThemeNewsDevelopmentArticlesText]]]


@tool(
    description=(
        "This function generates a macroeconomic commentary tailored to the specific needs of a client. "
        "the function creates a concise summary highlighting the most important points from the provided texts. "
        "The summary will be based on a comprehensive analysis of themes, developments, and related articles, "
        "incorporating any specific instructions or preferences mentioned by the client during their interaction. "
    ),
    category=ToolCategory.COMMENTARY,
)
async def write_commentary(args: WriteCommentaryInput, context: PlanRunContext) -> Text:
    # limit number of developments and articles
    developments = []
    articles = []

    for i, theme in enumerate(args.themes):
        # Get the first MAX_DEVELOPMENTS_PER_THEME developments for the current theme
        limited_developments = args.developments[i][:MAX_DEVELOPMENTS_PER_THEME]
        developments.append(limited_developments)

        # For each development, get the first MAX_ARTICLES_PER_DEVELOPMENT articles
        limited_articles = [
            args.articles[i][j][:MAX_ARTICLES_PER_DEVELOPMENT]
            for j in range(len(limited_developments))
        ]
        articles.append(limited_articles)

    # TODO we need guardrails on this
    gpt_context = create_gpt_context(
        GptJobType.AGENT_TOOLS, context.agent_id, GptJobIdType.AGENT_ID
    )
    llm = GPT(context=gpt_context, model=DEFAULT_SMART_MODEL)

    prompt_input = _prepare_commentary_prompt(
        themes=args.themes,
        developments=developments,
        articles=articles,
        chat_context=context.chat.get_gpt_input() if context.chat is not None else "",
    )

    result = await llm.do_chat_w_sys_prompt(
        main_prompt=prompt_input,
        sys_prompt=COMMENTARY_SYS_PROMPT.format(),
    )

    return Text(val=result)


def _prepare_commentary_prompt(
    themes: List[ThemeText],
    developments: List[List[ThemeNewsDevelopmentText]],
    articles: List[List[List[ThemeNewsDevelopmentArticlesText]]],
    chat_context: str,
) -> FilledPrompt:
    sections = []
    for i, theme in enumerate(themes):
        theme_str = Text.get_all_strs(theme)
        developments_str = "\n***\n".join(Text.get_all_strs(developments[i]))
        articles_str = "\n***\n".join(
            "\n".join(Text.get_all_strs(article)) for article in articles[i]
        )
        section = (
            f"Theme:\n-----\n{theme_str}\n-----\n"
            f"Developments:\n-----\n{developments_str}\n-----\n"
            f"Articles:\n-----\n{articles_str}\n-----"
        )
        sections.append(section)
    return COMMENTARY_PROMPT_MAIN.format(
        sections="\n\n".join(sections),
        chat_context=chat_context,
    )
