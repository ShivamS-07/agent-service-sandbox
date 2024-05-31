# Author(s): Mohammad Zarei

import datetime
from typing import List, Optional

from agent_service.GPT.constants import DEFAULT_SMART_MODEL
from agent_service.GPT.requests import GPT
from agent_service.io_types.text import (
    Text,
    ThemeNewsDevelopmentArticlesText,
    ThemeNewsDevelopmentText,
    ThemeText,
)
from agent_service.tool import ToolArgs, ToolCategory, tool
from agent_service.tools.news import (
    GetNewsArticlesForTopicsInput,
    get_news_articles_for_topics,
)
from agent_service.tools.themes import (
    GetMacroeconomicThemeInput,
    GetThemeDevelopmentNewsArticlesInput,
    GetThemeDevelopmentNewsInput,
    GetTopNThemesInput,
    get_macroeconomic_themes,
    get_news_articles_for_theme_developments,
    get_news_developments_about_theme,
    get_top_N_themes,
)
from agent_service.tools.tool_log import tool_log
from agent_service.types import PlanRunContext
from agent_service.utils.gpt_logging import GptJobIdType, GptJobType, create_gpt_context
from agent_service.utils.prompt_utils import Prompt

# Constants
MAX_ARTICLES_PER_DEVELOPMENT = 5
MAX_DEVELOPMENTS_PER_TOPIC = 10
MAX_MATCHED_ARTICLES_PER_TOPIC = 50


# PROMPTS
COMMENTARY_SYS_PROMPT = Prompt(
    name="LLM_COMMENTARY_SYS_PROMPT",
    template=(
        "You are a financial analyst tasked with writing a commentary according "
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
        "Analyze the following text(s) and write a commentary based on the needs of the client. "
        "The texts include themes, news developments or news articles related to the client's interests. "
        "Here are, all texts for your analysis, delimited by ----:\n"
        "----\n"
        "{texts}"
        "----\n"
        "Here is the transcript of your interaction with the client, delimited by ----:\n"
        "----\n"
        "{chat_context}"
        "----\n"
        "Now write your commentary."
    ),
)


class WriteCommentaryInput(ToolArgs):
    texts: List[Text]


@tool(
    description=(
        "This function should be used when user wants to write a commentary, articles or market summaries. "
        "This function generates a commentary either based for general market trends or "
        "based on specific topics mentioned by a client. "
        "The function creates a concise summary based on a comprehensive analysis of the provided texts. "
        "The commentary will be written in a professional tone, "
        "incorporating any specific instructions or preferences mentioned by the client during their interaction. "
        "The input to the function is prepared by the get_commentary_input tool. "
    ),
    category=ToolCategory.COMMENTARY,
)
async def write_commentary(args: WriteCommentaryInput, context: PlanRunContext) -> Text:
    # Write the commentary prompt
    gpt_context = create_gpt_context(
        GptJobType.AGENT_TOOLS, context.agent_id, GptJobIdType.AGENT_ID
    )
    llm = GPT(context=gpt_context, model=DEFAULT_SMART_MODEL)

    result = await llm.do_chat_w_sys_prompt(
        main_prompt=COMMENTARY_PROMPT_MAIN.format(
            texts="\n***\n".join(Text.get_all_strs(args.texts)),
            chat_context=context.chat.get_gpt_input() if context.chat is not None else "",
        ),
        sys_prompt=COMMENTARY_SYS_PROMPT.format(),
    )

    return Text(val=result)


class GetCommentaryTextsInput(ToolArgs):
    topics: List[str] = [""]
    start_date: Optional[datetime.date] = None  # type: ignore
    no_specific_topic: bool = False


@tool(
    description=(
        "This function can be used when  a client wants to write a commentary, article or market summary. "
        "This function collects and prepares all text inputs to be used by the write_commentary tool "
        "for writing a commentary or short articles and market summaries. "
        "This function MUST only be used for write commentary tool. "
        "If client wants a general commentary, 'no_specific_topic' MUST be set to True. "
    ),
    category=ToolCategory.COMMENTARY,
)
async def get_commentary_texts(
    args: GetCommentaryTextsInput, context: PlanRunContext
) -> List[Text]:

    # If general_commentary is True, get the texts for top 3 themes
    if args.no_specific_topic or args.topics == [""]:
        themes_texts: List[ThemeText] = await get_top_N_themes(  # type: ignore
            GetTopNThemesInput(start_date=args.start_date, theme_num=3), context
        )
        texts = await _get_theme_related_texts(themes_texts, context)
        await tool_log(
            log="Got texts for top 3 themes for general commentary.",
            context=context,
        )
        return texts

    # Try get the themes for each topics or find related articles
    texts = await _get_texts_for_topics(args, context)

    return texts


# Helper functions


async def _get_texts_for_topics(
    args: GetCommentaryTextsInput, context: PlanRunContext
) -> List[Text]:
    """
    This function gets the texts for the given topics. If the themes are found, it gets the related texts.
    If the themes are not found, it gets the articles related to the topic.
    """
    texts = []
    for topic in args.topics:
        themes = await get_macroeconomic_themes(
            GetMacroeconomicThemeInput(theme_refs=[topic]), context
        )
        if themes[0].val != "-1":  # type: ignore
            # If themes are found, get the related texts
            await tool_log(
                log=f"Getting theme related texts for topic: {topic}",
                context=context,
            )
            res = await _get_theme_related_texts(themes, context)  # type: ignore
            texts.extend(res)
        else:
            # If themes are not found, get the articles related to the topic
            await tool_log(
                log=f"No themes found for topic: {topic}, trying to match articles.",
                context=context,
            )
            matched_articles = await get_news_articles_for_topics(
                GetNewsArticlesForTopicsInput(
                    topics=[topic],
                    start_date=args.start_date,
                ),
                context,
            )
            matched_articles = matched_articles[0][:MAX_MATCHED_ARTICLES_PER_TOPIC]  # type: ignore
            texts.extend(matched_articles)  # type: ignore

    return texts


async def _combine_into_single_texts_list(
    themes: List[ThemeText],
    developments: List[List[ThemeNewsDevelopmentText]],
    articles: List[List[List[ThemeNewsDevelopmentArticlesText]]],
) -> List[Text]:
    """
    This function combines the themes, developments, and articles into a single list of Text objects.
    """
    texts = []
    for i in range(len(themes)):
        texts.append(themes[i])  # type: ignore
        for j in range(len(developments[i])):
            texts.append(developments[i][j])  # type: ignore
            for k in range(len(articles[i][j])):
                texts.append(articles[i][j][k])  # type: ignore
    return texts  # type: ignore


async def _get_theme_related_texts(
    themes_texts: List[ThemeText], context: PlanRunContext
) -> List[Text]:
    """
    This function gets the theme related texts for the given themes.
    """
    development_texts = await get_news_developments_about_theme(
        GetThemeDevelopmentNewsInput(themes=themes_texts), context
    )
    article_texts = await get_news_articles_for_theme_developments(  # type: ignore
        GetThemeDevelopmentNewsArticlesInput(developments_list=development_texts), context  # type: ignore
    )
    # limit number of developments and articles
    development_texts_limited = []
    article_texts_limited = []

    for i in range(len(themes_texts)):
        # Get the first MAX_DEVELOPMENTS_PER_TOPIC developments for the current theme
        limited_developments = development_texts[i][:MAX_DEVELOPMENTS_PER_TOPIC]  # type: ignore
        development_texts_limited.append(limited_developments)

        # For each development, get the first MAX_ARTICLES_PER_DEVELOPMENT articles
        limited_articles = [
            article_texts[i][j][:MAX_ARTICLES_PER_DEVELOPMENT]  # type: ignore
            for j in range(len(limited_developments))
        ]
        article_texts_limited.append(limited_articles)
    return await _combine_into_single_texts_list(
        themes_texts, development_texts_limited, article_texts_limited  # type: ignore
    )
