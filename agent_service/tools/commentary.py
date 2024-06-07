# Author(s): Mohammad Zarei

import asyncio
import datetime
from typing import List, Optional
from uuid import uuid4

from agent_service.GPT.constants import DEFAULT_SMART_MODEL
from agent_service.GPT.requests import GPT
from agent_service.io_types.text import Text, ThemeText
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
    get_top_N_macroeconomic_themes,
)
from agent_service.tools.tool_log import tool_log
from agent_service.types import ChatContext, Message, PlanRunContext
from agent_service.utils.date_utils import get_now_utc
from agent_service.utils.gpt_logging import GptJobIdType, GptJobType, create_gpt_context
from agent_service.utils.postgres import get_psql
from agent_service.utils.prefect import get_prefect_logger
from agent_service.utils.prompt_utils import Prompt

# Constants
MAX_ARTICLES_PER_DEVELOPMENT = 5
MAX_DEVELOPMENTS_PER_TOPIC = 10
MAX_MATCHED_ARTICLES_PER_TOPIC = 25


# PROMPTS
COMMENTARY_SYS_PROMPT = Prompt(
    name="COMMENTARY_SYS_PROMPT",
    template=(
        "You are a financial analyst tasked with writing a commentary according "
        "to the instructions of an important client. You will be provided with the texts "
        "as well as transcript of your conversation with the client. If the client has "
        "provided you with any specifics about the format or content of the summary, you "
        "must follow those instructions. If a specific topic is mentioned, you must only "
        "include information about that topic. Otherwise, you should write a normal prose "
        "summary that touches on what you be to be the most important points that you see "
        "across all the text you have been provided with. The most important points are those "
        "which are highlighted, repeated, or otherwise appear most relevant to the client's "
        "expressed interest, if any. If none of these criteria seem to apply, use your best "
        "judgment on what seems to be important. Unless the client says otherwise, your output "
        "should be must smaller (a small fraction) of all the text provided. Please be concise "
        "in your writing, and do not include any fluff. NEVER include ANY information about your "
        "portfolio that is not explicitly provided to you. NEVER PUT SOURCES INLINE. For example, if "
        "the input involves several news summaries, a single sentence or two would be "
        "appropriate. Individual texts in your collection are delimited by ***. "
        "You can use markdown formatting in your final output to highlight some points."
    ),
)

COMMENTARY_PROMPT_MAIN = Prompt(
    name="COMMENTARY_MAIN_PROMPT",
    template=(
        "Analyze the following text(s) and write a commentary based on the needs of the client. "
        "{previous_commentary}"
        "The texts include themes, news developments or news articles related to the client's interests. "
        "Here are, all texts for your analysis, delimited by ----:\n"
        "----\n"
        "{texts}"
        "----\n"
        "Here is the transcript of your interaction with the client, delimited by ----:\n"
        "----\n"
        "{chat_context}"
        "----\n"
        "Now write your commentary. "
    ),
)
PREVIOUS_COMMENTARY_MODIFICATION = Prompt(
    name="PREVIOUS_COMMENTARY_MODIFICATION",
    template=(
        "Here is the previous commentary you wrote for the client, delimited by ***. "
        "\n***\n"
        "{commentary}"
        "\n***\n"
        "Only act based on one of the following cases:\n"
        "- If client only asked for minor changes (adding more details, changing the tone, "
        "making it shorter, etc.) you MUST use the previous commentary as a base and "
        "make those changes. Ignore the following given texts.\n"
        "- If client asked for adding new topics or information to previous commentary, "
        "then you MUST analyze the following texts and adjust the previous commentary accordingly.\n"
        "- If client asked for a completely new commentary, you MUST ignore the previous commentary "
        "and write a new one based on the following texts."
    ),
)


class WriteCommentaryInput(ToolArgs):
    texts: List[Text]


@tool(
    description=(
        "This function should be used when client wants to write a commentary, articles or market summaries. "
        "This function generates a commentary either based for general market trends or "
        "based on specific topics mentioned by a client. "
        "The function creates a concise summary based on a comprehensive analysis of the provided texts. "
        "The commentary will be written in a professional tone, "
        "incorporating any specific instructions or preferences mentioned by the client during their interaction. "
        "The input to the function is prepared by the get_commentary_input tool."
    ),
    category=ToolCategory.COMMENTARY,
)
async def write_commentary(args: WriteCommentaryInput, context: PlanRunContext) -> Text:
    # Write the commentary prompt
    gpt_context = create_gpt_context(
        GptJobType.AGENT_TOOLS, context.agent_id, GptJobIdType.AGENT_ID
    )
    llm = GPT(context=gpt_context, model=DEFAULT_SMART_MODEL)

    # get previous commentary if exists
    db = get_psql()
    previous_commentary = (
        db.get_last_tool_output_for_plan(context.plan_id, context.plan_run_id, context.task_id)
        if context.task_id is not None
        else None
    )
    # If previous commnetary exists, add it to the prompt
    if previous_commentary:
        result = await llm.do_chat_w_sys_prompt(
            main_prompt=COMMENTARY_PROMPT_MAIN.format(
                previous_commentary=PREVIOUS_COMMENTARY_MODIFICATION.format(
                    previous_commentary=previous_commentary
                ),
                texts="\n***\n".join(Text.get_all_strs(args.texts)),
                chat_context=context.chat.get_gpt_input() if context.chat is not None else "",
            ),
            sys_prompt=COMMENTARY_SYS_PROMPT.format(),
        )
    else:
        result = await llm.do_chat_w_sys_prompt(
            main_prompt=COMMENTARY_PROMPT_MAIN.format(
                previous_commentary="",
                texts="\n***\n".join(Text.get_all_strs(args.texts)),
                chat_context=context.chat.get_gpt_input() if context.chat is not None else "",
            ),
            sys_prompt=COMMENTARY_SYS_PROMPT.format(),
        )

    return Text(val=result)


class GetCommentaryTextsInput(ToolArgs):
    topics: List[str] = []
    start_date: datetime.date = None  # type: ignore
    no_specific_topic: bool = False
    portfolio_id: Optional[str] = None


@tool(
    description=(
        "This function can be used when a client wants to write a commentary, article or market summary. "
        "This function collects and prepares all texts to be used by the write_commentary tool "
        "for writing a commentary or short articles and market summaries. "
        "This function MUST only be used for write commentary tool. "
        "If client wants a general commentary, 'no_specific_topic' MUST be set to True. "
        "Adjust start_date to get the text from that date based on client request. "
        "If no start_date is provided, the function will only get text in last month. "
    ),
    category=ToolCategory.COMMENTARY,
)
async def get_commentary_texts(
    args: GetCommentaryTextsInput, context: PlanRunContext
) -> List[Text]:
    if not args.topics:
        # no topics were given so we need to find some default ones
        if args.portfolio_id:
            # get portfolio related topics (logic William is adding)
            themes_texts: List[ThemeText] = await get_top_N_macroeconomic_themes(  # type: ignore
                GetTopNThemesInput(
                    start_date=args.start_date, theme_num=3, portfolio_id=args.portfolio_id
                ),
                context,
            )
            texts = await _get_theme_related_texts(themes_texts, context)
            await tool_log(
                log="Retrieved texts for top 3 themes related to portfolio.",
                context=context,
            )
            return texts
        else:
            # get popular topics
            themes_texts: List[ThemeText] = await get_top_N_macroeconomic_themes(  # type: ignore
                GetTopNThemesInput(start_date=args.start_date, theme_num=3), context
            )
            texts = await _get_theme_related_texts(themes_texts, context)
            await tool_log(
                log="Retrieved texts for top 3 themes for general commentary.",
                context=context,
            )
            return texts

    else:
        if args.portfolio_id:
            themes_texts: List[ThemeText] = await get_top_N_macroeconomic_themes(  # type: ignore
                GetTopNThemesInput(
                    start_date=args.start_date, theme_num=3, portfolio_id=args.portfolio_id
                ),
                context,
            )
            portfolio_texts = await _get_theme_related_texts(themes_texts, context)
            await tool_log(
                log="Retrieved texts for top 3 themes for general commentary.",
                context=context,
            )
            topic_texts = await _get_texts_for_topics(args, context)
            return portfolio_texts + topic_texts
        else:
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
    logger = get_prefect_logger(__name__)
    texts: List = []
    for topic in args.topics:
        try:
            themes = await get_macroeconomic_themes(
                GetMacroeconomicThemeInput(theme_refs=[topic]), context
            )
            await tool_log(
                log=f"Retrieving theme texts for topic: {topic}",
                context=context,
            )
            res = await _get_theme_related_texts(themes, context)  # type: ignore
            texts.extend(res)  # type: ignore

        except Exception as e:
            logger.warning(f"failed to get theme data for topic {topic}: {e}")
            # If themes are not found, get the articles related to the topic
            await tool_log(
                log=f"No themes found for topic: {topic}. Retrieving articles...",
                context=context,
            )
            try:
                matched_articles = await get_news_articles_for_topics(
                    GetNewsArticlesForTopicsInput(
                        topics=[topic],
                        start_date=args.start_date,
                        max_num_articles_per_topic=MAX_MATCHED_ARTICLES_PER_TOPIC,
                    ),
                    context,
                )
                texts.extend(matched_articles)  # type: ignore
            except Exception as e:
                logger.warning(f"failed to get news pool articles for topic {topic}: {e}")

    if len(texts) == 0:
        raise Exception("No data collected for commentary from available sources")

    return texts


async def _get_theme_related_texts(
    themes_texts: List[ThemeText], context: PlanRunContext
) -> List[Text]:
    """
    This function gets the theme related texts for the given themes.
    """
    res: List = []
    development_texts = await get_news_developments_about_theme(
        GetThemeDevelopmentNewsInput(
            themes=themes_texts, max_devs_per_theme=MAX_DEVELOPMENTS_PER_TOPIC
        ),
        context,
    )
    article_texts = await get_news_articles_for_theme_developments(  # type: ignore
        GetThemeDevelopmentNewsArticlesInput(
            developments_list=development_texts,  # type: ignore
            max_articles_per_development=MAX_ARTICLES_PER_DEVELOPMENT,
        ),
        context,  # type: ignore
    )
    res.extend(development_texts)  # type: ignore
    res.extend(article_texts)  # type: ignore
    return res


# Test
async def main() -> None:
    input_text = "Write a commentary on impact of cloud computing on military industrial complex."
    user_message = Message(message=input_text, is_user_message=True, message_time=get_now_utc())
    chat_context = ChatContext(messages=[user_message])

    context = PlanRunContext(
        agent_id=str(uuid4()),
        plan_id=str(uuid4()),
        user_id=str(uuid4()),
        plan_run_id=str(uuid4()),
        chat=chat_context,
        skip_db_commit=True,
        skip_task_cache=True,
        run_tasks_without_prefect=True,
    )
    texts = await get_commentary_texts(
        GetCommentaryTextsInput(
            topics=["cloud computing", "military industrial complex"],
            start_date=datetime.date(2024, 4, 1),
            no_specific_topic=False,
        ),
        context,
    )
    print("Length of texts: ", len(texts))  # type: ignore
    args = WriteCommentaryInput(
        texts=texts,  # type: ignore
    )
    result = await write_commentary(args, context)
    print(result)
    print("-------------------------------------------------")
    print("-------------------------------------------------")
    print("General commentary")
    texts = await get_commentary_texts(
        GetCommentaryTextsInput(
            topics=[""],
            start_date=datetime.date(2024, 4, 1),
            no_specific_topic=True,
        ),
        context,
    )
    print("Length of texts: ", len(texts))  # type: ignore
    args = WriteCommentaryInput(
        texts=texts,  # type: ignore
    )
    result = await write_commentary(args, context)
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
