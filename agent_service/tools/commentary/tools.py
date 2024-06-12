import asyncio
import datetime
from typing import List, Optional
from uuid import uuid4

from agent_service.GPT.constants import GPT4_O
from agent_service.GPT.requests import GPT
from agent_service.io_types.dates import DateRange
from agent_service.io_types.table import STOCK_ID_COL_NAME_DEFAULT
from agent_service.io_types.text import Text, ThemeText
from agent_service.tool import ToolArgs, ToolCategory, tool
from agent_service.tools.commentary.constants import MAX_MATCHED_ARTICLES_PER_TOPIC
from agent_service.tools.commentary.helpers import (
    get_portfolio_geography_prompt,
    get_previous_commentary_results,
    get_region_weights_from_portfolio_holdings,
    get_theme_related_texts,
    organize_commentary_texts,
)
from agent_service.tools.commentary.prompts import (
    COMMENTARY_PROMPT_MAIN,
    COMMENTARY_SYS_PROMPT,
    GEOGRAPHY_PROMPT,
    NO_PROMPT,
    PREVIOUS_COMMENTARY_PROMPT,
    WRITING_FORMAT_TEXT_DICT,
    WRITING_STYLE_PROMPT,
)
from agent_service.tools.news import (
    GetNewsArticlesForTopicsInput,
    get_news_articles_for_topics,
)
from agent_service.tools.portfolio import (
    GetPortfolioWorkspaceHoldingsInput,
    PortfolioID,
    get_portfolio_holdings,
)
from agent_service.tools.themes import (
    GetMacroeconomicThemeInput,
    GetTopNThemesInput,
    get_macroeconomic_themes,
    get_top_N_macroeconomic_themes,
)
from agent_service.tools.tool_log import tool_log
from agent_service.types import ChatContext, Message, PlanRunContext
from agent_service.utils.date_utils import get_now_utc
from agent_service.utils.gpt_logging import GptJobIdType, GptJobType, create_gpt_context
from agent_service.utils.prefect import get_prefect_logger

# TODO:
# 1. comeplte sectors_prompt
# 2. complete top_stocks_prompt
# 3. complete client_type_prompt
# 4. complete goal_prompt
# 5. complete writing_style_prompt
# 6. complete theme_prompt and theme_ourlook_prompt
# 7. complete watchlist_stocks_prompt
# 8. use a portfolio by defualt for geography_prompt


class WriteCommentaryInput(ToolArgs):
    texts: List[Text]
    portfolio_id: Optional[PortfolioID] = None


@tool(
    description=(
        "This function can be used when a client wants to write a commentary, article or summary of "
        "market trends or specific topics."
        "This function generates a commentary either based for general market trends or "
        "based on specific topics mentioned by a client. "
        "The function creates a concise summary based on a comprehensive analysis of the provided texts. "
        "The commentary will be written in a professional tone, "
        "incorporating any specific instructions or preferences mentioned by the client during their interaction. "
        "The input to the function is prepared by the get_commentary_input tool."
    ),
    category=ToolCategory.COMMENTARY,
    reads_chat=True,
)
async def write_commentary(args: WriteCommentaryInput, context: PlanRunContext) -> Text:
    # Create GPT context and llm model
    gpt_context = create_gpt_context(
        GptJobType.AGENT_TOOLS, context.agent_id, GptJobIdType.AGENT_ID
    )
    llm = GPT(context=gpt_context, model=GPT4_O)

    # get previous commentary if exists
    previous_commentaries = await get_previous_commentary_results(context)
    previous_commentaries_text = (
        "\n***\n".join(Text.get_all_strs(previous_commentaries))
        if len(previous_commentaries) > 0
        else None
    )
    if previous_commentaries_text:
        await tool_log(
            log=f"Retrieved {len(previous_commentaries)} previous commentaries.",
            context=context,
        )

    # Prepare the portfolio geography prompt
    geography_prompt = NO_PROMPT.format()
    if args.portfolio_id:
        portfolio_holdings_table = await get_portfolio_holdings(
            GetPortfolioWorkspaceHoldingsInput(portfolio_id=args.portfolio_id), context
        ).to_df()  # type: ignore
        # convert DF to dict[int, float]
        weighted_holdings = portfolio_holdings_table.set_index(STOCK_ID_COL_NAME_DEFAULT)[
            "Weight"
        ].to_dict()

        # Get the region weights from the portfolio holdings
        regions_to_weight = get_region_weights_from_portfolio_holdings(weighted_holdings)
        geography_prompt = GEOGRAPHY_PROMPT.format(
            portfolio_geography=await get_portfolio_geography_prompt(regions_to_weight)  # type: ignore
        )
    else:
        await tool_log(
            log="No portfolio name is provided. Skipping portfolio based commentary.",
            context=context,
        )

    # Organize the commentary texts into themes, developments and articles
    themes, developments, articles = await organize_commentary_texts(args.texts)
    await tool_log(
        log=f"Retrieved {len(themes)} themes, {len(developments)} developments, and {len(articles)} articles.",
        context=context,
    )
    # create main prompt
    main_prompt = COMMENTARY_PROMPT_MAIN.format(
        previous_commentary_prompt=(
            PREVIOUS_COMMENTARY_PROMPT.format(previous_commentaries=previous_commentaries_text)
            if previous_commentaries_text is not None
            else ""
        ),
        geography_prompt=geography_prompt,
        writing_style_prompt=WRITING_STYLE_PROMPT.format(
            writing_format=WRITING_FORMAT_TEXT_DICT.get("Long"),
        ),
        themes="\n***\n".join(Text.get_all_strs(themes)),
        developments="\n***\n".join(Text.get_all_strs(developments)),
        articles="\n***\n".join(Text.get_all_strs(articles)),
        chat_context=context.chat.get_gpt_input() if context.chat is not None else "",
    )
    # Write the commentary
    result = await llm.do_chat_w_sys_prompt(
        main_prompt=main_prompt,
        sys_prompt=COMMENTARY_SYS_PROMPT.format(),
    )

    return Text(val=result)


class GetCommentaryTextsInput(ToolArgs):
    topics: List[str] = None  # type: ignore
    start_date: Optional[datetime.date] = None
    date_range: Optional[DateRange] = None
    portfolio_id: Optional[str] = None


@tool(
    description=(
        "This function can be used when a client wants to write a commentary, article or summary of "
        "market trends or specific topics."
        "This function collects and prepares all texts to be used by the write_commentary tool "
        "for writing a commentary or short articles and market summaries. "
        "This function MUST only be used for write commentary tool. "
        "If client wants a general commentary with no specific topics in mind, topics MUST be None. "
        "Adjust start_date to get the text from that date based on client request. "
        "If no start_date is provided, the function will only get text in last month. "
    ),
    category=ToolCategory.COMMENTARY,
)
async def get_commentary_texts(
    args: GetCommentaryTextsInput, context: PlanRunContext
) -> List[Text]:
    if not args.start_date:
        if args.date_range:
            args.start_date = args.date_range.start_date
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
            theme_related_texts = await get_theme_related_texts(themes_texts, context)
            await tool_log(
                log="Retrieved texts for top themes related to portfolio (id: {args.portfolio_id}).",
                context=context,
            )
            return themes_texts + theme_related_texts
        else:
            # get popular topics
            themes_texts: List[ThemeText] = await get_top_N_macroeconomic_themes(  # type: ignore
                GetTopNThemesInput(start_date=args.start_date, theme_num=3), context
            )
            theme_related_texts = await get_theme_related_texts(themes_texts, context)
            await tool_log(
                log="No portfolio is provided. Retrieved texts for top 3 themes for general commentary.",
                context=context,
            )
            return themes_texts + theme_related_texts

    else:
        if args.portfolio_id:
            themes_texts: List[ThemeText] = await get_top_N_macroeconomic_themes(  # type: ignore
                GetTopNThemesInput(
                    start_date=args.start_date, theme_num=3, portfolio_id=args.portfolio_id
                ),
                context,
            )
            theme_related_texts = await get_theme_related_texts(themes_texts, context)
            await tool_log(
                log=f"Retrieved texts for top 3 themes related to portfolio (id: {args.portfolio_id}).",
                context=context,
            )
            topic_texts = await get_texts_for_topics(args, context)
            return themes_texts + theme_related_texts + topic_texts
        else:
            await tool_log(
                log=f"No portfolio is provided. Retrieving texts for the topics: {args.topics}.",
                context=context,
            )
            texts = await get_texts_for_topics(args, context)
            return texts


async def get_texts_for_topics(
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
            res = await get_theme_related_texts(themes, context)  # type: ignore
            texts.extend(res + themes)  # type: ignore

        except Exception as e:
            logger.warning(f"Failed to find any news theme for topic {topic}: {e}")
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
                logger.warning(f"Failed to get news pool articles for topic {topic}: {e}")

    if len(texts) == 0:
        raise Exception("No data collected for commentary from available sources")

    return texts


# Test
async def main() -> None:
    input_text = "Write a commentary on impact of cloud computing on military industrial complex."
    user_message = Message(message=input_text, is_user_message=True, message_time=get_now_utc())
    chat_context = ChatContext(messages=[user_message])

    context = PlanRunContext(
        agent_id="7cb9fb8f-690e-4535-8b48-f6e63494c366",
        plan_id="b3330500-9870-480d-bcb1-cf6fe6b487e3",
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
            start_date=datetime.date(2024, 4, 1),
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
