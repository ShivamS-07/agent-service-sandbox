import asyncio
import random
from typing import List, Optional
from uuid import uuid4

from agent_service.GPT.constants import GPT4_O
from agent_service.GPT.requests import GPT
from agent_service.io_type_utils import HistoryEntry
from agent_service.io_types.dates import DateRange
from agent_service.io_types.table import STOCK_ID_COL_NAME_DEFAULT
from agent_service.io_types.text import Text, TextGroup, ThemeText
from agent_service.tool import ToolArgs, ToolCategory, tool
from agent_service.tools.commentary.constants import (
    MAX_MATCHED_ARTICLES_PER_TOPIC,
    MAX_THEMES_PER_COMMENTARY,
    MAX_TOTAL_ARTICLES_PER_COMMENTARY,
)
from agent_service.tools.commentary.helpers import (
    get_portfolio_geography_prompt,
    get_previous_commentary_results,
    get_region_weights_from_portfolio_holdings,
    get_theme_related_texts,
    organize_commentary_texts,
    split_text_and_citation_ids,
)
from agent_service.tools.commentary.prompts import (
    COMMENTARY_PROMPT_MAIN,
    COMMENTARY_SYS_PROMPT,
    GEOGRAPHY_PROMPT,
    GET_COMMENTARY_INPUTS_DESCRIPTION,
    NO_PROMPT,
    PREVIOUS_COMMENTARY_PROMPT,
    WRITE_COMMENTARY_DESCRIPTION,
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

# - How to handle large size of texts?
#    - one option is do some filtering using another gpt to select the best texts based on user prompt
# 1. comeplte sectors_prompt
# 2. complete top_stocks_prompt
# 3. complete client_type_prompt
# 4. complete goal_prompt
# 6. complete theme_prompt and theme_outlook_prompt
# 7. complete watchlist_stocks_prompt
# 8. use a portfolio by defualt for geography_prompt


class WriteCommentaryInput(ToolArgs):
    inputs: List[Text]  # type: ignore
    portfolio_id: Optional[PortfolioID] = None


@tool(
    description=WRITE_COMMENTARY_DESCRIPTION,
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
    previous_commentary = previous_commentaries[0] if previous_commentaries else None
    if previous_commentary:
        await tool_log(
            log="Retrieved previous commentary.",
            context=context,
        )

    # Prepare the portfolio geography prompt
    geography_prompt = NO_PROMPT.format()
    if args.portfolio_id:
        portfolio_holdings_table = (
            await get_portfolio_holdings(
                GetPortfolioWorkspaceHoldingsInput(portfolio_id=args.portfolio_id), context
            )
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

    # Dedupluate the texts and organize the commentary texts into themes, developments and articles
    text_ids = set()
    deduplicated_texts = []
    for text in args.inputs:
        if isinstance(text, Text):
            if text.id not in text_ids:
                text_ids.add(text.id)
                deduplicated_texts.append(text)
    themes, developments, articles = await organize_commentary_texts(deduplicated_texts)

    # if number of articles is more than MAX_TOTAL_ARTICLES_PER_COMMENTARY,
    # randomly select specified number of themes, developments and articles
    if len(articles) > MAX_TOTAL_ARTICLES_PER_COMMENTARY:
        articles = random.sample(articles, MAX_TOTAL_ARTICLES_PER_COMMENTARY)

    all_texts = themes + developments + articles
    all_text_group = TextGroup(val=all_texts)

    await tool_log(
        log=f"Retrieved {len(themes)} themes, {len(developments)} developments, and {len(articles)} articles.",
        context=context,
    )

    # create main prompt
    previous_commentary_prompt = (
        PREVIOUS_COMMENTARY_PROMPT.format(
            previous_commentary=await Text.get_all_strs(previous_commentary)
        )
        if previous_commentary is not None
        else NO_PROMPT.format()
    )
    writing_style_prompt = WRITING_STYLE_PROMPT.format(
        writing_format=WRITING_FORMAT_TEXT_DICT.get("Long")
    )
    texts = await Text.get_all_strs(all_text_group, include_header=True, text_group_numbering=True)
    chat_context = context.chat.get_gpt_input() if context.chat is not None else ""

    main_prompt = COMMENTARY_PROMPT_MAIN.format(
        previous_commentary_prompt=previous_commentary_prompt.filled_prompt,
        geography_prompt=geography_prompt.filled_prompt,
        writing_style_prompt=writing_style_prompt.filled_prompt,
        texts=texts,
        chat_context=chat_context,
    )
    # save main prompt as text file for debugging
    # with open("main_prompt.txt", "w") as f:
    #     f.write(main_prompt.filled_prompt)

    # Write the commentary
    result = await llm.do_chat_w_sys_prompt(
        main_prompt=main_prompt,
        sys_prompt=COMMENTARY_SYS_PROMPT.format(),
    )
    text, citation_ids = await split_text_and_citation_ids(result)
    commentary = Text(val=text)
    commentary = commentary.inject_history_entry(
        HistoryEntry(title="Commentary", citations=all_text_group.get_citations(citation_ids))
    )

    return commentary


class GetCommentaryInputsInput(ToolArgs):
    topics: List[str] = None  # type: ignore
    date_range: Optional[DateRange] = None
    portfolio_id: Optional[str] = None
    general_commentary: Optional[bool] = False
    theme_num: Optional[int] = 3


@tool(
    description=GET_COMMENTARY_INPUTS_DESCRIPTION,
    category=ToolCategory.COMMENTARY,
)
async def get_commentary_inputs(
    args: GetCommentaryInputsInput, context: PlanRunContext
) -> List[Text]:

    texts: List[Text] = []

    # If general_commentary is True, get the top themes and related texts
    if args.general_commentary:
        if args.portfolio_id:
            await tool_log(
                log=f"Retrieving texts for top {args.theme_num} themes related to portfolio (id: {args.portfolio_id}).",
                context=context,
            )
        else:
            await tool_log(
                log=f"No portfolio is provided. Retrieving top {args.theme_num} market themes...",
                context=context,
            )
        # get top themes
        theme_num: int = args.theme_num if args.theme_num else 3
        themes_texts: List[ThemeText] = await get_top_N_macroeconomic_themes(  # type: ignore
            GetTopNThemesInput(
                date_range=args.date_range, theme_num=theme_num, portfolio_id=args.portfolio_id
            ),
            context,
        )
        themes_texts = themes_texts[:MAX_THEMES_PER_COMMENTARY]
        theme_related_texts = await get_theme_related_texts(themes_texts, context)
        texts.extend(themes_texts + theme_related_texts)
        await tool_log(
            log=f"Retrieved {len(texts)} theme related texts for top market trends.",
            context=context,
        )

    # If topics are provided, get the texts for the topics
    if args.topics:
        topic_texts = await get_texts_for_topics(args, context)
        await tool_log(
            log=f"Retrieved {len(topic_texts)} texts for topics {args.topics}.",
            context=context,
        )
        texts.extend(topic_texts)

    return texts


async def get_texts_for_topics(
    args: GetCommentaryInputsInput, context: PlanRunContext
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
                        date_range=args.date_range,
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
    input_text = "Write a general commentary with focus on impact of cloud computing on military industrial complex."
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
    texts = await get_commentary_inputs(
        GetCommentaryInputsInput(
            topics=["cloud computing", "military industrial complex"],
            date_range=None,
            general_commentary=True,
            theme_num=4,
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
