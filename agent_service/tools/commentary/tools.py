import asyncio
from datetime import date, timedelta
from typing import List, Optional
from uuid import uuid4

from agent_service.GPT.constants import NO_PROMPT
from agent_service.GPT.requests import GPT
from agent_service.io_type_utils import HistoryEntry
from agent_service.io_types.dates import DateRange
from agent_service.io_types.stock import StockID
from agent_service.io_types.text import Text, TextGroup, ThemeText
from agent_service.tool import ToolArgs, ToolCategory, tool
from agent_service.tools.commentary.constants import (
    COMMENTARY_LLM,
    MAX_DEVELOPMENTS_PER_COMMENTARY,
    MAX_STOCKS_PER_COMMENTARY,
    MAX_THEMES_PER_COMMENTARY,
    MAX_TOTAL_ARTICLES_PER_COMMENTARY,
)
from agent_service.tools.commentary.helpers import (
    filter_most_important_citations,
    get_previous_commentary_results,
    get_texts_for_topics,
    get_theme_related_texts,
    organize_commentary_texts,
    prepare_main_prompt,
    prepare_portfolio_prompt,
    prepare_stocks_stats_prompt,
    split_text_and_citation_ids,
)
from agent_service.tools.commentary.prompts import (
    CLIENTELE_TEXT_DICT,
    CLIENTELE_TYPE_PROMPT,
    COMMENTARY_SYS_PROMPT,
    GET_COMMENTARY_INPUTS_DESCRIPTION,
    LONG_WRITING_STYLE,
    PREVIOUS_COMMENTARY_PROMPT,
    SIMPLE_CLIENTELE,
    UPDATE_COMMENTARY_INSTRUCTIONS,
    WATCHLIST_PROMPT,
    WRITE_COMMENTARY_DESCRIPTION,
    WRITING_FORMAT_TEXT_DICT,
    WRITING_STYLE_PROMPT,
)
from agent_service.tools.other_text import (
    GetAllTextDataForStocksInput,
    get_all_text_data_for_stocks,
)
from agent_service.tools.portfolio import PortfolioID
from agent_service.tools.themes import (
    GetTopNThemesInput,
    get_top_N_macroeconomic_themes,
)
from agent_service.tools.tool_log import tool_log
from agent_service.tools.watchlist import (
    GetStocksForUserAllWatchlistsInput,
    get_stocks_for_user_all_watchlists,
)
from agent_service.types import ChatContext, Message, PlanRunContext
from agent_service.utils.date_utils import get_now_utc
from agent_service.utils.gpt_logging import GptJobIdType, GptJobType, create_gpt_context
from agent_service.utils.prefect import get_prefect_logger

logger = get_prefect_logger(__name__)


class WriteCommentaryInput(ToolArgs):
    date_range: DateRange = DateRange(
        start_date=date.today() - timedelta(days=30),
        end_date=date.today(),
    )
    inputs: List[Text]
    stock_ids: Optional[List[StockID]] = None
    client_type: Optional[str] = "Simple"
    writing_format: Optional[str] = "Long"
    portfolio_id: Optional[PortfolioID] = None


@tool(
    description=WRITE_COMMENTARY_DESCRIPTION,
    category=ToolCategory.COMMENTARY,
    reads_chat=True,
    update_instructions=UPDATE_COMMENTARY_INSTRUCTIONS,
)
async def write_commentary(args: WriteCommentaryInput, context: PlanRunContext) -> Text:

    # get previous commentary if exists
    previous_commentaries = await get_previous_commentary_results(context)
    previous_commentary = previous_commentaries[0] if previous_commentaries else None
    if previous_commentary:
        await tool_log(
            log="Retrieved previous commentary.",
            context=context,
            associated_data=previous_commentary,
        )

    # Prepare the portfolio prompt
    portfolio_prompt = NO_PROMPT
    if args.portfolio_id:
        portfolio_prompt = await prepare_portfolio_prompt(
            args.portfolio_id, args.date_range, context
        )
    else:
        await tool_log(
            log="No portfolio name is provided. Skipping portfolio based commentary.",
            context=context,
        )

    # Prepare the stock performance prompt
    stocks_stats_prompt = NO_PROMPT
    if args.stock_ids:
        stocks_stats_prompt = await prepare_stocks_stats_prompt(
            args.stock_ids, args.date_range, context
        )

    # Dedupluate the texts and organize the commentary texts into themes, developments and articles
    text_ids = set()
    deduplicated_texts = []
    for text in args.inputs:
        if isinstance(text, Text):
            if text.id not in text_ids:
                text_ids.add(text.id)
                deduplicated_texts.append(text)
    text_mapping = await organize_commentary_texts(deduplicated_texts)
    # check the max number of texts in each type
    if len(text_mapping["Theme Description"]) > MAX_THEMES_PER_COMMENTARY:
        text_mapping["Theme Description"] = text_mapping["Theme Description"][
            :MAX_THEMES_PER_COMMENTARY
        ]
    if len(text_mapping["News Development Summary"]) > MAX_DEVELOPMENTS_PER_COMMENTARY:
        text_mapping["News Development Summary"] = text_mapping["News Development Summary"][
            :MAX_DEVELOPMENTS_PER_COMMENTARY
        ]
    if len(text_mapping["News Article Summary"]) > MAX_TOTAL_ARTICLES_PER_COMMENTARY:
        text_mapping["News Article Summary"] = text_mapping["News Article Summary"][
            :MAX_TOTAL_ARTICLES_PER_COMMENTARY
        ]
    # show number of texts of each type
    await tool_log(
        log=f"Retrieved texts of each type: {', '.join([f'{k}: {len(v)}' for k, v in text_mapping.items()])}",
        context=context,
    )
    # convert text_mapping to list of texts
    all_text_group = TextGroup(
        val=[text for text_list in text_mapping.values() for text in text_list]
    )

    # Prepare previous commentary prompt
    previous_commentary_prompt = (
        PREVIOUS_COMMENTARY_PROMPT.format(
            previous_commentary=await Text.get_all_strs(previous_commentary)
        )
        if previous_commentary is not None
        else NO_PROMPT
    )
    # Prepare watchlist prompt
    watchlist_prompt = NO_PROMPT
    watchlist_stocks = await get_stocks_for_user_all_watchlists(
        GetStocksForUserAllWatchlistsInput(), context
    )
    await tool_log(
        log="Retrieving watchlist stocks for commentary.",
        context=context,
        associated_data=watchlist_stocks,
    )
    # check the number of watchlist stocks
    if len(watchlist_stocks) > MAX_STOCKS_PER_COMMENTARY:  # type: ignore
        watchlist_stocks = watchlist_stocks[:MAX_STOCKS_PER_COMMENTARY]  # type: ignore
        await tool_log(
            log=(
                f"Number of watchlist stocks is more than {MAX_STOCKS_PER_COMMENTARY}. "
                f"Only first {MAX_STOCKS_PER_COMMENTARY} stocks will be considered."
            ),
            context=context,
        )

    watchlist_prompt = WATCHLIST_PROMPT.format(
        watchlist_stocks=", ".join([await stock.to_gpt_input() for stock in watchlist_stocks])  # type: ignore
    )

    # Prepare client type prompt
    client_type = args.client_type if args.client_type else "Simple"
    client_type_prompt = CLIENTELE_TYPE_PROMPT.format(
        client_type=CLIENTELE_TEXT_DICT.get(client_type, SIMPLE_CLIENTELE)
    )

    # Prepare writing style prompt
    writing_format = args.writing_format if args.writing_format else "Long"
    writing_style_prompt = WRITING_STYLE_PROMPT.format(
        writing_format=WRITING_FORMAT_TEXT_DICT.get(writing_format, LONG_WRITING_STYLE)
    )

    # Prepare texts for commentary
    texts = await Text.get_all_strs(all_text_group, include_header=True, text_group_numbering=True)

    # Prepare the main prompt using the above prompts
    main_prompt = await prepare_main_prompt(
        portfolio_prompt,
        stocks_stats_prompt,
        previous_commentary_prompt,
        watchlist_prompt,
        client_type_prompt,
        writing_style_prompt,
        texts,  # type: ignore
        text_mapping,
        context,
    )

    # save main prompt as text file for debugging
    # with open("main_prompt.txt", "w") as f:
    #     f.write(main_prompt.filled_prompt)

    # Write the commentary
    await tool_log(
        log=f"Writing commentary for '{client_type}' client in '{writing_format}' format.",
        context=context,
    )
    # Create GPT context and llm model
    gpt_context = create_gpt_context(
        GptJobType.AGENT_TOOLS, context.agent_id, GptJobIdType.AGENT_ID
    )
    llm = GPT(context=gpt_context, model=COMMENTARY_LLM)

    # get commentary text and citations
    for _ in range(3):
        try:
            result = await llm.do_chat_w_sys_prompt(
                main_prompt=main_prompt,
                sys_prompt=COMMENTARY_SYS_PROMPT.format(),
            )
            commentary_text, citation_ids = await split_text_and_citation_ids(result)
            break
        except Exception as e:
            logger.exception(f"Failed to split text and citation ids: {e}")

    # filter most important citations if more than 50
    if len(citation_ids) > 50:
        # filter most important citations
        citation_ids = await filter_most_important_citations(
            texts, commentary_text, citation_ids  # type: ignore
        )
    # create commentary object
    commentary = Text(val=commentary_text)
    commentary = commentary.inject_history_entry(
        HistoryEntry(title="Commentary", citations=all_text_group.get_citations(citation_ids))
    )

    return commentary


class GetCommentaryInputsInput(ToolArgs):
    topics: Optional[List[str]] = None
    stock_ids: Optional[List[StockID]] = None
    date_range: DateRange = DateRange(
        start_date=date.today() - timedelta(days=30),
        end_date=date.today(),
    )
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

    logger = get_prefect_logger(__name__)
    texts: List[Text] = []

    # If general_commentary is True, get the top themes and related texts
    if args.general_commentary:
        if args.portfolio_id:
            await tool_log(
                log=f"Portfolio is provided. Retrieving top {args.theme_num} market themes...",
                context=context,
            )
        else:
            await tool_log(
                log=f"No portfolio is provided. Retrieving top {args.theme_num} market themes...",
                context=context,
            )
        try:
            # get top themes
            theme_num: int = args.theme_num if args.theme_num else 3
            themes_texts: List[ThemeText] = await get_top_N_macroeconomic_themes(  # type: ignore
                GetTopNThemesInput(
                    date_range=args.date_range, theme_num=theme_num, portfolio_id=args.portfolio_id
                ),
                context,
            )
            await tool_log(
                log=f"Retrieved {len(themes_texts)} top themes for commentary.",
                context=context,
                associated_data=themes_texts,
            )
            themes_texts = themes_texts[:MAX_THEMES_PER_COMMENTARY]
            theme_related_texts = await get_theme_related_texts(
                themes_texts, args.date_range, context
            )
            texts.extend(themes_texts + theme_related_texts)
            await tool_log(
                log=f"Retrieved {len(texts)} theme related texts for top market trends.",
                context=context,
                associated_data=theme_related_texts,
            )
        except Exception as e:
            logger.exception(f"Failed to get top themes and related texts: {e}")

    # If topics are provided, get the texts for the topics
    if args.topics:
        try:
            topic_texts = await get_texts_for_topics(args.topics, args.date_range, context)
            await tool_log(
                log=f"Retrieved {len(topic_texts)} texts for topics {args.topics}.",
                context=context,
            )
            texts.extend(topic_texts)
        except Exception as e:
            logger.exception(f"Failed to get texts for topics: {e}")

    # If stock_ids are provided, get the texts for the stock_ids
    if args.stock_ids:
        if len(args.stock_ids) > MAX_STOCKS_PER_COMMENTARY:
            await tool_log(
                log=(
                    f"Number of stock ids provided is more than {MAX_STOCKS_PER_COMMENTARY}. "
                    f"Only first {MAX_STOCKS_PER_COMMENTARY} stocks will be considered."
                ),
                context=context,
            )
            args.stock_ids = args.stock_ids[:MAX_STOCKS_PER_COMMENTARY]
        try:
            stock_texts: List[Text] = await get_all_text_data_for_stocks(  # type: ignore
                GetAllTextDataForStocksInput(stock_ids=args.stock_ids, date_range=args.date_range),
                context,
            )
            await tool_log(
                log=f"Retrieved {len(stock_texts)} texts for given stock ids.",
                context=context,
                associated_data=stock_texts,
            )
            texts.extend(stock_texts)
        except Exception as e:
            logger.exception(f"Failed to get texts for stock ids: {e}")

    return texts


# Test
async def main() -> None:
    input_text = "Write a general commentary with focus on cloud computing and my portfolio."
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
