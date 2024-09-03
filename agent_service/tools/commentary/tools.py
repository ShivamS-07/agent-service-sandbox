from datetime import date, timedelta
from typing import Dict, List, Optional, Union

from agent_service.GPT.constants import NO_PROMPT
from agent_service.GPT.requests import GPT
from agent_service.GPT.tokens import GPTTokenizer
from agent_service.io_type_utils import HistoryEntry
from agent_service.io_types.dates import DateRange
from agent_service.io_types.stock import StockID
from agent_service.io_types.table import STOCK_ID_COL_NAME_DEFAULT, StockTable, Table
from agent_service.io_types.text import (
    StockNewsDevelopmentText,
    Text,
    TextGroup,
    ThemeNewsDevelopmentArticlesText,
    ThemeNewsDevelopmentText,
    ThemeText,
)
from agent_service.tool import ToolArgs, ToolCategory, tool
from agent_service.tools.commentary.constants import (
    COMMENTARY_LLM,
    MAX_DEVELOPMENTS_PER_COMMENTARY,
    MAX_STOCKS_PER_COMMENTARY,
    MAX_THEMES_PER_COMMENTARY,
    MAX_TOTAL_ARTICLES_PER_COMMENTARY,
)
from agent_service.tools.commentary.helpers import (
    get_previous_commentary_results,
    get_texts_for_topics,
    get_theme_related_texts,
    organize_commentary_inputs,
    prepare_main_prompt,
    prepare_portfolio_prompt,
    prepare_universe_prompt,
)
from agent_service.tools.commentary.prompts import (
    CLIENTELE_TEXT_DICT,
    CLIENTELE_TYPE_PROMPT,
    COMMENTARY_SYS_PROMPT,
    GET_COMMENTARY_INPUTS_DESCRIPTION,
    LONG_WRITING_STYLE,
    PREVIOUS_COMMENTARY_PROMPT,
    SIMPLE_CLIENTELE,
    STOCKS_STATS_PROMPT,
    UPDATE_COMMENTARY_INSTRUCTIONS,
    WATCHLIST_PROMPT,
    WRITE_COMMENTARY_DESCRIPTION,
    WRITING_FORMAT_TEXT_DICT,
    WRITING_STYLE_PROMPT,
)
from agent_service.tools.LLM_analysis.utils import extract_citations_from_gpt_output
from agent_service.tools.news import (
    GetNewsArticlesForStockDevelopmentsInput,
    GetNewsDevelopmentsAboutCompaniesInput,
    get_all_news_developments_about_companies,
    get_news_articles_for_stock_developments,
)
from agent_service.tools.portfolio import (
    BENCHMARK_PERFORMANCE_LEVELS,
    PORTFOLIO_PERFORMANCE_LEVELS,
    GetPortfolioBenchmarkHoldingsInput,
    GetPortfolioBenchmarkPerformanceInput,
    GetPortfolioHoldingsInput,
    GetPortfolioPerformanceInput,
    get_portfolio_benchmark_holdings,
    get_portfolio_benchmark_performance,
    get_portfolio_holdings,
    get_portfolio_performance,
)
from agent_service.tools.themes import (
    GetTopNThemesInput,
    get_top_N_macroeconomic_themes,
)
from agent_service.tools.tool_log import tool_log
from agent_service.tools.universe import (
    STOCK_PERFORMANCE_TABLE_NAME,
    UNIVERSE_PERFORMANCE_LEVELS,
    GetStocksPerformanceInput,
    GetUniverseHoldingsInput,
    GetUniversePerformanceInput,
    get_stocks_performance,
    get_universe_holdings,
    get_universe_performance,
)
from agent_service.tools.watchlist import (
    GetStocksForUserAllWatchlistsInput,
    get_stocks_for_user_all_watchlists,
)
from agent_service.types import PlanRunContext
from agent_service.utils.gpt_logging import GptJobIdType, GptJobType, create_gpt_context
from agent_service.utils.prefect import get_prefect_logger


class WriteCommentaryInput(ToolArgs):
    inputs: List[Union[Text, Table]]
    client_type: Optional[str] = "Simple"
    writing_format: Optional[str] = "Long"


@tool(
    description=WRITE_COMMENTARY_DESCRIPTION,
    category=ToolCategory.COMMENTARY,
    reads_chat=True,
    update_instructions=UPDATE_COMMENTARY_INSTRUCTIONS,
)
async def write_commentary(args: WriteCommentaryInput, context: PlanRunContext) -> Text:
    logger = get_prefect_logger(__name__)

    # get previous commentary if exists
    previous_commentaries = await get_previous_commentary_results(context)
    previous_commentary = previous_commentaries[0] if previous_commentaries else None
    if previous_commentary:
        await tool_log(
            log="Retrieved previous commentary.",
            context=context,
            associated_data=previous_commentary,
        )

    # Prepare previous commentary prompt
    previous_commentary_prompt = (
        PREVIOUS_COMMENTARY_PROMPT.format(
            previous_commentary=await Text.get_all_strs(previous_commentary)
        )
        if previous_commentary is not None
        else NO_PROMPT
    )

    # Prepare the portfolio prompt
    portfolio_prompt = NO_PROMPT
    portfolio_related_tables: List[Table] = []
    for input in args.inputs:
        if isinstance(input, Table):
            if input.title:
                if ("portfolio" in input.title.lower()) or ("benchmark" in input.title.lower()):
                    portfolio_related_tables.append(input)
    if portfolio_related_tables:
        try:
            portfolio_prompt = await prepare_portfolio_prompt(
                portfolio_related_tables, context=context
            )
        except Exception as e:
            logger.info(f"Failed to prepare portfolio prompt: {e}")

    # Prepare the universe performance prompt
    universe_performance_prompt = NO_PROMPT
    universe_related_tables: List[Table] = []
    for input in args.inputs:
        if isinstance(input, Table) and input.title:
            if "universe" in input.title.lower():
                universe_related_tables.append(input)
    if universe_related_tables:
        try:
            universe_performance_prompt = await prepare_universe_prompt(
                universe_related_tables, context=context
            )
        except Exception as e:
            logger.info(f"Failed to prepare universe performance prompt: {e}")
    # Dedupluate the texts and organize the commentary texts into themes, developments and articles
    input_mapping = await organize_commentary_inputs(args.inputs)

    # seperate text_mapping from input_mapping
    text_mapping: Dict[str, List[Text]] = {
        key: value for key, value in input_mapping.items() if key != "Tables"  # type: ignore
    }
    # check the max number of texts in each type
    if ThemeText.text_type in text_mapping:
        text_type = ThemeText.text_type
        if len(text_mapping[text_type]) > MAX_THEMES_PER_COMMENTARY:
            text_mapping[text_type] = text_mapping[text_type][:MAX_THEMES_PER_COMMENTARY]
    if ThemeNewsDevelopmentText.text_type in text_mapping:
        text_type = ThemeNewsDevelopmentText.text_type
        if len(text_mapping[text_type]) > MAX_DEVELOPMENTS_PER_COMMENTARY:
            text_mapping[text_type] = text_mapping[text_type][:MAX_DEVELOPMENTS_PER_COMMENTARY]
    if ThemeNewsDevelopmentArticlesText.text_type in text_mapping:
        text_type = ThemeNewsDevelopmentArticlesText.text_type
        if len(text_mapping[text_type]) > MAX_TOTAL_ARTICLES_PER_COMMENTARY:
            text_mapping[text_type] = text_mapping[text_type][:MAX_TOTAL_ARTICLES_PER_COMMENTARY]
    # show number of texts of each type if there are values in text_mapping
    if text_mapping:
        await tool_log(
            log=f"Texts used for commentary: {', '.join([f'{k}: {len(v)}' for k, v in text_mapping.items()])}",
            context=context,
        )
    # convert text_mapping to list of texts
    all_text_group = TextGroup(
        val=[text for text_list in text_mapping.values() for text in text_list]
    )

    # Prepare the stock performance prompt
    stocks_stats_prompt = NO_PROMPT
    for table in input_mapping.get("Tables", []):
        if isinstance(table, Table) and STOCK_PERFORMANCE_TABLE_NAME == table.title:
            stocks_stats_df = table.to_df()[STOCK_ID_COL_NAME_DEFAULT].apply(
                lambda x: x.company_name
            )
            stocks_stats_prompt = STOCKS_STATS_PROMPT.format(
                stock_stats=stocks_stats_df.to_string(index=False)
            )

    # Prepare watchlist prompt
    watchlist_prompt = NO_PROMPT
    watchlist_stocks: List[StockID] = await get_stocks_for_user_all_watchlists(  # type: ignore
        GetStocksForUserAllWatchlistsInput(), context
    )
    watchlist_stock_names = [stock.company_name for stock in watchlist_stocks]

    watchlist_prompt = WATCHLIST_PROMPT.format(
        watchlist_stocks=", ".join([stock for stock in watchlist_stock_names])  # type: ignore
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
    texts_str: str = await Text.get_all_strs(  # type: ignore
        all_text_group, include_header=True, text_group_numbering=True
    )

    # Prepare the main prompt using the above prompts
    main_prompt = await prepare_main_prompt(
        previous_commentary_prompt=previous_commentary_prompt,
        portfolio_prompt=portfolio_prompt,
        universe_performance_prompt=universe_performance_prompt,
        stocks_stats_prompt=stocks_stats_prompt,
        watchlist_prompt=watchlist_prompt,
        client_type_prompt=client_type_prompt,
        writing_style_prompt=writing_style_prompt,
        texts=texts_str,
        text_mapping=text_mapping,
        context=context,
    )
    main_prompt_token_length = GPTTokenizer(COMMENTARY_LLM).get_token_length(
        main_prompt.filled_prompt
    )
    logger.info(f"Length of tokens in main prompt: {main_prompt_token_length}")

    # save main prompt as text file for debugging
    # with open("main_prompt.txt", "w") as f:
    #     f.write(main_prompt.filled_prompt)

    # Write the commentary
    await tool_log(
        log=f"Writing a '{client_type}' commentary in '{writing_format}' format.",
        context=context,
    )
    # Create GPT context and llm model
    gpt_context = create_gpt_context(
        GptJobType.AGENT_TOOLS, context.agent_id, GptJobIdType.AGENT_ID
    )
    llm = GPT(context=gpt_context, model=COMMENTARY_LLM)
    result = await llm.do_chat_w_sys_prompt(
        main_prompt=main_prompt, sys_prompt=COMMENTARY_SYS_PROMPT.format(), no_cache=True
    )
    commentary_text, citations = await extract_citations_from_gpt_output(
        result, all_text_group, context
    )
    if texts_str and not citations:  # missing all citations, do a retry
        result = await llm.do_chat_w_sys_prompt(
            main_prompt=main_prompt, sys_prompt=COMMENTARY_SYS_PROMPT.format(), no_cache=True
        )
        commentary_text, citations = await extract_citations_from_gpt_output(
            result, all_text_group, context
        )
        logger.info(f"Size of citations: {len(citations)}")  # type:ignore

    # create commentary object
    commentary: Text = Text(val=commentary_text or result)
    commentary = commentary.inject_history_entry(
        HistoryEntry(title="Commentary", citations=citations)  # type:ignore
    )

    return commentary


class GetCommentaryInputsInput(ToolArgs):
    stock_ids: List[StockID] = []
    topics: Optional[List[str]] = None
    universe_name: Optional[str] = None
    date_range: DateRange = DateRange(
        start_date=date.today() - timedelta(days=30),
        end_date=date.today(),
    )
    portfolio_id: Optional[str] = None
    market_trend: Optional[bool] = False
    theme_num: Optional[int] = 3


@tool(
    description=GET_COMMENTARY_INPUTS_DESCRIPTION,
    category=ToolCategory.COMMENTARY,
)
async def get_commentary_inputs(
    args: GetCommentaryInputsInput, context: PlanRunContext
) -> List[Union[Text, Table]]:

    logger = get_prefect_logger(__name__)
    texts: List[Text] = []
    tables: List[Table] = []

    # If market_trend is True, get the top themes and related texts
    if args.market_trend:
        try:
            # get top themes
            theme_num: int = args.theme_num if args.theme_num else 3
            themes_texts_set = set()
            if args.portfolio_id:
                try:
                    themes_texts_portfolio_related: List[ThemeText] = (
                        await get_top_N_macroeconomic_themes(  # type: ignore
                            GetTopNThemesInput(
                                date_range=args.date_range,
                                theme_num=theme_num,
                                portfolio_id=args.portfolio_id,
                            ),
                            context,
                        )
                    )
                    themes_texts_set.update(themes_texts_portfolio_related)
                except Exception as e:
                    logger.exception(f"Failed to get top themes for portfolio: {e}")

            themes_texts_general: List[ThemeText] = await get_top_N_macroeconomic_themes(  # type: ignore
                GetTopNThemesInput(
                    date_range=args.date_range, theme_num=theme_num, portfolio_id=None
                ),
                context,
            )
            # combine themes_texts and remove duplicates
            themes_texts_set.update(themes_texts_general)
            themes_texts_list = list(themes_texts_set)

            await tool_log(
                log=f"Retrieved {len(themes_texts_list)} top themes for commentary.",
                context=context,
                associated_data=themes_texts_list,
            )
            themes_texts_list = themes_texts_list[:MAX_THEMES_PER_COMMENTARY]
            theme_related_texts = await get_theme_related_texts(
                themes_texts_list, args.date_range, context
            )
            texts.extend(themes_texts_list + theme_related_texts)
            await tool_log(
                log=f"Retrieved {len(texts)} theme related texts for top market trends.",
                context=context,
                associated_data=theme_related_texts,
            )
        except Exception as e:
            logger.exception(f"Failed to get top themes and related texts: {e}")

    # If topics are provided, get the texts for the topics
    if args.topics:
        args.topics = args.topics + [args.universe_name] if args.universe_name else args.topics
        try:
            topic_texts = await get_texts_for_topics(args.topics, args.date_range, context)
            await tool_log(
                log=f"Retrieved {len(topic_texts)} texts for topics: {args.topics}.",
                context=context,
                associated_data=topic_texts,
            )
            texts.extend(topic_texts)
        except Exception as e:
            logger.exception(f"Failed to get texts for topics: {e}")

    # if universe_name is provided, find top contributers/performers and performance tables
    if args.universe_name:
        try:
            # get universe holdings table
            universe_stocks: Table = await get_universe_holdings(  # type: ignore
                GetUniverseHoldingsInput(universe_name=args.universe_name), context
            )
            tables.append(universe_stocks)
            # get universe performance in all levels
            for performance_level in UNIVERSE_PERFORMANCE_LEVELS:
                universe_performance_table: Table = await get_universe_performance(  # type: ignore
                    GetUniversePerformanceInput(
                        universe_name=args.universe_name,
                        date_range=args.date_range,
                        performance_level=performance_level,
                    ),
                    context,
                )
                # add universe performance table to tables
                tables.append(universe_performance_table)
                if performance_level == "security":
                    # this will be used to get top and bottom 3 contributers
                    uni_perf_df = universe_performance_table.to_df()
            # get top and bottom 3 contributers and performers
            top_contributers = (
                uni_perf_df.sort_values("weighted-return", ascending=False)
                .head(3)[STOCK_ID_COL_NAME_DEFAULT]
                .tolist()
            )
            bottom_contributers = (
                uni_perf_df.sort_values("weighted-return", ascending=True)
                .head(3)[STOCK_ID_COL_NAME_DEFAULT]
                .tolist()
            )
            top_performers = (
                uni_perf_df.sort_values("return", ascending=False)
                .head(3)[STOCK_ID_COL_NAME_DEFAULT]
                .tolist()
            )
            bottom_performers = (
                uni_perf_df.sort_values("return", ascending=True)
                .head(3)[STOCK_ID_COL_NAME_DEFAULT]
                .tolist()
            )

            # deduplicate top/bottom contributers and performers and stock_ids
            args.stock_ids = list(
                set(
                    top_contributers
                    + bottom_contributers
                    + top_performers
                    + bottom_performers
                    + args.stock_ids
                )
            )

            await tool_log(
                log=f"Retrieved top 3 and bottom 3 contributers and performers in {args.universe_name}.",
                context=context,
            )

        except Exception as e:
            logger.exception(
                f"Failed to get stocks/performances for universe {args.universe_name}: {e}"
            )

    # if portfolio_id is provided, get the portfolio top performers/contributers and performance tables
    if args.portfolio_id:
        try:
            # get portfolio/ benchamrk holdings tables - default and expanded
            for expand_etfs in [False, True]:
                portfolio_holdings_table: StockTable = await get_portfolio_holdings(  # type: ignore
                    GetPortfolioHoldingsInput(
                        portfolio_id=args.portfolio_id,
                        expand_etfs=expand_etfs,
                        fetch_default_stats=False,
                    ),
                    context,
                )
                tables.append(portfolio_holdings_table)
                benchmark_holdings_table: Table = await get_portfolio_benchmark_holdings(  # type: ignore
                    GetPortfolioBenchmarkHoldingsInput(
                        portfolio_id=args.portfolio_id, expand_etfs=expand_etfs
                    ),
                    context,
                )
                tables.append(benchmark_holdings_table)

            # get portfolio benchmark performance in all levels
            for performance_level in BENCHMARK_PERFORMANCE_LEVELS:
                portfolio_benchmark_perf_table: Table = await get_portfolio_benchmark_performance(  # type: ignore
                    GetPortfolioBenchmarkPerformanceInput(
                        portfolio_id=args.portfolio_id,
                        date_range=args.date_range,
                        performance_level=performance_level,
                    ),
                    context,
                )
                # add portfolio benchmark performance table to tables
                tables.append(portfolio_benchmark_perf_table)

            # get portfolio performance in all levels
            for performance_level in PORTFOLIO_PERFORMANCE_LEVELS:
                portfolio_performance_table: Table = await get_portfolio_performance(  # type: ignore
                    GetPortfolioPerformanceInput(
                        portfolio_id=args.portfolio_id,
                        date_range=args.date_range,
                        performance_level=performance_level,
                    ),
                    context,
                )
                # add portfolio performance table to tables
                tables.append(portfolio_performance_table)
                if performance_level == "stock":
                    # this will be used to get top and bottom 3 contributers
                    port_perf_df = portfolio_performance_table.to_df()
            # get top and bottom 3 contributers and performers
            top_contributers = (
                port_perf_df.sort_values("weighted-return", ascending=False)
                .head(3)[STOCK_ID_COL_NAME_DEFAULT]
                .tolist()
            )
            bottom_contributers = (
                port_perf_df.sort_values("weighted-return", ascending=True)
                .head(3)[STOCK_ID_COL_NAME_DEFAULT]
                .tolist()
            )
            top_performers = (
                port_perf_df.sort_values("return", ascending=False)
                .head(3)[STOCK_ID_COL_NAME_DEFAULT]
                .tolist()
            )
            bottom_performers = (
                port_perf_df.sort_values("return", ascending=True)
                .head(3)[STOCK_ID_COL_NAME_DEFAULT]
                .tolist()
            )

            # deduplicate top/bottom contributers and performers and stock_ids
            args.stock_ids = list(
                set(
                    top_contributers
                    + bottom_contributers
                    + top_performers
                    + bottom_performers
                    + args.stock_ids
                )
            )

            await tool_log(
                log="Retrieved top 3 and bottom 3 contributers and performers in portfolio.",
                context=context,
            )
        except Exception as e:
            logger.exception(
                f"Failed to get stocks/performances for portfolio {args.portfolio_id}: {e}"
            )

    # If stock_ids are provided, get the texts for the stock_ids
    if args.stock_ids:
        if len(args.stock_ids) > MAX_STOCKS_PER_COMMENTARY:
            await tool_log(
                log=(
                    f"Number of stocks is more than {MAX_STOCKS_PER_COMMENTARY}. "
                    f"Only first {MAX_STOCKS_PER_COMMENTARY} stocks will be considered."
                ),
                context=context,
            )
            args.stock_ids = args.stock_ids[:MAX_STOCKS_PER_COMMENTARY]
        try:
            # get stock performance tables
            stock_performance_table: Table = await get_stocks_performance(  # type: ignore
                GetStocksPerformanceInput(stock_ids=args.stock_ids, date_range=args.date_range),
                context,
            )
            tables.append(stock_performance_table)
            # get news developments and articles for stock ids
            stock_devs: List[StockNewsDevelopmentText] = (
                await get_all_news_developments_about_companies(  # type: ignore
                    GetNewsDevelopmentsAboutCompaniesInput(
                        stock_ids=args.stock_ids, date_range=args.date_range
                    ),
                    context,
                )
            )
            stock_news: List[Text] = await get_news_articles_for_stock_developments(  # type: ignore
                GetNewsArticlesForStockDevelopmentsInput(
                    developments_list=stock_devs,
                ),
                context,
            )
            stock_texts: List[Text] = stock_devs + stock_news
            await tool_log(
                log=f"Retrieved {len(stock_texts)} news texts for given stock ids.",
                context=context,
                associated_data=stock_texts,
            )
            texts.extend(stock_texts)
        except Exception as e:
            logger.exception(f"Failed to get texts for stock ids: {e}")

    return texts + tables
