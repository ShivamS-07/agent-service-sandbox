import json
import random
import re
from collections import defaultdict
from datetime import date
from typing import Any, Dict, List, Tuple

import pandas as pd
from data_access_layer.core.dao.securities import SecuritiesMetadataDAO

from agent_service.external.pa_backtest_svc_client import (
    get_stock_performance_for_date_range,
)
from agent_service.GPT.constants import (
    FILTER_CONCURRENCY,
    GPT4_O,
    MAX_TOKENS,
    NO_PROMPT,
)
from agent_service.GPT.requests import GPT
from agent_service.GPT.tokens import GPTTokenizer
from agent_service.io_types.dates import DateRange
from agent_service.io_types.stock import StockID
from agent_service.io_types.table import STOCK_ID_COL_NAME_DEFAULT, StockTable, Table
from agent_service.io_types.text import Text, TextGroup, ThemeText
from agent_service.tools.commentary.constants import (
    COMMENTARY_LLM,
    MAX_ARTICLES_PER_DEVELOPMENT,
    MAX_DEVELOPMENTS_PER_TOPIC,
    MAX_MATCHED_ARTICLES_PER_TOPIC,
)
from agent_service.tools.commentary.prompts import (
    COMMENTARY_PROMPT_MAIN,
    FILTER_CITATIONS_PROMPT,
    GEOGRAPHY_PROMPT,
    PORTFOLIO_PROMPT,
    STOCKS_STATS_PROMPT,
    SUMMARIZE_TEXT_PROMPT,
)
from agent_service.tools.news import (
    GetNewsArticlesForTopicsInput,
    get_news_articles_for_topics,
)
from agent_service.tools.portfolio import (
    GetPortfolioPerformanceInput,
    GetPortfolioWorkspaceHoldingsInput,
    PortfolioID,
    get_portfolio_holdings,
    get_portfolio_performance,
)
from agent_service.tools.tables import TableColumnMetadata, TableColumnType
from agent_service.tools.themes import (
    GetMacroeconomicThemeInput,
    GetThemeDevelopmentNewsArticlesInput,
    GetThemeDevelopmentNewsInput,
    get_macroeconomic_themes,
    get_news_articles_for_theme_developments,
    get_news_developments_about_theme,
)
from agent_service.tools.tool_log import tool_log
from agent_service.types import PlanRunContext
from agent_service.utils.async_utils import gather_with_concurrency
from agent_service.utils.postgres import get_psql
from agent_service.utils.prefect import get_prefect_logger
from agent_service.utils.prompt_utils import FilledPrompt
from agent_service.utils.string_utils import clean_to_json_if_needed

logger = get_prefect_logger(__name__)


# Helper functions
async def split_text_and_citation_ids(GPT_ouput: str) -> Tuple[str, List[int]]:
    lines = GPT_ouput.replace("\n\n", "\n").split("\n")
    citation_ids = json.loads(clean_to_json_if_needed(lines[-1]))
    main_text = "\n".join(lines[:-1])
    return main_text, citation_ids


async def get_sec_metadata_dao() -> SecuritiesMetadataDAO:
    return SecuritiesMetadataDAO(cache_sec_metadata=True)


async def get_theme_related_texts(
    themes_texts: List[ThemeText], date_range: DateRange, context: PlanRunContext
) -> List[Text]:
    """
    This function gets the theme related texts for the given themes.
    """
    # print("themes texts size", len(themes_texts))
    res: List[Any] = []
    development_texts: List[Any] = await get_news_developments_about_theme(  # type: ignore
        GetThemeDevelopmentNewsInput(
            themes=themes_texts,
            max_devs_per_theme=MAX_DEVELOPMENTS_PER_TOPIC,
            date_range=date_range,
        ),
        context,
    )
    # print("development texts size", len(development_texts))
    article_texts: List[Any] = await get_news_articles_for_theme_developments(  # type: ignore
        GetThemeDevelopmentNewsArticlesInput(
            developments_list=development_texts,
            max_articles_per_development=MAX_ARTICLES_PER_DEVELOPMENT,
        ),
        context,
    )
    # print("article texts size", len(article_texts))
    res.extend(development_texts)
    res.extend(article_texts)
    return res


async def organize_commentary_texts(texts: List[Text]) -> Dict[str, List[Text]]:
    """
    This function organizes the commentary texts into a dictionary with the text descriptions
    as the key.
    """
    res = defaultdict(list)
    for text in texts:
        res[text.text_type].append(text)
    # shuffle the texts order in each key so when removing the texts, the order is random
    for key in res:
        random.shuffle(res[key])
    return res


async def get_portfolio_geography_str(regions_to_weight: List[Tuple[str, float]]) -> str:
    # convert weights to int percentages
    portfolio_geography = "\n".join(
        [f"{tup[0]}: {int(tup[1] * 100)}%" for tup in regions_to_weight]
    )
    return portfolio_geography


async def get_region_weights_from_portfolio_holdings(
    portfolio_holdings_df: pd.DataFrame,
) -> List[Tuple[str, float]]:
    """
    Given a mapping from GBI ID to a weight, return a list of ranked (region,
    weight) tuples sorted in descending order by weight.
    """
    # convert DF to dict[int, float]
    weighted_holdings = {}
    for i in range(len(portfolio_holdings_df[STOCK_ID_COL_NAME_DEFAULT])):
        gbi_id = portfolio_holdings_df[STOCK_ID_COL_NAME_DEFAULT][i].gbi_id
        weight = portfolio_holdings_df["Weight"][i]
        weighted_holdings[gbi_id] = weight

    dao = await get_sec_metadata_dao()
    sec_meta_map = dao.get_security_metadata(list(weighted_holdings.keys())).get()
    region_weight_map: Dict[str, float] = defaultdict(float)
    for meta in sec_meta_map.values():
        region_weight_map[meta.country] += weighted_holdings[meta.gbi_id]

    output = list(region_weight_map.items())
    return sorted(output, reverse=True, key=lambda tup: tup[1])


async def get_previous_commentary_results(context: PlanRunContext) -> List[Text]:
    """
    This function gets the previous commentary results for the given agent context.
    """
    db = get_psql(skip_commit=False)
    # get all plans for the agent
    plans, _, plan_ids = db.get_all_execution_plans(context.agent_id)

    # find the plan_ids that have used commentary tool
    plan_ids_with_commentary = []
    task_ids_with_commentary = []
    for plan, plan_id in zip(plans, plan_ids):
        for tool in plan.nodes:
            if tool.tool_name == "write_commentary":
                plan_ids_with_commentary.append(plan_id)
                task_ids_with_commentary.append(tool.tool_task_id)

    # get the previous commentary results
    previous_commentary_results = []
    for plan_id, task_id in zip(plan_ids_with_commentary, task_ids_with_commentary):
        # get the last tool output of the latest plan run
        res = db.get_last_tool_output_for_plan(context.agent_id, plan_id, task_id)

        # if the output is a Text object, add it to the list
        # this should ne adjusted when output is other stuff
        if isinstance(res, Text):
            previous_commentary_results.append(res)
    return previous_commentary_results


async def match_daterange_to_timedelta(date_range: DateRange) -> str:
    # Define the list of predefined intervals in terms of days
    INTERVALS = {
        "1W": 7,
        "1M": 30,  # Approximation for 1 month as 30 days
        "3M": 90,  # Approximation for 3 months as 90 days
        "6M": 180,  # Approximation for 6 months as 180 days
        "9M": 270,  # Approximation for 9 months as 270 days
        "1Y": 365,  # Approximation for 1 year as 365 days
    }
    # Calculate the duration of the date range in days
    duration = (date_range.end_date - date.today()).days

    # Find the closest match based on days
    closest_match = min(INTERVALS, key=lambda x: abs(INTERVALS[x] - duration))

    return closest_match


async def filter_most_important_citations(
    citations: List[int], texts: str, commentary_result: str
) -> List[int]:
    """this function filters the most important citations from the given list of citations, texts and
    commentary result.
    """
    llm = GPT(model=GPT4_O)

    result = await llm.do_chat_w_sys_prompt(
        main_prompt=FILTER_CITATIONS_PROMPT.format(
            texts=texts, citations=citations, commentary_result=commentary_result
        ),
        sys_prompt=NO_PROMPT,
    )
    cleaned_result = re.sub(r"[^\d,]", "", result)
    filtered_citations = list(map(int, cleaned_result.strip("[]").split(",")))
    return filtered_citations


async def get_texts_for_topics(
    topics: List[str], date_range: DateRange, context: PlanRunContext
) -> List[Text]:
    """
    This function gets the texts for the given topics. If the themes are found, it gets the related texts.
    If the themes are not found, it gets the articles related to the topic.
    """

    texts: List = []
    topics = topics if topics else []
    for topic in topics:
        try:
            themes = await get_macroeconomic_themes(
                GetMacroeconomicThemeInput(theme_refs=[topic]), context
            )
            await tool_log(
                log=f"Found {len(themes)} themes for topic: {topic}.",  # type: ignore
                context=context,
                associated_date=themes,
            )
            res = await get_theme_related_texts(themes, context)  # type: ignore
            await tool_log(
                log=f"Found {len(res)} theme-related texts for topic: {topic}.",
                context=context,
                associated_date=res,
            )
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
                        date_range=date_range,
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


async def prepare_portfolio_prompt(
    portfolio_id: PortfolioID, date_range: DateRange, context: PlanRunContext
) -> FilledPrompt:
    """
    This function prepares the portfolio prompt for the commentary.
    """
    portfolio_holdings_table: StockTable = await get_portfolio_holdings(  # type: ignore
        GetPortfolioWorkspaceHoldingsInput(portfolio_id=portfolio_id), context
    )
    portfolio_holdings_df = portfolio_holdings_table.to_df()

    # Get the region weights from the portfolio holdings
    regions_to_weight = await get_region_weights_from_portfolio_holdings(portfolio_holdings_df)
    portfolio_geography = await get_portfolio_geography_str(regions_to_weight)
    # Prepare the geography prompt
    await tool_log(
        log="Retrieved region weights from the portfolio holdings.",
        context=context,
        associated_data=regions_to_weight,
    )
    portfolio_geography_prompt = GEOGRAPHY_PROMPT.format(portfolio_geography=portfolio_geography)

    # get the overall portfolio performance
    overall_performance_table: Table = await get_portfolio_performance(  # type: ignore
        GetPortfolioPerformanceInput(
            portfolio_id=portfolio_id,
            performance_level="overall",
        ),
        context,
    )
    await tool_log(
        log="Retrieved overall portfolio performance.",
        context=context,
        associated_data=overall_performance_table,
    )
    # get the sector level portfolio performance
    sector_performance_horizon = (
        await match_daterange_to_timedelta(date_range) if date_range else "1M"
    )
    sector_performance_table: Table = await get_portfolio_performance(  # type: ignore
        GetPortfolioPerformanceInput(
            portfolio_id=portfolio_id,
            performance_level="sector",
            sector_performance_horizon=sector_performance_horizon,
        ),
        context,
    )
    await tool_log(
        log=(
            "Retrieved sector level portfolio performance."
            "for duration {sector_performance_horizon}."
        ),
        context=context,
        associated_data=sector_performance_table,
    )
    # get stock level portfolio performance
    stock_performance_table: Table = await get_portfolio_performance(  # type: ignore
        GetPortfolioPerformanceInput(
            portfolio_id=portfolio_id,
            performance_level="stock",
            date_range=date_range,
        ),
        context,
    )
    await tool_log(
        log="Retrieved stock level portfolio performance.",
        context=context,
        associated_data=stock_performance_table,
    )

    # Prepare the portfolio prompt
    portfolio_prompt = PORTFOLIO_PROMPT.format(
        portfolio_holdings=str(portfolio_holdings_df),
        portfolio_geography_prompt=portfolio_geography_prompt.filled_prompt,
        portfolio_performance_overall=str(overall_performance_table.to_df()),
        portfolio_performance_by_sector=str(sector_performance_table.to_df()),
        portfolio_performance_by_stock=str(stock_performance_table.to_df()),
    )
    return portfolio_prompt


async def prepare_stocks_stats_prompt(
    stock_ids: List[StockID], date_range: DateRange, context: PlanRunContext
) -> FilledPrompt:
    """
    This function prepares the stock stats prompt for the commentary.

    """

    gbi_ids = [stock.gbi_id for stock in stock_ids]
    stock_performance = await get_stock_performance_for_date_range(
        gbi_ids=gbi_ids,
        start_date=date_range.start_date,
        end_date=date_range.end_date,
        user_id=context.user_id,
    )
    # Create a DataFrame/Table for the stock performance
    stock_performance_df = pd.DataFrame(
        {
            STOCK_ID_COL_NAME_DEFAULT: await StockID.from_gbi_id_list(gbi_ids),
            "return": [stock.performance for stock in stock_performance.stock_performance_list],
        }
    )
    stock_performance_table = StockTable.from_df_and_cols(
        data=stock_performance_df,
        columns=[
            TableColumnMetadata(label=STOCK_ID_COL_NAME_DEFAULT, col_type=TableColumnType.STOCK),
            TableColumnMetadata(label="return", col_type=TableColumnType.FLOAT),
        ],
    )
    await tool_log(
        log=f"Retrieved performances for the stock ids for {date_range.start_date} to {date_range.end_date}.",
        context=context,
        associated_data=stock_performance_table,
    )

    stocks_stats_prompt = STOCKS_STATS_PROMPT.format(
        stock_stats=str(stock_performance_df),
    )
    return stocks_stats_prompt


async def summarize_text_mapping(text_mapping: Dict[str, List[Text]]) -> Dict[str, List[Text]]:
    """
    This function summarizes some texts in the text mapping.
    """
    llm = GPT(model=GPT4_O)
    for text_type in text_mapping:
        if text_type in ("SEC filing", "Company Description"):
            tasks = []
            texts_str_list = await Text.get_all_strs(text_mapping[text_type], include_header=True)
            for text in texts_str_list:
                tasks.append(
                    llm.do_chat_w_sys_prompt(
                        SUMMARIZE_TEXT_PROMPT.format(text=text),
                        sys_prompt=NO_PROMPT,
                    )
                )
            results = await gather_with_concurrency(tasks, n=FILTER_CONCURRENCY)
            for i, result in enumerate(results):
                text_mapping[text_type][i] = Text(
                    val=result,
                )

    return text_mapping


async def prepare_main_prompt(
    previous_commentary_prompt: FilledPrompt,
    portfolio_prompt: FilledPrompt,
    stocks_stats_prompt: FilledPrompt,
    watchlist_prompt: FilledPrompt,
    client_type_prompt: FilledPrompt,
    writing_style_prompt: FilledPrompt,
    texts: str,
    text_mapping: Dict[str, List[Text]],
    context: PlanRunContext,
) -> FilledPrompt:
    """
    This function prepares the main prompt for the commentary.
    """
    chat_context = context.chat.get_gpt_input() if context.chat is not None else ""

    # show the length of tokens in text_mapping
    for text_type, text_list in text_mapping.items():
        text_list_str = str(await Text.get_all_strs(TextGroup(val=text_list), include_header=True))
        logger.info(
            f"Length of tokens in {text_type}: {GPTTokenizer(COMMENTARY_LLM).get_token_length(text_list_str)}"
        )

    main_prompt = COMMENTARY_PROMPT_MAIN.format(
        previous_commentary_prompt=previous_commentary_prompt.filled_prompt,
        portfolio_prompt=portfolio_prompt.filled_prompt,
        stocks_stats_prompt=stocks_stats_prompt.filled_prompt,
        watchlist_prompt=watchlist_prompt.filled_prompt,
        client_type_prompt=client_type_prompt.filled_prompt,
        writing_style_prompt=writing_style_prompt.filled_prompt,
        texts=texts,
        chat_context=chat_context,
    )

    main_prompt_token_length = GPTTokenizer(COMMENTARY_LLM).get_token_length(
        main_prompt.filled_prompt
    )
    logger.info(f"Length of tokens in main prompt: {main_prompt_token_length}")
    texts_token_length = GPTTokenizer(COMMENTARY_LLM).get_token_length(texts)
    logger.info(f"Length of tokens in texts: {texts_token_length}")
    # if main prompt is too long, summerize some texts
    if main_prompt_token_length > MAX_TOKENS[COMMENTARY_LLM]:

        text_mapping_summarized = await summarize_text_mapping(text_mapping)
        # Prepare texts for commentary
        all_text_group = TextGroup(
            val=[text for text_list in text_mapping_summarized.values() for text in text_list]
        )
        summarized_texts = await Text.get_all_strs(
            all_text_group, include_header=True, text_group_numbering=True
        )
        # truncate texts if needed
        summarized_texts_token_length = GPTTokenizer(COMMENTARY_LLM).get_token_length(
            str(summarized_texts)
        )
        logger.info(f"Length of tokens in summerzied texts: {summarized_texts_token_length}")
        trunctated_texts = GPTTokenizer(COMMENTARY_LLM).do_truncation_if_needed(
            str(summarized_texts),
            [
                previous_commentary_prompt.filled_prompt,
                portfolio_prompt.filled_prompt,
                stocks_stats_prompt.filled_prompt,
                watchlist_prompt.filled_prompt,
                client_type_prompt.filled_prompt,
                writing_style_prompt.filled_prompt,
                chat_context,
            ],
        )
        main_prompt = COMMENTARY_PROMPT_MAIN.format(
            previous_commentary_prompt=previous_commentary_prompt.filled_prompt,
            portfolio_prompt=portfolio_prompt.filled_prompt,
            stocks_stats_prompt=stocks_stats_prompt.filled_prompt,
            watchlist_prompt=watchlist_prompt.filled_prompt,
            client_type_prompt=client_type_prompt.filled_prompt,
            writing_style_prompt=writing_style_prompt.filled_prompt,
            texts=trunctated_texts,
            chat_context=chat_context,
        )
        main_prompt_token_length = GPTTokenizer(COMMENTARY_LLM).get_token_length(
            main_prompt.filled_prompt
        )
        logger.info(f"Length of tokens in main prompt (after): {main_prompt_token_length}")
        # show the length of tokens in text_mapping
        for text_type, text_list in text_mapping.items():
            text_list_str: str = await Text.get_all_strs(TextGroup(val=text_list), include_header=True)  # type: ignore
            logger.info(
                f"Length of tokens in {text_type}: {GPTTokenizer(COMMENTARY_LLM).get_token_length(text_list_str)}"
            )
    return main_prompt
