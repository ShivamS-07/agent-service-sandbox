import datetime
import json
import random
import re
from collections import defaultdict
from datetime import date
from typing import Dict, List, Optional, Tuple

import pandas as pd
from data_access_layer.core.dao.securities import SecuritiesMetadataDAO

from agent_service.external.pa_backtest_svc_client import (
    get_stock_performance_for_date_range,
)
from agent_service.GPT.constants import GPT4_O, NO_PROMPT
from agent_service.GPT.requests import GPT
from agent_service.io_types.dates import DateRange
from agent_service.io_types.stock import StockID
from agent_service.io_types.table import (
    STOCK_ID_COL_NAME_DEFAULT,
    StockTable,
    TableColumnMetadata,
    TableColumnType,
)
from agent_service.io_types.text import Text, ThemeText
from agent_service.tools.commentary.constants import (
    MAX_ARTICLES_PER_DEVELOPMENT,
    MAX_DEVELOPMENTS_PER_TOPIC,
    MAX_MATCHED_ARTICLES_PER_TOPIC,
)
from agent_service.tools.commentary.prompts import (
    FILTER_CITATIONS_PROMPT,
    GEOGRAPHY_PROMPT,
    PORTFOLIO_PROMPT,
    STOCK_PERFORMANCE_PROMPT,
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
from agent_service.utils.postgres import get_psql
from agent_service.utils.prefect import get_prefect_logger
from agent_service.utils.prompt_utils import FilledPrompt
from agent_service.utils.string_utils import clean_to_json_if_needed


# Helper functions
async def split_text_and_citation_ids(GPT_ouput: str) -> Tuple[str, List[int]]:
    lines = GPT_ouput.replace("\n\n", "\n").split("\n")
    citation_ids = json.loads(clean_to_json_if_needed(lines[-1]))
    main_text = "\n".join(lines[:-1])
    return main_text, citation_ids


async def get_sec_metadata_dao() -> SecuritiesMetadataDAO:
    return SecuritiesMetadataDAO(cache_sec_metadata=True)


async def get_theme_related_texts(
    themes_texts: List[ThemeText], context: PlanRunContext
) -> List[Text]:
    """
    This function gets the theme related texts for the given themes.
    """
    # print("themes texts size", len(themes_texts))
    res: List = []
    development_texts = await get_news_developments_about_theme(
        GetThemeDevelopmentNewsInput(
            themes=themes_texts, max_devs_per_theme=MAX_DEVELOPMENTS_PER_TOPIC
        ),
        context,
    )
    # print("development texts size", len(development_texts))
    article_texts = await get_news_articles_for_theme_developments(  # type: ignore
        GetThemeDevelopmentNewsArticlesInput(
            developments_list=development_texts,  # type: ignore
            max_articles_per_development=MAX_ARTICLES_PER_DEVELOPMENT,
        ),
        context,  # type: ignore
    )
    # print("article texts size", len(article_texts))
    res.extend(development_texts)  # type: ignore
    res.extend(article_texts)  # type: ignore
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
    topics: List[str], date_range: Optional[DateRange], context: PlanRunContext
) -> List[Text]:
    """
    This function gets the texts for the given topics. If the themes are found, it gets the related texts.
    If the themes are not found, it gets the articles related to the topic.
    """
    logger = get_prefect_logger(__name__)

    texts: List = []
    topics = topics if topics else []
    for topic in topics:
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
    portfolio_id: PortfolioID, date_range: Optional[DateRange], context: PlanRunContext
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
    overall_performance_table = await get_portfolio_performance(
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
    sector_performance_table = await get_portfolio_performance(
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
    stock_performance_table = await get_portfolio_performance(
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
        portfolio_performance_overall=str(overall_performance_table.to_df()),  # type: ignore
        portfolio_performance_by_sector=str(sector_performance_table.to_df()),  # type: ignore
        portfolio_performance_by_stock=str(stock_performance_table.to_df()),  # type: ignore
    )
    return portfolio_prompt


async def prepare_stock_performance_prompt(
    stock_ids: List[StockID], date_range: Optional[DateRange], context: PlanRunContext
) -> FilledPrompt:
    """
    This function prepares the stock performance prompt for the commentary.
    """
    # get the stock performance for the date range
    gbi_ids = [stock.gbi_id for stock in stock_ids]
    if date_range is None:
        date_range = DateRange(
            start_date=date.today() - datetime.timedelta(days=30),
            end_date=date.today(),
        )
    else:
        date_range = date_range
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
        log=f"Retrieved stock performance for the given stock ids for date: {date_range}.",
        context=context,
        associated_data=stock_performance_table,
    )
    stock_performance_prompt = STOCK_PERFORMANCE_PROMPT.format(
        stock_performance=str(stock_performance_df)
    )
    return stock_performance_prompt
