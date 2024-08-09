import json
import random
import re
from collections import defaultdict
from datetime import date
from typing import Any, Dict, List, Tuple

import pandas as pd
from data_access_layer.core.dao.securities import SecuritiesMetadataDAO
from feature_service_proto_v1.feature_metadata_service_pb2 import (
    GetAllFeaturesMetadataResponse,
)
from gbi_common_py_utils.numpy_common import NumpySheet

from agent_service.external.feature_svc_client import (
    get_all_features_metadata,
    get_feature_data,
)
from agent_service.external.pa_backtest_svc_client import (
    get_stock_performance_for_date_range,
)
from agent_service.GPT.constants import (
    FILTER_CONCURRENCY,
    GPT4_O,
    GPT4_O_MINI,
    MAX_TOKENS,
    NO_PROMPT,
)
from agent_service.GPT.requests import GPT
from agent_service.GPT.tokens import GPTTokenizer
from agent_service.io_types.dates import DateRange
from agent_service.io_types.stock import StockID
from agent_service.io_types.table import STOCK_ID_COL_NAME_DEFAULT, StockTable, Table
from agent_service.io_types.text import StatisticsText, Text, TextGroup, ThemeText
from agent_service.tools.commentary.constants import (
    COMMENTARY_LLM,
    MAX_ARTICLES_PER_DEVELOPMENT,
    MAX_DEVELOPMENTS_PER_TOPIC,
    MAX_MATCHED_ARTICLES_PER_TOPIC,
)
from agent_service.tools.commentary.prompts import (
    CHOOSE_STATISTICS_PROMPT,
    COMMENTARY_PROMPT_MAIN,
    COMMENTARY_SYS_PROMPT,
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
    GetPortfolioBenchmarkHoldingsInput,
    GetPortfolioHoldingsInput,
    GetPortfolioPerformanceInput,
    PortfolioID,
    get_portfolio_benchmark_holdings,
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
from agent_service.utils.gpt_logging import GptJobIdType, GptJobType, create_gpt_context
from agent_service.utils.postgres import get_psql
from agent_service.utils.prefect import get_prefect_logger
from agent_service.utils.prompt_utils import FilledPrompt
from agent_service.utils.string_utils import clean_to_json_if_needed

logger = get_prefect_logger(__name__)
PERFORMANCE_LEVELS = ["sector", "security", "stock", "overall", "monthly", "daily"]


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
    statistics_texts = await get_statistics_for_theme(
        context=context, texts=themes_texts, date_range=date_range
    )
    res.extend(development_texts)  # type: ignore
    res.extend(article_texts)  # type: ignore
    res.extend(statistics_texts)  # type: ignore
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
    portfolio_geography = "\n".join([f"{tup[0]}: {int(tup[1])}%" for tup in regions_to_weight])
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


async def get_statistics_for_theme(
    context: PlanRunContext, texts: List[ThemeText], date_range: DateRange
) -> List[StatisticsText]:
    # get all the features
    resp = await get_all_features_metadata(context.user_id, filtered=True)
    all_statistic_lookup: Dict[str, GetAllFeaturesMetadataResponse.FeatureMetadata] = {
        feature_metadata.feature_id: feature_metadata
        for feature_metadata in resp.features
        if feature_metadata.importance <= 2
    }
    name_to_id_map = {
        feature_metadata.name: feature_metadata.feature_id
        for feature_metadata in resp.features
        if feature_metadata.importance <= 2
    }
    all_statistics_str = "\n".join(
        f"Statistic name {metadata.name}"
        for name, metadata in all_statistic_lookup.items()
        if metadata.is_global  # for now, we filter out all the non global features
    )
    # Create GPT context and llm model
    gpt_context = create_gpt_context(
        GptJobType.AGENT_TOOLS, context.agent_id, GptJobIdType.AGENT_ID
    )
    llm = GPT(model=GPT4_O, context=gpt_context)

    result = await llm.do_chat_w_sys_prompt(
        main_prompt=CHOOSE_STATISTICS_PROMPT.format(
            statistic_list=all_statistics_str, theme_info="\n".join([text.val for text in texts])
        ),
        sys_prompt=NO_PROMPT,
    )
    json_obj: Dict[str, List[str]] = json.loads(clean_to_json_if_needed(result))
    statistics_list = json_obj.get("statistics", [])
    feature_ids = []
    for feature_name in statistics_list:
        feature_id = name_to_id_map.get(feature_name, None)
        if feature_id:
            feature_ids.append(feature_id)
    # Using the chosen stats that have been selected create the prompt
    all_stats_texts: List[StatisticsText] = []
    if len(feature_ids) == 0:
        return all_stats_texts
    data = await get_feature_data(
        statistic_ids=feature_ids,
        stock_ids=[],
        from_date=date_range.start_date,
        to_date=date_range.end_date,
        user_id=context.user_id,
    )
    global_data = data.global_data
    np_sheet = None
    for data in global_data:
        np_sheet = NumpySheet.initialize_from_proto_bytes(data.data_sheet, cols_are_dates=False)
    if np_sheet:
        index = [
            all_statistic_lookup.get(feature_id, feature_id).name for feature_id in np_sheet.rows
        ]
        df = pd.DataFrame(np_sheet.np_data, index=index, columns=np_sheet.columns)
        feature_dict = df.to_dict(orient="index")
        all_stats_texts = [
            StatisticsText(id=str(feature), val=json.dumps(feature_data))
            for feature, feature_data in feature_dict.items()
        ]
    return all_stats_texts


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
            themes: List[ThemeText] = await get_macroeconomic_themes(  # type: ignore
                GetMacroeconomicThemeInput(theme_refs=[topic]), context
            )
            await tool_log(
                log=f"Found {len(themes)} themes for topic: {topic}.",
                context=context,
                associated_data=themes,
            )
            res = await get_theme_related_texts(
                themes_texts=themes, date_range=date_range, context=context
            )
            await tool_log(
                log=f"Found {len(res)} theme-related texts for topic: {topic}.",
                context=context,
                associated_data=res,
            )
            texts.extend(res + themes)

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
        GetPortfolioHoldingsInput(
            portfolio_id=portfolio_id,
            fetch_stats=False,
        ),
        context,
    )
    portfolio_holdings_df = portfolio_holdings_table.to_df()

    portfolio_holdings_expanded_table: StockTable = await get_portfolio_holdings(  # type: ignore
        GetPortfolioHoldingsInput(
            portfolio_id=portfolio_id,
            expand_etfs=True,
            fetch_stats=False,
        ),
        context,
    )
    portfolio_holdings_expanded_df = portfolio_holdings_expanded_table.to_df()

    # Get the region weights from the portfolio holdings
    regions_to_weight = await get_region_weights_from_portfolio_holdings(
        portfolio_holdings_expanded_df
    )
    portfolio_geography = await get_portfolio_geography_str(regions_to_weight)
    # Prepare the geography prompt
    await tool_log(
        log="Retrieved region weights from the portfolio holdings.",
        context=context,
        associated_data=regions_to_weight,
    )
    portfolio_geography_prompt = GEOGRAPHY_PROMPT.format(portfolio_geography=portfolio_geography)
    performance_dict: Dict[str, pd.DataFrame] = {}
    # get the sector level portfolio performance
    sector_performance_horizon = (
        await match_daterange_to_timedelta(date_range) if date_range else "1M"
    )
    # get the portfolio performance on levels in PERFORMANCE_LEVELS
    # TODO: This can be parallelized
    for performance_level in PERFORMANCE_LEVELS:
        performance_table: Table = await get_portfolio_performance(  # type: ignore
            GetPortfolioPerformanceInput(
                portfolio_id=portfolio_id,
                performance_level=performance_level,
                sector_performance_horizon=sector_performance_horizon,
                date_range=date_range,
            ),
            context,
        )
        performance_dict[f"portfolio_{performance_level}"] = performance_table.to_df()
        await tool_log(
            log=f"Retrieved {performance_level} portfolio performance.",
            context=context,
            associated_data=performance_dict[f"portfolio_{performance_level}"],
        )

    # get benchmark performance on stock level
    benchmark_holdings = await get_portfolio_benchmark_holdings(
        GetPortfolioBenchmarkHoldingsInput(portfolio_id=portfolio_id, expand_etfs=True), context
    )
    benchmark_holdings_df = benchmark_holdings.to_df()  # type: ignore
    gbi_ids = [stock.gbi_id for stock in benchmark_holdings_df[STOCK_ID_COL_NAME_DEFAULT]]
    benchmark_stock_performance = await get_stock_performance_for_date_range(
        gbi_ids=gbi_ids,
        start_date=date_range.start_date,
        user_id=context.user_id,
    )
    data = {
        STOCK_ID_COL_NAME_DEFAULT: await StockID.from_gbi_id_list(gbi_ids),
        "return": [
            stock.performance for stock in benchmark_stock_performance.stock_performance_list
        ],
        "benchmark-weight": benchmark_holdings_df["Weight"].values,
    }
    benchmark_stock_performance_df = pd.DataFrame(data)
    benchmark_stock_performance_df["return"] = benchmark_stock_performance_df["return"] * 100
    benchmark_stock_performance_df["weighted-return"] = (
        benchmark_stock_performance_df["return"]
        * benchmark_stock_performance_df["benchmark-weight"]
        / 100
    ).values
    benchmark_stock_performance_df = benchmark_stock_performance_df.sort_values(
        by="weighted-return", ascending=False
    )

    performance_dict["benchmark_stock"] = benchmark_stock_performance_df

    # extract top contributing stocks
    performance_dict["portfolio_stock_pos"] = performance_dict["portfolio_stock"].head(5)
    performance_dict["portfolio_stock_neg"] = performance_dict["portfolio_stock"].tail(5)
    performance_dict["benchmark_stock_pos"] = performance_dict["benchmark_stock"].head(5)
    performance_dict["benchmark_stock_neg"] = performance_dict["benchmark_stock"].tail(5)

    # Prepare the portfolio prompt
    portfolio_prompt = PORTFOLIO_PROMPT.format(
        portfolio_holdings=str(portfolio_holdings_df),
        portfolio_geography_prompt=portfolio_geography_prompt.filled_prompt,
        portfolio_performance_by_overall=str(performance_dict["portfolio_overall"]),
        portfolio_performance_by_monthly=str(performance_dict["portfolio_monthly"]),
        portfolio_performance_by_daily=str(performance_dict["portfolio_daily"]),
        portfolio_performance_by_sector=str(performance_dict["portfolio_sector"]),
        portfolio_performance_by_security=str(performance_dict["portfolio_security"]),
        portfolio_performance_by_stock_positive=str(performance_dict["portfolio_stock_pos"]),
        portfolio_performance_by_stock_negative=str(performance_dict["portfolio_stock_neg"]),
        benchmark_performance_by_stock_positive=str(performance_dict["benchmark_stock_pos"]),
        benchmark_performance_by_stock_negative=str(performance_dict["benchmark_stock_neg"]),
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


async def summarize_text_mapping(
    text_mapping: Dict[str, List[Text]], agent_id: str
) -> Dict[str, List[Text]]:
    """
    This function summarizes some texts in the text mapping.
    """
    gpt_context = create_gpt_context(GptJobType.AGENT_TOOLS, agent_id, GptJobIdType.AGENT_ID)
    llm = GPT(context=gpt_context, model=GPT4_O_MINI)
    for text_type in text_mapping:
        if text_type in ("SEC filing", "Company Description", "Earnings Call Summary"):
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
        today=str(date.today().strftime("%Y-%m-%d")),
    )

    main_prompt_token_length = GPTTokenizer(COMMENTARY_LLM).get_token_length(
        main_prompt.filled_prompt
    )
    logger.info(f"Length of tokens in main prompt: {main_prompt_token_length}")
    # if main prompt is too long, summerize some texts
    if main_prompt_token_length > MAX_TOKENS[COMMENTARY_LLM]:
        texts = GPTTokenizer(COMMENTARY_LLM).do_truncation_if_needed(
            texts,
            [
                COMMENTARY_PROMPT_MAIN.template,
                COMMENTARY_SYS_PROMPT.template,
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
            texts=texts,
            chat_context=chat_context,
            today=str(date.today().strftime("%Y-%m-%d")),
        )
        main_prompt_token_length = GPTTokenizer(COMMENTARY_LLM).get_token_length(
            main_prompt.filled_prompt
        )
        logger.info(
            f"Length of tokens in main prompt (after truncations): {main_prompt_token_length}"
        )
        # show the length of tokens in text_mapping
        for text_type, text_list in text_mapping.items():
            text_list_str: str = await Text.get_all_strs(TextGroup(val=text_list), include_header=True)  # type: ignore
            logger.info(
                f"Length of tokens in {text_type}: {GPTTokenizer(COMMENTARY_LLM).get_token_length(text_list_str)}"
            )
    return main_prompt
