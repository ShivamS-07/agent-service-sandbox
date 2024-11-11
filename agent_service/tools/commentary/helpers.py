import json
import random
from collections import defaultdict
from datetime import date
from typing import TYPE_CHECKING, Any, Dict, List, Union

import pandas as pd
from feature_service_proto_v1.feature_metadata_service_pb2 import (
    GetAllFeaturesMetadataResponse,
)
from gbi_common_py_utils.numpy_common import NumpySheet

from agent_service.external.feature_svc_client import (
    get_all_variables_metadata,
    get_feature_data,
)
from agent_service.GPT.constants import GPT4_O, MAX_TOKENS, NO_PROMPT
from agent_service.GPT.requests import GPT
from agent_service.GPT.tokens import GPTTokenizer
from agent_service.io_types.dates import DateRange
from agent_service.io_types.stock import StockID
from agent_service.io_types.table import STOCK_ID_COL_NAME_DEFAULT, Table
from agent_service.io_types.text import StatisticsText, Text, TextGroup, ThemeText
from agent_service.tools.commentary.constants import (
    COMMENTARY_LLM,
    MAX_ARTICLES_PER_DEVELOPMENT,
    MAX_DEVELOPMENTS_PER_TOPIC,
    MAX_MATCHED_ARTICLES_PER_TOPIC,
)
from agent_service.tools.commentary.prompts import CHOOSE_STATISTICS_PROMPT
from agent_service.tools.news import (
    GetNewsAndWebPagesForTopicsInput,
    get_news_and_web_pages_for_topics,
)
from agent_service.tools.portfolio import (
    BENCHMARK_HOLDING_TABLE_NAME_EXPANDED,
    BENCHMARK_HOLDING_TABLE_NAME_NOT_EXPANDED,
    BENCHMARK_PERFORMANCE_LEVELS,
    BENCHMARK_PERFORMANCE_TABLE_BASE_NAME,
    PORTFOLIO_HOLDING_TABLE_NAME_EXPANDED,
    PORTFOLIO_HOLDING_TABLE_NAME_NOT_EXPANDED,
    PORTFOLIO_PERFORMANCE_LEVELS,
    PORTFOLIO_PERFORMANCE_TABLE_BASE_NAME,
)
from agent_service.tools.stocks import get_metadata_for_stocks
from agent_service.tools.themes import (
    GetMacroeconomicThemeInput,
    GetThemeDevelopmentNewsArticlesInput,
    GetThemeDevelopmentNewsInput,
    get_macroeconomic_themes,
    get_news_articles_for_theme_developments,
    get_news_developments_about_theme,
)
from agent_service.tools.tool_log import tool_log
from agent_service.tools.universe import (
    UNIVERSE_HOLDINGS_TABLE_NAME,
    UNIVERSE_PERFORMANCE_LEVELS,
    UNIVERSE_PERFORMANCE_TABLE_BASE_NAME,
)
from agent_service.types import PlanRunContext
from agent_service.utils.gpt_logging import GptJobIdType, GptJobType, create_gpt_context
from agent_service.utils.postgres import get_psql
from agent_service.utils.prefect import get_prefect_logger
from agent_service.utils.prompt_utils import FilledPrompt, Prompt
from agent_service.utils.string_utils import clean_to_json_if_needed

logger = get_prefect_logger(__name__)

if TYPE_CHECKING:
    from data_access_layer.core.dao.securities import SecuritiesMetadataDAO


async def get_sec_metadata_dao() -> "SecuritiesMetadataDAO":
    from data_access_layer.core.dao.securities import SecuritiesMetadataDAO

    return SecuritiesMetadataDAO(cache_sec_metadata=True)


async def get_theme_related_texts(
    themes_texts: List[ThemeText], date_range: DateRange, context: PlanRunContext
) -> List[Text]:
    """
    This function gets the theme related texts for the given themes.
    """
    res: List[Any] = []
    development_texts: List[Any] = await get_news_developments_about_theme(  # type: ignore
        GetThemeDevelopmentNewsInput(
            themes=themes_texts,
            max_devs_per_theme=MAX_DEVELOPMENTS_PER_TOPIC,
            date_range=date_range,
        ),
        context,
    )
    logger.info(f"Found {len(development_texts)} developments for themes.")
    article_texts: List[Any] = await get_news_articles_for_theme_developments(  # type: ignore
        GetThemeDevelopmentNewsArticlesInput(
            developments_list=development_texts,
            max_articles_per_development=MAX_ARTICLES_PER_DEVELOPMENT,
        ),
        context,
    )
    logger.info(f"Found {len(article_texts)} articles for developments.")
    statistics_texts = await get_statistics_for_theme(
        context=context, texts=themes_texts, date_range=date_range
    )
    logger.info(f"Found {len(statistics_texts)} statistics for themes.")
    res.extend(development_texts)  # type: ignore
    res.extend(article_texts)  # type: ignore
    res.extend(statistics_texts)  # type: ignore
    await tool_log(
        log=f"Retrieved {len(res)} theme related texts for top market trends.",
        context=context,
        associated_data=res,
    )
    return res


async def organize_commentary_inputs(
    inputs: List[Union[Text, Table]]
) -> Dict[str, List[Union[Text, Table]]]:
    """
    This function organizes the commentary inputs into a dictionary with the input desctription or title
    as the key.
    """
    input_ids = set()
    deduplicated_inputs: List[Union[Text, Table]] = []
    for input in inputs:
        if isinstance(input, Text):
            if input.id not in input_ids:
                input_ids.add(input.id)
                deduplicated_inputs.append(input)
        else:
            # append all tables
            deduplicated_inputs.append(input)

    res = defaultdict(list)
    for input in inputs:
        if isinstance(input, Text):
            res[input.text_type].append(input)  # type: ignore
        elif isinstance(input, Table):
            res["Tables"].append(input)  # type: ignore
    # shuffle the texts order in each key so when removing the texts, the order is random
    for key in res:
        random.shuffle(res[key])

    return res  # type: ignore


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


async def get_statistics_for_theme(
    context: PlanRunContext, texts: List[ThemeText], date_range: DateRange
) -> List[StatisticsText]:
    # get all the features
    resp = await get_all_variables_metadata(context.user_id)
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
                matched_articles = await get_news_and_web_pages_for_topics(
                    GetNewsAndWebPagesForTopicsInput(
                        topics=[topic],
                        date_range=date_range,
                        max_num_articles_per_topic=MAX_MATCHED_ARTICLES_PER_TOPIC,
                    ),
                    context,
                )
                texts.extend(matched_articles)  # type: ignore
            except Exception as e:
                logger.warning(f"Failed to get news pool articles for topic {topic}: {e}")

    await tool_log(
        log=f"Retrieved {len(texts)} texts for topics: {topics}.",
        context=context,
        associated_data=texts,
    )
    return texts


async def prepare_geography_prompt(
    geography_prompt: Prompt, portfolio_holdings_expanded_df: pd.DataFrame, context: PlanRunContext
) -> FilledPrompt:
    """
    This function prepares the geography prompt for the commentary.
    """
    stock_ids = portfolio_holdings_expanded_df[STOCK_ID_COL_NAME_DEFAULT].to_list()
    stock_weights = portfolio_holdings_expanded_df["Weight"].to_list()
    df = await get_metadata_for_stocks(stock_ids, context)

    # we might not have found metadata for all the gbi_ids (rare)
    # so lets select the ones we found and keep them for history
    orig_stock_ids = {s.gbi_id: s for s in stock_ids}
    orig_stock_weights = {s.gbi_id: w for s, w in zip(stock_ids, stock_weights)}
    new_stock_ids = [orig_stock_ids.get(id) for id in list(df["gbi_id"])]
    new_stock_weights = [orig_stock_weights.get(id) for id in list(df["gbi_id"])]
    df["Security"] = new_stock_ids
    df["Weight"] = new_stock_weights

    # group by Country and sum the weights
    df = df.groupby("Country").agg({"Weight": "sum"}).reset_index()

    portfolio_geography_prompt = geography_prompt.format(
        portfolio_geography_df=df.to_string(index=False)
    )
    return portfolio_geography_prompt


async def prepare_portfolio_prompt(
    portfolio_prompt: Prompt,
    geography_prompt: Prompt,
    portfolio_related_tables: List[Table],
    context: PlanRunContext,
) -> FilledPrompt:
    """
    This function prepares the portfolio prompt for the commentary
    """
    table_mapping: Dict[str, pd.DataFrame] = {}
    for table in portfolio_related_tables:
        if table.title == PORTFOLIO_HOLDING_TABLE_NAME_EXPANDED:
            portfolio_holdings_expanded_df = table.to_df()
            table_mapping["portfolio_holdings_expanded"] = portfolio_holdings_expanded_df
        elif table.title == PORTFOLIO_HOLDING_TABLE_NAME_NOT_EXPANDED:
            portfolio_holdings_df = table.to_df()
            table_mapping["portfolio_holdings"] = portfolio_holdings_df
        elif table.title == BENCHMARK_HOLDING_TABLE_NAME_EXPANDED:
            benchmark_holdings_expanded_df = table.to_df()
            table_mapping["benchmark_holdings_expanded"] = benchmark_holdings_expanded_df
        elif table.title == BENCHMARK_HOLDING_TABLE_NAME_NOT_EXPANDED:
            benchmark_holdings_df = table.to_df()
            table_mapping["benchmark_holdings"] = benchmark_holdings_df

        # get the portfolio performance on levels in PORTFOLIO_PERFORMANCE_LEVELS
        for performance_level in PORTFOLIO_PERFORMANCE_LEVELS:
            if table.title == PORTFOLIO_PERFORMANCE_TABLE_BASE_NAME + performance_level:
                table_mapping[f"portfolio_perf_{performance_level}"] = table.to_df()

        # get the benchmark performance on levels in BENCHMARK_PERFORMANCE_LEVELS
        for performance_level in BENCHMARK_PERFORMANCE_LEVELS:
            if table.title == BENCHMARK_PERFORMANCE_TABLE_BASE_NAME + performance_level:
                table_mapping[f"benchmark_perf_{performance_level}"] = table.to_df()
    # check if the table is not being added then add a dummy table
    for performance_level in PORTFOLIO_PERFORMANCE_LEVELS:
        if f"portfolio_perf_{performance_level}" not in table_mapping:
            table_mapping[f"portfolio_perf_{performance_level}"] = pd.DataFrame()
    for performance_level in BENCHMARK_PERFORMANCE_LEVELS:
        if f"benchmark_perf_{performance_level}" not in table_mapping:
            table_mapping[f"benchmark_perf_{performance_level}"] = pd.DataFrame()
    # Prepare the geography prompt
    portfolio_geography_prompt = await prepare_geography_prompt(
        geography_prompt, portfolio_holdings_expanded_df, context
    )
    # convert StockID column to company name to aviod commentary using tickers
    for df in table_mapping:
        if STOCK_ID_COL_NAME_DEFAULT in table_mapping[df].columns:
            table_mapping[df][STOCK_ID_COL_NAME_DEFAULT] = table_mapping[df][
                STOCK_ID_COL_NAME_DEFAULT
            ].apply(lambda x: x.company_name)

    # extract top contributing stocks
    table_mapping["portfolio_stock_pos"] = table_mapping["portfolio_perf_stock"].head(5)
    table_mapping["portfolio_stock_neg"] = table_mapping["portfolio_perf_stock"].tail(5)
    table_mapping["benchmark_holdings_pos"] = table_mapping["benchmark_perf_stock"].head(5)
    table_mapping["benchmark_holdings_neg"] = table_mapping["benchmark_perf_stock"].tail(5)

    # Prepare the portfolio prompt
    portfolio_prompt_filled = portfolio_prompt.format(
        portfolio_holdings=table_mapping["portfolio_holdings"].to_string(index=False),
        portfolio_geography_prompt=portfolio_geography_prompt.filled_prompt,
        portfolio_performance_by_overall=table_mapping["portfolio_perf_overall"].to_string(
            index=False
        ),
        portfolio_performance_by_daily=table_mapping["portfolio_perf_daily"].to_string(index=False),
        portfolio_performance_by_sector=table_mapping["portfolio_perf_sector"].to_string(
            index=False
        ),
        portfolio_performance_by_security=table_mapping["portfolio_perf_security"].to_string(
            index=False
        ),
        portfolio_performance_by_stock_positive=table_mapping["portfolio_stock_pos"].to_string(
            index=False
        ),
        portfolio_performance_by_stock_negative=table_mapping["portfolio_stock_neg"].to_string(
            index=False
        ),
        benchmark_performance_by_stock_positive=table_mapping["benchmark_holdings_pos"].to_string(
            index=False
        ),
        benchmark_performance_by_stock_negative=table_mapping["benchmark_holdings_neg"].to_string(
            index=False
        ),
    )
    return portfolio_prompt_filled


async def prepare_universe_prompt(
    universe_performance_prompt: Prompt,
    universe_related_tables: List[Table],
) -> FilledPrompt:
    """
    This function prepares the universe prompt for the commentary.
    """
    table_mapping: Dict[str, pd.DataFrame] = {}
    for table in universe_related_tables:
        if table.title == UNIVERSE_HOLDINGS_TABLE_NAME:
            universe_holdings_df = table.to_df()
            table_mapping["universe_holdings"] = universe_holdings_df

        for performance_level in UNIVERSE_PERFORMANCE_LEVELS:
            if table.title == UNIVERSE_PERFORMANCE_TABLE_BASE_NAME + performance_level:
                table_mapping[f"universe_perf_{performance_level}"] = table.to_df()
    # check if the table is not being added then add a dummy table
    for performance_level in UNIVERSE_PERFORMANCE_LEVELS:
        if f"universe_perf_{performance_level}" not in table_mapping:
            table_mapping[f"universe_perf_{performance_level}"] = pd.DataFrame()
    # convert StockID column to company name to aviod commentary using tickers
    for df in table_mapping:
        if STOCK_ID_COL_NAME_DEFAULT in table_mapping[df].columns:
            table_mapping[df][STOCK_ID_COL_NAME_DEFAULT] = table_mapping[df][
                STOCK_ID_COL_NAME_DEFAULT
            ].apply(lambda x: x.company_name)

    # extract top contributing stocks
    table_mapping["universe_best_contributers"] = (
        table_mapping["universe_perf_security"]
        .sort_values(by="weighted-return", ascending=False)
        .head(5)
    )
    table_mapping["universe_worst_contributers"] = (
        table_mapping["universe_perf_security"]
        .sort_values(by="weighted-return", ascending=True)
        .head(5)
    )
    table_mapping["universe_best_performers"] = (
        table_mapping["universe_perf_security"].sort_values(by="return", ascending=False).head(5)
    )
    table_mapping["universe_worst_performers"] = (
        table_mapping["universe_perf_security"].sort_values(by="return", ascending=True).head(5)
    )

    # Prepare the portfolio prompt
    universe_prompt_filled = universe_performance_prompt.format(
        overall_performance=table_mapping["universe_perf_overall"].to_string(index=False),
        sector_performance=table_mapping["universe_perf_sector"].to_string(index=False),
        daily_performance=table_mapping["universe_perf_daily"].to_string(index=False),
        best_contributors=table_mapping["universe_best_contributers"].to_string(index=False),
        worst_contributors=table_mapping["universe_worst_contributers"].to_string(index=False),
        best_performers=table_mapping["universe_best_performers"].to_string(index=False),
        worst_performers=table_mapping["universe_worst_performers"].to_string(index=False),
    )

    return universe_prompt_filled


async def prepare_main_prompt(
    commentary_sys_prompt: Prompt,
    commentary_main_prompt: Prompt,
    previous_commentary_prompt: FilledPrompt,
    portfolio_prompt: FilledPrompt,
    universe_performance_prompt: FilledPrompt,
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

    main_prompt = commentary_main_prompt.format(
        previous_commentary_prompt=previous_commentary_prompt.filled_prompt,
        portfolio_prompt=portfolio_prompt.filled_prompt,
        universe_performance_prompt=universe_performance_prompt.filled_prompt,
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
                commentary_main_prompt.template,
                commentary_sys_prompt.template,
                previous_commentary_prompt.filled_prompt,
                portfolio_prompt.filled_prompt,
                stocks_stats_prompt.filled_prompt,
                watchlist_prompt.filled_prompt,
                client_type_prompt.filled_prompt,
                writing_style_prompt.filled_prompt,
                chat_context,
            ],
            output_len=10000,
        )
        main_prompt = commentary_main_prompt.format(
            previous_commentary_prompt=previous_commentary_prompt.filled_prompt,
            portfolio_prompt=portfolio_prompt.filled_prompt,
            universe_performance_prompt=universe_performance_prompt.filled_prompt,
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
        # show the length of tokens in input_mapping
        for text_type, text_list in text_mapping.items():
            text_list_str: str = await Text.get_all_strs(TextGroup(val=text_list), include_header=True)  # type: ignore
            logger.info(
                f"Length of tokens in {text_type}: {GPTTokenizer(COMMENTARY_LLM).get_token_length(text_list_str)}"
            )
    return main_prompt


async def get_top_bottom_stocks(
    tables: List[Table], top_n_stocks: int, table_title: str
) -> List[StockID]:
    perf_df_found = False
    for table in tables:
        if table.title == table_title:
            perf_df = table.to_df()
            perf_df_found = True

    if not perf_df_found:
        raise ValueError(f"Performance table ({table_title}) for stock level not found in tables.")

    # get top and bottom 3 contributers and performers
    top_contributers = (
        perf_df.sort_values("weighted-return", ascending=False)
        .head(top_n_stocks)[STOCK_ID_COL_NAME_DEFAULT]
        .tolist()
    )
    bottom_contributers = (
        perf_df.sort_values("weighted-return", ascending=True)
        .head(top_n_stocks)[STOCK_ID_COL_NAME_DEFAULT]
        .tolist()
    )
    top_performers = (
        perf_df.sort_values("return", ascending=False)
        .head(top_n_stocks)[STOCK_ID_COL_NAME_DEFAULT]
        .tolist()
    )
    bottom_performers = (
        perf_df.sort_values("return", ascending=True)
        .head(top_n_stocks)[STOCK_ID_COL_NAME_DEFAULT]
        .tolist()
    )

    return top_contributers + bottom_contributers + top_performers + bottom_performers
