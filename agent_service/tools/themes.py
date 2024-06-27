# Author(s): Mohammad Zarei, Julian Brooke
import asyncio
import datetime
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

from nlp_service_proto_v1.themes_pb2 import ThemeOutlookType
from pa_portfolio_service_proto_v1.workspace_pb2 import StockAndWeight
from stock_universe_service_proto_v1.security_metadata_service_pb2 import (
    GetEtfHoldingsForDateResponse,
)

from agent_service.external import sec_meta_svc_client
from agent_service.external.nlp_svc_client import (
    get_all_themes_for_user,
    get_security_themes,
    get_top_themes,
)
from agent_service.external.pa_backtest_svc_client import (
    get_themes_with_impacted_stocks,
)
from agent_service.GPT.constants import DEFAULT_SMART_MODEL, NO_PROMPT
from agent_service.GPT.requests import GPT
from agent_service.io_type_utils import HistoryEntry, Score
from agent_service.io_types.dates import DateRange
from agent_service.io_types.stock import StockID
from agent_service.io_types.text import (
    Text,
    ThemeNewsDevelopmentArticlesText,
    ThemeNewsDevelopmentText,
    ThemeText,
)
from agent_service.tool import ToolArgs, ToolCategory, ToolRegistry, tool
from agent_service.tools.portfolio import PortfolioID
from agent_service.types import ChatContext, Message, PlanRunContext
from agent_service.utils.date_utils import get_now_utc
from agent_service.utils.gpt_logging import GptJobIdType, GptJobType, create_gpt_context
from agent_service.utils.postgres import get_psql
from agent_service.utils.prefect import get_prefect_logger
from agent_service.utils.prompt_utils import Prompt

# TODO: Don't use db


class ThemePostgres:
    def __init__(self) -> None:
        self.db = get_psql()

    def get_boosted_themes_lookup(self) -> Dict[str, str]:
        sql = """
            SELECT theme_id::TEXT, theme_name, active, user_id::TEXT, active, status
            FROM nlp_service.themes
            WHERE status = 'COMPLETE' AND user_id IS NULL and active = true
        """
        records = self.db.generic_read(sql)
        return {record["theme_name"]: record["theme_id"] for record in records}

    def get_theme_outlooks(self, theme_id: str) -> Tuple[str, str]:
        sql = """
            SELECT theme_id::TEXT, positive_label, negative_label
            FROM nlp_service.themes
            WHERE theme_id = %(theme_id)s
        """
        records = self.db.generic_read(sql, params={"theme_id": theme_id})
        return (records[0]["positive_label"], records[0]["negative_label"])

    def get_theme_stock_polarity_reason_similarity_lookup(
        self, theme_id: str
    ) -> Dict[int, Tuple[bool, str, float]]:
        sql = """
                SELECT theme_id, gbi_id, is_positive_polarity, is_primary_impact,
                reason, reason_json, similarity_score
                FROM nlp_service.theme_impact_stocks
                WHERE theme_id = %(theme_id)s AND is_primary_impact = true
        """
        records = self.db.generic_read(sql, params={"theme_id": theme_id})
        return {
            record["gbi_id"]: (
                record["is_positive_polarity"],
                (
                    "\n".join(record["reason_json"].values())
                    if record["reason_json"]
                    else record["reason"]
                ),
                record["similarity_score"],
            )
            for record in records
        }

    def get_news_dev_about_theme(self, theme_id: str) -> List[str]:
        sql = """
            SELECT development_id::TEXT, is_major_development, article_count
            FROM nlp_service.theme_developments
            WHERE theme_id = %s
            ORDER BY is_major_development DESC, article_count DESC
        """
        records = self.db.generic_read(sql, [theme_id])
        return [record["development_id"] for record in records]

    def get_news_articles_for_dev(self, development_id: str) -> List[str]:
        sql = """
            SELECT news_id::TEXT
            FROM nlp_service.theme_news
            WHERE development_id = %s
            AND published_at >= NOW() - INTERVAL '2 years'
            ORDER BY published_at DESC
        """
        records = self.db.generic_read(sql, [development_id])
        return [record["news_id"] for record in records]


# Initialize the db
db = ThemePostgres()

LOOKUP_PROMPT = Prompt(
    name="THEME_LOOKUP_PROMPT",
    template=(
        "Your task is to identify which (if any) of a provided list of "
        "macroeconomic themes correspond to a provided reference to a theme. "
        "If there is an exact match, or one with significant semantic overlap, write that match. "
        "Do not be too picky, if there's one that stands out as being clearly relevant, choose it. "
        "Otherwise write None. Do not write anything else. Here is the list of themes:\n"
        "---\n"
        "{all_themes}\n"
        "---\n"
        "And here is the user provided theme references you are trying to match: "
        "{user_theme}."
        "Now output the match from the list if you have found one: "
    ),
)

OUTLOOK_PROMPT = Prompt(
    name="THEME_OUTLOOK_PROMPT",
    template=(
        "Your task is to identify which of the two possible trends associated with a "
        "particular macroeconomic theme better fits the client's expectation for the future, based on the"
        "mention of the theme in the chat context. Return either either the number 1 or the number 2, nothing else. "
        "Here is the chat context:\n"
        "---\n"
        "{chat_context}\n"
        "---\n"
        "And here are the two options:\n"
        "1: {pos_trend}\n2: {neg_trend}"
    ),
)


class GetMacroeconomicThemeInput(ToolArgs):
    theme_refs: List[str]


@tool(
    description=(
        "This function searches for existing macroeconomic themes "
        "The search is based on a list of string references to the themes. "
        "A list of theme text objects is returned."
        "The string reference should be brief as possible while containing the core macroeconomic concept of interest, "
        "do not include language that indicates the polarity/direction, for example if the query is "
        "`we want rising oil price losers`, the theme reference is `Oil Prices`, both `rising` and `losers "
        "should be excluded."
    ),
    category=ToolCategory.THEME,
)
async def get_macroeconomic_themes(
    args: GetMacroeconomicThemeInput, context: PlanRunContext
) -> List[ThemeText]:
    gpt_context = create_gpt_context(
        GptJobType.AGENT_TOOLS, context.agent_id, GptJobIdType.AGENT_ID
    )
    llm = GPT(context=gpt_context, model=DEFAULT_SMART_MODEL)
    theme_id_lookup = db.get_boosted_themes_lookup()
    matched_themes = [
        await llm.do_chat_w_sys_prompt(
            LOOKUP_PROMPT.format(all_themes="\n".join(theme_id_lookup.keys()), user_theme=theme),
            NO_PROMPT,
        )
        for theme in args.theme_refs
    ]
    # drop duplicates
    matched_themes = set(matched_themes)

    themes = [
        ThemeText(id=theme_id_lookup[theme]) for theme in matched_themes if theme in theme_id_lookup
    ]
    if not themes:
        raise Exception("was not able to find any relevant themes matching the input")
    return themes  # type: ignore


class GetStocksAffectedByThemesInput(ToolArgs):
    themes: List[ThemeText]
    positive: bool


@tool(
    description=(
        "This function returns a list of stocks (stock identifiers) that are either positively (if "
        "positive is True) or negatively affected (if positive is False) by the macroeconomic themes"
        " The list of stocks includes information about how they are affect by the theme."
    ),
    category=ToolCategory.THEME,
)
async def get_stocks_affected_by_theme(
    args: GetStocksAffectedByThemesInput, context: PlanRunContext
) -> List[StockID]:
    if context.chat is None:
        return []
    db = ThemePostgres()
    gpt_context = create_gpt_context(
        GptJobType.AGENT_TOOLS, context.agent_id, GptJobIdType.AGENT_ID
    )
    llm = GPT(context=gpt_context, model=DEFAULT_SMART_MODEL)
    final_stocks_reason: Dict[int, List[Tuple[str, float]]] = defaultdict(list)
    for theme in args.themes:
        pos_trend, neg_trend = db.get_theme_outlooks(theme.id)
        result = await llm.do_chat_w_sys_prompt(
            OUTLOOK_PROMPT.format(
                chat_context=context.chat.get_gpt_input(), pos_trend=pos_trend, neg_trend=neg_trend
            ),
            NO_PROMPT,
        )
        is_neg_trend = result == "2"

        stock_polarity_reason_lookup = db.get_theme_stock_polarity_reason_similarity_lookup(
            theme.id
        )
        for stock, (polarity, reason, score) in stock_polarity_reason_lookup.items():
            if is_neg_trend:
                polarity = not polarity
            if polarity == args.positive:  # matches the desired polarity
                final_stocks_reason[stock].append((reason, score))
    if len(final_stocks_reason) == 0:
        raise Exception("Found no stocks affected by this theme/polarity combination")
    stock_ids = await StockID.from_gbi_id_list(list(final_stocks_reason.keys()))
    final_stock_ids = []
    for stock_id in stock_ids:
        for reason, similarity in final_stocks_reason[stock_id.gbi_id]:
            final_stock_ids.append(
                stock_id.inject_history_entry(
                    HistoryEntry(
                        explanation=reason,
                        title="Connection to theme",
                        score=Score.scale_input(similarity, lb=0, ub=1),
                    )
                )
            )

    return final_stock_ids


class GetMacroeconomicThemesAffectingStocksInput(ToolArgs):
    stock_ids: List[StockID]


@tool(
    description=(
        "This function takes a list of stock identifiers and returns a list of "
        "macroeconomic themes that are affecting the stocks."
        "This tool can be used when the themes related to a list of stocks are needed. "
    ),
    category=ToolCategory.THEME,
    tool_registry=ToolRegistry,
)
async def get_macroeconomic_themes_affecting_stocks(
    args: GetMacroeconomicThemesAffectingStocksInput, context: PlanRunContext
) -> List[ThemeText]:
    resp = await get_security_themes(
        user_id=context.user_id, gbi_ids=[stock_id.gbi_id for stock_id in args.stock_ids]
    )

    gbi_id_to_idx = {e.gbi_id: idx for idx, e in enumerate(resp.security_themes)}

    result: Set[ThemeText] = set()
    for stock_id in args.stock_ids:
        gbi_id = stock_id.gbi_id
        if gbi_id not in gbi_id_to_idx:
            continue

        idx = gbi_id_to_idx[stock_id.gbi_id]
        result.update([ThemeText(id=t.theme_id.id) for t in resp.security_themes[idx].themes])

    return list(result)


class GetMacroeconomicThemeOutlookInput(ToolArgs):
    themes: List[ThemeText]


@tool(
    description=(
        "This function takes a list of macroeconomic themes and returns the information about their current "
        "trend or outlook for each theme. There is one outlook text for each input theme."
    ),
    category=ToolCategory.THEME,
    tool_registry=ToolRegistry,
)
async def get_macroeconomic_theme_outlook(
    args: GetMacroeconomicThemeOutlookInput, context: PlanRunContext
) -> List[Text]:
    resp = await get_all_themes_for_user(user_id=context.user_id)
    theme_outlook_lookup = {}
    for theme in resp.themes:
        if theme.owner_id.id:
            # we only support boosted themes for now because user themes may not have outlooks
            continue

        if theme.current_outlook:
            if theme.current_outlook == ThemeOutlookType.THEME_OUTLOOK_POS:
                theme_outlook_lookup[theme.theme_id.id] = theme.positive_polarity_label
            else:
                theme_outlook_lookup[theme.theme_id.id] = theme.negative_polarity_label

    success_count = 0
    output: List[Text] = []
    for theme in args.themes:
        if theme.id in theme_outlook_lookup:
            output.append(Text(val=theme_outlook_lookup[theme.id]))
            success_count += 1
        else:
            output.append(Text())

    if success_count == 0:
        raise Exception("No outlooks for these themes")

    return output


class GetThemeDevelopmentNewsInput(ToolArgs):
    # the themes to get the news for
    themes: List[ThemeText]
    max_devs_per_theme: Optional[int] = None


@tool(
    description=(
        "This function takes a list of macroeconomic themes as theme text object "
        "and returns the list of corresponding developments news for all themes."
        "max_devs_per_theme is an optional parameter to limit the number of developments per theme, "
        "and None means no limit on the number of developments per theme. "
    ),
    category=ToolCategory.THEME,
    tool_registry=ToolRegistry,
)
async def get_news_developments_about_theme(
    args: GetThemeDevelopmentNewsInput, context: PlanRunContext
) -> List[ThemeNewsDevelopmentText]:
    """
    This function takes a list of theme(s) and returns the news development for them.

    Args:
        args (GetThemeDevelopmentNewsInput): The theme to get the news for.
        context (PlanRunContext): The context of the plan run.

    Returns:
        List[ThemeNewsDevelopmentText]: List of news development for the themes.
    """
    res = []
    for theme in args.themes:
        ids = db.get_news_dev_about_theme(theme.id)
        if args.max_devs_per_theme:
            ids = ids[: args.max_devs_per_theme]
        res.extend([ThemeNewsDevelopmentText(id=id) for id in ids])
    return res


class GetThemeDevelopmentNewsArticlesInput(ToolArgs):
    # the theme developments news to get the articles for
    developments_list: List[ThemeNewsDevelopmentText]
    max_articles_per_development: Optional[int] = None


@tool(
    description=(
        "This function takes a lists of macroeconomic theme news developments "
        "and returns the news articles for all theme news developments."
        "max_articles_per_development is an optional parameter to limit the "
        "number of articles per news development, and `None` means no limit on the "
        "number of articles per news development."
    ),
    category=ToolCategory.THEME,
    tool_registry=ToolRegistry,
)
async def get_news_articles_for_theme_developments(
    args: GetThemeDevelopmentNewsArticlesInput, context: PlanRunContext
) -> List[ThemeNewsDevelopmentArticlesText]:
    """
    This function takes a list of list of theme news developments
    and returns the article text object for each theme news development.

    Args:
        args (GetThemeDevelopmentNewsInput): The theme text object to get the news for.
        context (PlanRunContext): The context of the plan run.

    Returns:
        List[ThemeNewsDevelopmentArticlesText]: The list of article text object
            for each theme news development.
    """
    res = []
    for development in args.developments_list:
        news_ids = db.get_news_articles_for_dev(development.id)
        if args.max_articles_per_development:
            news_ids = news_ids[: args.max_articles_per_development]
        res.extend([ThemeNewsDevelopmentArticlesText(id=id) for id in news_ids])

    if len(res) == 0:
        raise Exception("No articles relevant to theme found")

    return res


class GetTopNThemesInput(ToolArgs):
    start_date: Optional[datetime.date] = None
    date_range: Optional[DateRange] = None
    theme_num: int = 3
    portfolio_id: Optional[PortfolioID] = None


@tool(
    description=(
        "This function can be used when you need to get news themes or topics related to a user portfolio "
        "(if portfolio_id is provided) or general news themes. "
        "The number of themes and the start date can be specified by user. "
        "If start date is provided use get_date_from_date_str tool to get the date. "
        "Adjust the theme_num to get the desired number of themes/topics. "
        "The tool can be used "
        " - when user does not have a specific theme in mind and wants to know the top themes "
        " - when user wants to know the top themes after a specific start date "
        " - when user wants to know the top themes for a specific portfolio"
    ),
    category=ToolCategory.THEME,
    tool_registry=ToolRegistry,
)
async def get_top_N_macroeconomic_themes(
    args: GetTopNThemesInput, context: PlanRunContext
) -> List[ThemeText]:
    """
    This function returns the top N themes for a user based on the date range and number of themes.

    Args:
        args (GetTopNThemesInput): The input arguments for the tool.
        context (PlanRunContext): The context of the plan run.

    Returns:
        List[ThemeText]: The list of theme text objects.
    """

    # helper function to get the date range
    async def _get_date_range(
        start_date: Optional[datetime.date],
    ) -> Tuple[datetime.timedelta, str]:
        if start_date is None:
            return datetime.timedelta(days=30), "1M"
        today = datetime.date.today()
        diff = today - start_date
        date_ranges = [
            (datetime.timedelta(weeks=1), "1W"),
            (datetime.timedelta(weeks=2), "2W"),
            (datetime.timedelta(days=30), "1M"),
            (datetime.timedelta(days=90), "3M"),
            (datetime.timedelta(days=91), "1Q"),
            (datetime.timedelta(days=182), "6M"),
            (datetime.timedelta(days=365), "1Y"),
        ]
        # Find the closest date range
        closest_range = min(date_ranges, key=lambda x: abs(x[0] - diff))
        return closest_range

    logger = get_prefect_logger(__name__)
    start_date = args.start_date
    if not start_date:
        if args.date_range:
            start_date = args.date_range.start_date
    time_delta, date_range = await _get_date_range(start_date)
    themes: List[ThemeText] = []
    if args.portfolio_id:
        # TODO: fix this endpoint to get ids instead of theme names
        resp = await get_top_themes(
            user_id=context.user_id,
            section_types=["THEME"],
            date_range=date_range,
            number_per_section=args.theme_num,
            portfolio_id=args.portfolio_id,
        )
        theme_refs: List[str] = [str(t.name) for t in resp.topics]
        theme_refs = theme_refs[: args.theme_num]
        themes = await get_macroeconomic_themes(  # type: ignore
            GetMacroeconomicThemeInput(theme_refs=theme_refs), context
        )
    # If we do not have a portfolio Id what we do is instead construct
    # a portfolio based on the S&P 500 and find relevant themes
    elif not args.portfolio_id:
        logger.info("No portfolio id provided using holdings in S&P 500")
        SPY_SP500_GBI_ID = 10076  # same on dev & prod
        response: GetEtfHoldingsForDateResponse = await sec_meta_svc_client.get_etf_holdings(
            SPY_SP500_GBI_ID, context.user_id
        )
        SPY_SP500_HOLDINGS = [
            StockAndWeight(gbi_id=holding.gbi_id, weight=holding.weight)
            for holding in response.etf_universe_holdings[0].holdings.weighted_securities
        ]
        # if start date is None default it to a
        themes_with_impact = await get_themes_with_impacted_stocks(
            stocks=SPY_SP500_HOLDINGS,
            start_date=start_date if start_date else datetime.date.today() - time_delta,
            end_date=(start_date + time_delta) if start_date else datetime.date.today(),
            user_id=context.user_id,
        )
        themes = [ThemeText(id=theme.theme_id) for theme in themes_with_impact]

    return themes[: args.theme_num]


async def main() -> None:
    input_text = "I want to find good stocks to short so that if a recession happens I'm protected."
    user_message = Message(message=input_text, is_user_message=True, message_time=get_now_utc())
    chat_context = ChatContext(messages=[user_message])
    plan_context = PlanRunContext(
        agent_id="123",
        plan_id="123",
        user_id="123",  # set a real user id for this to work
        plan_run_id="123",
        chat=chat_context,
        run_tasks_without_prefect=True,
        skip_db_commit=True,
    )
    themes: List[ThemeText] = await get_macroeconomic_themes(  # type: ignore
        args=GetMacroeconomicThemeInput(theme_refs=["Gen AI"]), context=plan_context
    )

    print(themes)
    stocks = await get_stocks_affected_by_theme(
        GetStocksAffectedByThemesInput(themes=themes, positive=False), context=plan_context
    )
    print(stocks)

    outlooks = await get_macroeconomic_theme_outlook(
        args=GetMacroeconomicThemeOutlookInput(themes=themes), context=plan_context
    )
    print(outlooks)

    more_themes = await get_macroeconomic_themes_affecting_stocks(
        GetMacroeconomicThemesAffectingStocksInput(stock_ids=stocks), context=plan_context  # type: ignore
    )
    print(more_themes)


if __name__ == "__main__":
    asyncio.run(main())
