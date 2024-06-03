# Author(s): Mohammad Zarei, Julian Brooke

import asyncio
import datetime
from typing import Dict, List, Optional, Tuple

from nlp_service_proto_v1.themes_pb2 import ThemeOutlookType

from agent_service.external.nlp_svc_client import (
    get_all_themes_for_user,
    get_security_themes,
    get_top_themes,
)
from agent_service.GPT.constants import DEFAULT_CHEAP_MODEL, NO_PROMPT
from agent_service.GPT.requests import GPT
from agent_service.io_types.misc import StockID
from agent_service.io_types.text import (
    ThemeNewsDevelopmentArticlesText,
    ThemeNewsDevelopmentText,
    ThemeText,
)
from agent_service.tool import ToolArgs, ToolCategory, ToolRegistry, tool
from agent_service.types import ChatContext, Message, PlanRunContext
from agent_service.utils.date_utils import get_now_utc
from agent_service.utils.gpt_logging import GptJobIdType, GptJobType, create_gpt_context
from agent_service.utils.postgres import get_psql
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

    def get_theme_stock_polarity_lookup(self, theme_id: str) -> Dict[int, bool]:
        sql = """
                SELECT theme_id, gbi_id, is_positive_polarity, is_primary_impact
                FROM nlp_service.theme_impact_stocks
                WHERE theme_id = %(theme_id)s AND is_primary_impact = true
        """
        records = self.db.generic_read(sql, params={"theme_id": theme_id})
        return {record["gbi_id"]: record["is_positive_polarity"] for record in records}


LOOKUP_PROMPT = Prompt(
    name="THEME_LOOKUP_PROMPT",
    template=(
        "Your task is to identify which (if any) of a provided list of "
        "macroeconomic themes correspond to a provided reference to a theme. "
        "If there is an exact match, or one with a strong semantic overlap, return it, "
        "otherwise return None. Do not return anything else. Here is the list of themes:\n"
        "---\n"
        "{all_themes}\n"
        "---\n"
        "And here is the user provided theme references you are trying to match: "
        "{user_theme}."
        "Now output the matches from the list if you have found them: "
    ),
)

OUTLOOK_PROMPT = Prompt(
    name="THEME_OUTLOOK_PROMPT",
    template=(
        "Your task is to identify which of the two possible trends associated with a "
        "particular macroeconomic theme is a better fit for a client need, based on provided "
        "chat context. Return either either the number 1 or the number 2, nothing else. "
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
    ),
    category=ToolCategory.THEME,
)
async def get_macroeconomic_themes(
    args: GetMacroeconomicThemeInput, context: PlanRunContext
) -> List[ThemeText]:
    db = ThemePostgres()
    gpt_context = create_gpt_context(
        GptJobType.AGENT_TOOLS, context.agent_id, GptJobIdType.AGENT_ID
    )
    llm = GPT(context=gpt_context, model=DEFAULT_CHEAP_MODEL)
    theme_id_lookup = db.get_boosted_themes_lookup()
    matched_themes = [
        await llm.do_chat_w_sys_prompt(
            LOOKUP_PROMPT.format(all_themes="\n".join(theme_id_lookup), user_theme=theme),
            NO_PROMPT,
        )
        for theme in args.theme_refs
    ]
    themes = [
        ThemeText(id=theme_id_lookup[theme]) for theme in matched_themes if theme in theme_id_lookup
    ]
    if not themes:
        raise Exception("was not able to find any relevant themes matching the input")
    return themes  # type: ignore


class GetStocksAffectedByThemeInput(ToolArgs):
    theme: ThemeText
    positive: bool


@tool(
    description=(
        "This function returns a list of stocks (stock identifiers) that are either positively (if "
        "positive is True) or negatively affected (if positive is False) by the macroeconomic theme"
    ),
    category=ToolCategory.THEME,
)
async def get_stocks_affected_by_theme(
    args: GetStocksAffectedByThemeInput, context: PlanRunContext
) -> List[StockID]:
    if context.chat is None:
        return []
    db = ThemePostgres()
    gpt_context = create_gpt_context(
        GptJobType.AGENT_PLANNER, context.agent_id, GptJobIdType.AGENT_ID
    )
    llm = GPT(context=gpt_context, model=DEFAULT_CHEAP_MODEL)
    pos_trend, neg_trend = db.get_theme_outlooks(args.theme.id)
    result = await llm.do_chat_w_sys_prompt(
        OUTLOOK_PROMPT.format(
            chat_context=context.chat.get_gpt_input(), pos_trend=pos_trend, neg_trend=neg_trend
        ),
        NO_PROMPT,
    )
    is_neg_trend = result == "2"
    stock_polarity_lookup = db.get_theme_stock_polarity_lookup(args.theme.id)
    final_stocks: List[int] = []
    for stock, polarity in stock_polarity_lookup.items():
        if is_neg_trend:
            polarity = not polarity
        if polarity == args.positive:  # matches the desired polarity
            final_stocks.append(stock)
    if len(final_stocks) == 0:
        raise Exception("Found no stocks affected by this theme/polarity combination")
    return await StockID.from_gbi_id_list(final_stocks)


class GetMacroeconomicThemesAffectingStocksInput(ToolArgs):
    stock_ids: List[StockID]


@tool(
    description=(
        "This function takes a list of stock identifiers and returns a list of lists of "
        "macroeconomic theme text objects that are affecting the stocks. Each theme list "
        "corresponds to a stock in the input list."
    ),
    category=ToolCategory.THEME,
    tool_registry=ToolRegistry,
)
async def get_macroeconomic_themes_affecting_stocks(
    args: GetMacroeconomicThemesAffectingStocksInput, context: PlanRunContext
) -> List[List[ThemeText]]:
    resp = await get_security_themes(user_id=context.user_id, gbi_ids=args.stock_ids)

    gbi_id_to_idx = {e.gbi_id: idx for idx, e in enumerate(resp.security_themes)}

    result: List[List[ThemeText]] = []
    for stock_id in args.stock_ids:
        if stock_id not in gbi_id_to_idx:
            result.append([])

        idx = gbi_id_to_idx[stock_id.gbi_id]
        result.append([ThemeText(id=t.theme_id.id) for t in resp.security_themes[idx].themes])

    return result


class GetMacroeconomicThemeOutlookInput(ToolArgs):
    theme: ThemeText


@tool(
    description=(
        "This function takes a macroeconomic theme and returns the information about its current "
        "trend or outlook"
    ),
    category=ToolCategory.THEME,
    tool_registry=ToolRegistry,
)
async def get_macroeconomic_theme_outlook(
    args: GetMacroeconomicThemeOutlookInput, context: PlanRunContext
) -> str:
    resp = await get_all_themes_for_user(user_id=context.user_id)
    for theme in resp.themes:
        if not theme.owner_id.id:
            # we only support boosted themes for now because user themes may not have outlooks
            continue

        if theme.theme_id.id == args.theme.id:
            if theme.current_outlook == ThemeOutlookType.THEME_OUTLOOK_POS:
                return theme.positive_polarity_label
            else:
                return theme.negative_polarity_label

    raise Exception("No outlook for this theme")


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
    db = get_psql()
    res = []
    for theme in args.themes:
        sql = """
        SELECT development_id::TEXT, is_major_development, article_count
        FROM nlp_service.theme_developments
        WHERE theme_id = %s
        ORDER BY is_major_development DESC, article_count DESC
        """
        rows = db.generic_read(sql, [theme.id])
        if args.max_devs_per_theme:
            rows = rows[: args.max_devs_per_theme]
        ids = [row["development_id"] for row in rows]
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
    db = get_psql()
    for development in args.developments_list:
        sql = """
        SELECT news_id::TEXT
        FROM nlp_service.theme_news
        WHERE development_id = %s
        AND published_at >= NOW() - INTERVAL '2 years'
        ORDER BY published_at DESC
        """
        rows = db.generic_read(sql, [development.id])
        news_ids = [row["news_id"] for row in rows]
        if args.max_articles_per_development:
            news_ids = news_ids[: args.max_articles_per_development]
        res.extend([ThemeNewsDevelopmentArticlesText(id=id) for id in news_ids])

    if len(res) == 0:
        raise Exception("No articles relevant to theme found")

    return res


class GetTopNThemesInput(ToolArgs):
    start_date: Optional[datetime.date] = None
    theme_num: int = 3


@tool(
    description=(
        "This function returns the top N themes based on the date range and number of themes. "
        "Top themes are those that seem to be trending in the news. "
        "The tool can be used when the user does not provide any themes to focus on."
        "You MUST NEVER use this tool when a user has provided a topic, since you are not "
        "likely that the top themes will correspond to the themes of interest for the user, "
        "instead you should use the get_news_articles_for_topics tool which can get general news "
        "topics if there is not a matching theme for a topic. Never call this function and then apply a "
        "filter to a topic, it does not make sense!"
    ),
    category=ToolCategory.THEME,
    tool_registry=ToolRegistry,
)
async def get_top_N_themes(args: GetTopNThemesInput, context: PlanRunContext) -> List[ThemeText]:
    """
    This function returns the top N themes for a user based on the date range and number of themes.

    Args:
        args (GetTopNThemesInput): The input arguments for the tool.
        context (PlanRunContext): The context of the plan run.

    Returns:
        List[ThemeText]: The list of theme text objects.
    """

    # helper function to get the date range
    async def _get_date_range(start_date: Optional[datetime.date]) -> str:
        if start_date is None:
            return "1M"
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
        return closest_range[1]

    # TODO: fix this endpoint to get ids instead of theme names
    resp = await get_top_themes(
        user_id=context.user_id,
        section_types=["THEME"],
        date_range=await _get_date_range(args.start_date),
        number_per_section=args.theme_num,
    )
    theme_refs: List[str] = [str(t.name) for t in resp.topics]
    themes: List[ThemeText] = await get_macroeconomic_themes(  # type: ignore
        GetMacroeconomicThemeInput(theme_refs=theme_refs), context
    )
    return themes


async def main() -> None:
    input_text = "I want to find good stocks to short so that if a recession happens I'm protected."
    user_message = Message(message=input_text, is_user_message=True, message_time=get_now_utc())
    chat_context = ChatContext(messages=[user_message])
    plan_context = PlanRunContext(
        agent_id="123",
        plan_id="123",
        user_id="123",
        plan_run_id="123",
        chat=chat_context,
        run_tasks_without_prefect=True,
        skip_db_commit=True,
    )
    themes: List[ThemeText] = await get_macroeconomic_themes(  # type: ignore
        args=GetMacroeconomicThemeInput(theme_refs=["recession"]), context=plan_context
    )

    print(themes)
    stocks = await get_stocks_affected_by_theme(
        GetStocksAffectedByThemeInput(theme=themes[0], positive=False), context=plan_context
    )
    print(stocks)


if __name__ == "__main__":
    asyncio.run(main())
