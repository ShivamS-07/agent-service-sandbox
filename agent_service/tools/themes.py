# Author(s): Mohammad Zarei, Julian Brooke

import asyncio
import datetime
from typing import Dict, List, Optional, Tuple
from uuid import uuid4

from nlp_service_proto_v1.themes_pb2 import ThemeOutlookType

from agent_service.external.nlp_svc_client import (
    get_all_themes_for_user,
    get_security_themes,
    get_top_themes,
)
from agent_service.GPT.constants import DEFAULT_CHEAP_MODEL, NO_PROMPT
from agent_service.GPT.requests import GPT
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
        # TODO we should actually throw an error and use it to revise the plan
        print(f"No themes found for: {args.theme_refs}!")
        themes = [ThemeText(id=str(uuid4()))]
    return themes


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
) -> List[int]:
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
    return final_stocks


class GetMacroeconomicThemesAffectingStocksInput(ToolArgs):
    stock_ids: List[int]


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

        idx = gbi_id_to_idx[stock_id]
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

    return "No outlook"


class GetThemeDevelopmentNewsInput(ToolArgs):
    # the themes to get the news for
    themes: List[ThemeText]


@tool(
    description=(
        "This function takes a list of macroeconomic themes as theme text object "
        "and returns the list of corresponding developments news for each theme."
    ),
    category=ToolCategory.THEME,
    tool_registry=ToolRegistry,
)
async def get_news_developments_about_theme(
    args: GetThemeDevelopmentNewsInput, context: PlanRunContext
) -> List[List[ThemeNewsDevelopmentText]]:
    """
    This function takes a list of theme(s) and returns the development news for them.

    Args:
        args (GetThemeDevelopmentNewsInput): The theme to get the news for.
        context (PlanRunContext): The context of the plan run.

    Returns:
        List[List[ThemeNewsDevelopmentText]]: List of news for each theme.
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
        ids = [row["development_id"] for row in rows]
        res.append([ThemeNewsDevelopmentText(id=id) for id in ids])
    return res


class GetThemeDevelopmentNewsArticlesInput(ToolArgs):
    # the theme developments news to get the articles for
    developments_list: List[List[ThemeNewsDevelopmentText]]
    start_date: Optional[datetime.date] = None


@tool(
    description=(
        "This function takes a list of lists of macroeconomic theme news developments "
        "and returns the news articles for each theme news development "
        "that are published after a given start date or "
        "in the last 30 days if no start date is provided."
    ),
    category=ToolCategory.THEME,
    tool_registry=ToolRegistry,
)
async def get_news_articles_for_theme_developments(
    args: GetThemeDevelopmentNewsArticlesInput, context: PlanRunContext
) -> List[List[List[ThemeNewsDevelopmentArticlesText]]]:
    """
    This function takes a list of list of theme news developments
    and returns the article text object for each theme news development.

    Args:
        args (GetThemeDevelopmentNewsInput): The theme text object to get the news for.
        context (PlanRunContext): The context of the plan run.

    Returns:
        List[List[List[ThemeNewsDevelopmentArticlesText]]]: The list of article text object
            for each theme news development.
    """
    start_date = args.start_date
    if not start_date:
        start_date = (get_now_utc() - datetime.timedelta(days=30)).date()
    res = []
    db = get_psql()
    for developments in args.developments_list:
        articles = []
        for development in developments:
            sql = """
            SELECT news_id::TEXT
            FROM nlp_service.theme_news
            WHERE development_id = %s
            AND published_at >= %s
            ORDER BY published_at DESC
            """
            rows = db.generic_read(sql, [development.id, start_date])
            news_ids = [row["news_id"] for row in rows]
            articles.append([ThemeNewsDevelopmentArticlesText(id=id) for id in news_ids])
        res.append(articles)
    return res


class GetTopNThemesInput(ToolArgs):
    date_range: str = "1M"
    theme_num: int = 3


@tool(
    description=(
        "This function returns the top N themes based on the date range and number of themes. "
        "The tool can be used when the user does not provide any themes to focus on, "
        "or when the user wants a general market trend commentary. "
        "The date range MUST be one of the following values: "
        "['1W', '2W', '1M', '3M', '1Q', '6M', '1Y']. Default is '1M'. "
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
    # TODO: fix this this endpoint to get ids instead of theme names
    resp = await get_top_themes(
        user_id=context.user_id,
        section_types=["THEME"],
        date_range=args.date_range,
        number_per_section=args.theme_num,
    )
    theme_refs = [t.name for t in resp.topics]
    # print(theme_refs)
    themes = await get_macroeconomic_themes(
        GetMacroeconomicThemeInput(theme_refs=theme_refs), context
    )
    return themes  # type: ignore


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
    )
    themes: List[ThemeText] = await get_macroeconomic_themes(  # type: ignore
        args=GetMacroeconomicThemeInput(theme_refs=["recession"]), context=plan_context
    )
    stocks = await get_stocks_affected_by_theme(
        GetStocksAffectedByThemeInput(theme=themes[0], positive=False), context=plan_context
    )
    print(stocks)


if __name__ == "__main__":
    asyncio.run(main())
