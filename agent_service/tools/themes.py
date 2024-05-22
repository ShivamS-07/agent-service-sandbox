import asyncio
from typing import Dict, List, Tuple

from agent_service.external.nlp_svc_client import get_security_themes
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
    template="Your task is to identify which (if any) of a provided list of macroeconomic themes correspond to a particular provided reference to a theme. If there is an exact match, or one with a strong semantic overlap, return it, otherwise return None. Do not return anything else. Here is the list of themes:\n---\n{all_themes}\n---\nAnd here is the user provided theme you are trying to match: {user_theme}. Now output the match from the list if you have found it: ",  # noqa: E501
)

OUTLOOK_PROMPT = Prompt(
    name="THEME_OUTLOOK_PROMPT",
    template="Your task is to identify which of the two possible trends associated with a particular macroeconomic theme is a better fit for a client need, based on provided chat context. Return either either the number 1 or the number 2, nothing else. Here is the chat context:\n---\n{chat_context}\n---\nAnd here are the two options:\n1: {pos_trend}\n2: {neg_trend}",  # noqa: E501
)


class GetMacroeconomicThemeInput(ToolArgs):
    theme_ref: str


@tool(
    description="This searches for an existing analysis of a macroeconomic theme and its effects "
    "on stocks. The search is based on a string reference to the theme. A theme text object"
    " is returned",
    category=ToolCategory.THEME,
)
async def get_macroeconomic_theme(
    args: GetMacroeconomicThemeInput, context: PlanRunContext
) -> ThemeText:
    db = ThemePostgres()
    gpt_context = create_gpt_context(
        GptJobType.AGENT_PLANNER, context.agent_id, GptJobIdType.AGENT_ID
    )
    llm = GPT(context=gpt_context, model=DEFAULT_CHEAP_MODEL)
    theme_id_lookup = db.get_boosted_themes_lookup()
    result = await llm.do_chat_w_sys_prompt(
        LOOKUP_PROMPT.format(all_themes="\n".join(theme_id_lookup), user_theme=args.theme_ref),
        NO_PROMPT,
    )
    if result in theme_id_lookup:
        return ThemeText(id=theme_id_lookup[result])
    else:
        # TODO we should actually throw an error and use it to revise the plan
        return ThemeText(id="-1")


class GetStocksAffectedByThemeInput(ToolArgs):
    theme: ThemeText
    positive: bool


@tool(
    description=(
        "This gets a list of stocks (stock identifiers) that are either positively (if positive "
        "is True) or negatively affected (if positive is False) by the theme indicated by theme_id"
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
        "This function takes a list of stock identifiers and returns a list of lists of theme text"
        " objects that are affecting the stocks. Each theme list corresponds to a stock in the"
        " input list."
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


class GetThemeDevelopmentNewsInput(ToolArgs):
    # the theme text object to get the news for
    theme: ThemeText


@tool(
    description=(
        "This function takes a theme text object"
        " and returns the development news for that theme."
    ),
    category=ToolCategory.THEME,
    tool_registry=ToolRegistry,
)
async def get_news_developments_about_theme(
    args: GetThemeDevelopmentNewsInput, context: PlanRunContext
) -> List[ThemeNewsDevelopmentText]:
    """
    This function takes a theme and returns the development news for that theme.

    Args:
        args (GetThemeDevelopmentNewsInput): The theme to get the news for.
        context (PlanRunContext): The context of the plan run.

    Returns:
        List[ThemeNewsDevelopmentText]: List of news for the theme.
    """
    sql = """
    SELECT development_id::TEXT
    FROM nlp_service.theme_developments
    WHERE theme_id = %s
    """
    db = get_psql()
    rows = db.generic_read(sql, [args.theme.id])
    development_ids = [row["development_id"] for row in rows]
    if not development_ids:
        raise ValueError(f"No developments found for theme {args.theme.id}")
    return [ThemeNewsDevelopmentText(id=id) for id in development_ids]


class GetThemeDevelopmentNewsArticlesInput(ToolArgs):
    # the theme development news to get the articles for
    development: ThemeNewsDevelopmentText


@tool(
    description=(
        "This function takes a theme news development text object"
        " and returns the news articles text object for that theme news development."
    ),
    category=ToolCategory.THEME,
    tool_registry=ToolRegistry,
)
async def get_news_articles_for_theme_developments(
    args: GetThemeDevelopmentNewsArticlesInput, context: PlanRunContext
) -> List[ThemeNewsDevelopmentArticlesText]:
    """
    This function takes a theme news development text object
    and returns the article text object for that theme.

    Args:
        args (GetThemeDevelopmentNewsInput): The theme text object to get the news for.
        context (PlanRunContext): The context of the plan run.

    Returns:
        ThemeNewsDevelopmentArticlesText: The article text object for the theme news development.
    """
    sql = """
    SELECT news_id::TEXT
    FROM nlp_service.theme_news
    WHERE development_id = %s
    """
    db = get_psql()
    rows = db.generic_read(sql, [args.development.id])
    news_ids = [row["news_id"] for row in rows]
    if not news_ids:
        raise ValueError(f"No developments found for theme {args.development.id}")
    return [ThemeNewsDevelopmentArticlesText(id=id) for id in news_ids]


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
    theme: ThemeText = await get_macroeconomic_theme(  # type: ignore
        args=GetMacroeconomicThemeInput(theme_ref="recession"), context=plan_context
    )
    stocks = await get_stocks_affected_by_theme(
        GetStocksAffectedByThemeInput(theme=theme, positive=False), context=plan_context
    )
    print(stocks)


if __name__ == "__main__":
    asyncio.run(main())
