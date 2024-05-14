import asyncio
from typing import Dict, List, Tuple

from agent_service.GPT.constants import DEFAULT_CHEAP_MODEL, NO_PROMPT
from agent_service.GPT.requests import GPT
from agent_service.tool import ToolArgs, ToolCategory, tool
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
    "on stocks. The search is based on a string reference to the theme. An theme identifier"
    " is returned",
    category=ToolCategory.THEME,
)
async def get_macroeconomic_theme(args: GetMacroeconomicThemeInput, context: PlanRunContext) -> str:
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
        return theme_id_lookup[result]
    else:
        # TODO we should actually throw an error and use it to revise the plan
        return ""


class GetStocksAffectedByThemeInput(ToolArgs):
    theme_id: str
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
    pos_trend, neg_trend = db.get_theme_outlooks(args.theme_id)
    result = await llm.do_chat_w_sys_prompt(
        OUTLOOK_PROMPT.format(
            chat_context=context.chat.get_gpt_input(), pos_trend=pos_trend, neg_trend=neg_trend
        ),
        NO_PROMPT,
    )
    is_neg_trend = result == "2"
    stock_polarity_lookup = db.get_theme_stock_polarity_lookup(args.theme_id)
    final_stocks: List[int] = []
    for stock, polarity in stock_polarity_lookup.items():
        if is_neg_trend:
            polarity = not polarity
        if polarity == args.positive:  # matches the desired polarity
            final_stocks.append(stock)
    return final_stocks


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
    theme_id: str = await get_macroeconomic_theme(  # type: ignore
        args=GetMacroeconomicThemeInput(theme_ref="recession"), context=plan_context
    )
    stocks = await get_stocks_affected_by_theme(
        GetStocksAffectedByThemeInput(theme_id=theme_id, positive=False), context=plan_context
    )
    print(stocks)


if __name__ == "__main__":
    asyncio.run(main())
