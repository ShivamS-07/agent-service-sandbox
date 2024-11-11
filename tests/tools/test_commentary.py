import datetime
import unittest
from unittest import IsolatedAsyncioTestCase
from uuid import uuid4

from agent_service.io_types.stock import StockID
from agent_service.io_types.table import Table
from agent_service.io_types.text import Text
from agent_service.tools.commentary.tools import (
    GetCommentaryInputsInput,
    WriteCommentaryInput,
    get_commentary_inputs,
    write_commentary,
)
from agent_service.tools.dates import DateRange
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
from agent_service.tools.universe import (
    UNIVERSE_HOLDINGS_TABLE_NAME,
    UNIVERSE_PERFORMANCE_LEVELS,
    UNIVERSE_PERFORMANCE_TABLE_BASE_NAME,
)
from agent_service.types import ChatContext, Message, PlanRunContext
from agent_service.utils.date_utils import get_now_utc

# the sample user_id, and portfolio_id are used for testing portfolio based commentary.
# belongs to the user "Mohammad Zarei"
# the agent/plan id and sample plan id are used for testing the edit previous commentary
user_id = "3b997275-dcfe-4c19-8bb2-3e1366c4d5f3"
agent_id = "7cb9fb8f-690e-4535-8b48-f6e63494c366"
plan_id = "b3330500-9870-480d-bcb1-cf6fe6b487e3"
portfolio_id = "eb820abf-d5a3-450d-85df-b72825e6dcff"

AAPL = StockID(gbi_id=714, isin="", symbol="AAPL", company_name="Apple")
ERGB = StockID(
    gbi_id=434782, isin="", symbol="ERGB", company_name="ErgoBilt, Inc."
)  # ergonomic chairs
TRQ = StockID(
    gbi_id=19694, isin="", symbol="TRQ", company_name="Turquoise Hill Resources Ltd."
)  # mining company


class TestCommentary(IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        input_text = (
            "Write a commentary on impact of cloud computing on military industrial complex"
            "and AAPL, ERGB and TRQ stocks."
        )
        user_message = Message(message=input_text, is_user_message=True, message_time=get_now_utc())
        chat_context = ChatContext(messages=[user_message])

        self.context = PlanRunContext(
            agent_id=agent_id,
            plan_id=plan_id,
            user_id=user_id,
            plan_run_id=str(uuid4()),
            chat=chat_context,
            skip_db_commit=True,
            skip_task_cache=True,
            run_tasks_without_prefect=True,
        )
        self.date_range = DateRange(
            start_date=datetime.date(2024, 11, 6),
            end_date=datetime.date(2024, 11, 8),
        )

    async def test_get_commentary_inputs(self):
        inputs = await get_commentary_inputs(
            GetCommentaryInputsInput(
                topics=["cloud computing"],
                date_ramge=self.date_range,
                stock_ids=[AAPL],
                portfolio_id=portfolio_id,
                universe_name="SP500",
                macroeconomic=True,
            ),
            self.context,
        )
        # check if ThemeText exists in the result
        self.assertTrue(any("ThemeText" in input.__class__.__name__ for input in inputs))
        # check if ThemeNewsDevelopmentText exists in the result
        self.assertTrue(
            any("ThemeNewsDevelopmentText" in input.__class__.__name__ for input in inputs)
        )
        # check if ThemeNewsDevelopmentArticlesText exists in the result
        self.assertTrue(
            any("ThemeNewsDevelopmentArticlesText" in input.__class__.__name__ for input in inputs)
        )
        # check if universe/portfolio/benchmark holding/perfoermance tables exist in the result
        checks = set(
            [
                PORTFOLIO_PERFORMANCE_TABLE_BASE_NAME + performance_level
                for performance_level in PORTFOLIO_PERFORMANCE_LEVELS
            ]
            + [
                BENCHMARK_PERFORMANCE_TABLE_BASE_NAME + performance_level
                for performance_level in BENCHMARK_PERFORMANCE_LEVELS
            ]
            + [
                UNIVERSE_PERFORMANCE_TABLE_BASE_NAME + performance_level
                for performance_level in UNIVERSE_PERFORMANCE_LEVELS
            ]
            + [
                UNIVERSE_HOLDINGS_TABLE_NAME,
                PORTFOLIO_HOLDING_TABLE_NAME_EXPANDED,
                PORTFOLIO_HOLDING_TABLE_NAME_NOT_EXPANDED,
                BENCHMARK_HOLDING_TABLE_NAME_EXPANDED,
                BENCHMARK_HOLDING_TABLE_NAME_NOT_EXPANDED,
            ]
        )
        for input in inputs:
            if isinstance(input, Table):
                if input.title in checks:
                    checks.remove(input.title)
        print("Tables that failed to find: ", checks)
        self.assertTrue(len(checks) == 0)

    @unittest.skip("skipped slow test")
    async def test_write_commentary(self):
        texts = await get_commentary_inputs(
            GetCommentaryInputsInput(
                topics=["cloud computing", "military industrial complex"],
                date_ramge=self.date_range,
                stock_ids=[AAPL, ERGB, TRQ],
                portfolio_id=portfolio_id,
            ),
            self.context,
        )
        print("Length of texts: ", len(texts))  # type: ignore

        self.args = WriteCommentaryInput(
            inputs=texts,  # type: ignore
            portfolio_id=portfolio_id,
            date_range=self.date_range,
            stock_ids=[AAPL, ERGB, TRQ],
        )
        result = await write_commentary(self.args, self.context)
        self.assertIsInstance(result, Text)

        # general commentary
        texts = await get_commentary_inputs(
            GetCommentaryInputsInput(
                topics=["cloud computing", "military industrial complex"],
                date_ramge=self.date_range,
                macroeconomic=True,
            ),
            self.context,
        )
        print("Length of texts: ", len(texts))  # type: ignore
        self.args = WriteCommentaryInput(
            inputs=texts,  # type: ignore
        )
        result = await write_commentary(self.args, self.context)
        self.assertIsInstance(result, Text)
