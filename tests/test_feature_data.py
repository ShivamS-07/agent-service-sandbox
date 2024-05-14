import datetime
from unittest import IsolatedAsyncioTestCase

from agent_service.tools.feature_data import (
    FeatureDataInput,
    get_feature_data,
    get_latest_date,
)
from agent_service.types import ChatContext, Message, PlanRunContext

AAPL = 714
AMZN = 149
MSFT = 6963

CLOSE_PRICE = "spiq_close"


class TestStockIdentifierLookup(IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.context = PlanRunContext(
            plan_id="test",
            plan_run_id="test",
            user_id="test",
            user_email="test",
            task_id="test",
            agent_id="test",
            chat=ChatContext(messages=[Message(message="test", is_user_message=True)]),
        )

    async def test_feature_data_3_stock(self):
        args = FeatureDataInput(stock_ids=[AAPL, AMZN, MSFT], field_id=CLOSE_PRICE)
        result = await get_feature_data(args, self.context)
        self.assertEqual(result.val.shape[1], 3)  # num_stocks
        self.assertEqual(result.val.shape[0], 1)  # num_dates

    async def test_feature_data_1_stock_many_dates(self):
        args = FeatureDataInput(
            stock_ids=[AAPL],
            field_id=CLOSE_PRICE,
            start_date=datetime.date(2020, 1, 1),
            end_date=datetime.date(2020, 3, 1),
        )
        result = await get_feature_data(args, self.context)
        self.assertEqual(result.val.shape[1], 1)  # num_stocks
        self.assertGreater(result.val.shape[0], 20)  # num_dates

    async def test_get_latest_date(self):
        # test doesnt throw and returns the same value
        val1 = get_latest_date()
        val2 = get_latest_date()
        self.assertEqual(val1, val2)
