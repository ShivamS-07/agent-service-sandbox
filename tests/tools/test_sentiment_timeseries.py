import datetime
import unittest

from agent_service.io_types.dates import DateRange
from agent_service.io_types.stock import StockID
from agent_service.tools.news_sentiment_time_series import (
    GetNewsSentimentTimeSeriesInput,
    get_news_sentiment_time_series,
)
from agent_service.types import PlanRunContext

AAPL = StockID(gbi_id=714, isin="", symbol="AAPL", company_name="")


class TestTimeSeriesNewsSentiment(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.context = PlanRunContext.get_dummy()

    async def test_time_series_news_sentiment(self):
        end_date = datetime.datetime.now().date()
        start_date = end_date - datetime.timedelta(days=30)
        daterange = DateRange(start_date=start_date, end_date=end_date)
        args = GetNewsSentimentTimeSeriesInput(stock_ids=[AAPL], date_range=daterange)
        result = await get_news_sentiment_time_series(args, self.context)
        df = result.to_df()
        self.assertGreater(len(df), 0)  # num_dates
