# type: ignore
from typing import List

from agent_service.io_type_utils import IOType
from agent_service.io_types.stock import StockID
from regression_test.test_regression import TestExecutionPlanner, get_output, skip_in_ci


class TestStockRank(TestExecutionPlanner):
    @skip_in_ci
    def test_ranking_for_top5(self):
        num_top_companies = 5
        prompt = "Rank the top 5 companies working on generative AI in the TSX60"

        def validate_output(prompt: str, output: IOType):
            ranked_stocks: List[StockID] = get_output(output)
            self.assertEqual(num_top_companies, len(ranked_stocks))
            for stock in ranked_stocks:
                self.assertIsInstance(stock, StockID)
                self.assertGreaterEqual(len(stock.history), 1)
                self.assertIsInstance(stock.history[-1].explanation)
                self.assertIsNone(stock.history[-1].score)

        self.prompt_test(
            prompt=prompt,
            validate_output=validate_output,
            required_tools=["filter_and_rank_stocks_by_profile"],
        )

    @skip_in_ci
    def test_ranking_all_stocks(self):
        prompt = (
            "Rank the in the QQQ by their commitment to ESG (Environmental, Social, and Governance)"
        )

        def validate_output(prompt: str, output: IOType):
            ranked_stocks: List[StockID] = get_output(output)
            for stock in ranked_stocks:
                self.assertIsInstance(stock, StockID)
                self.assertGreaterEqual(len(stock.history), 1)
                self.assertIsInstance(stock.history[-1].explanation)
                self.assertIsNone(stock.history[-1].score)

        self.prompt_test(
            prompt=prompt,
            validate_output=validate_output,
            required_tools=["filter_and_rank_stocks_by_profile"],
        )
