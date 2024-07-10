# type: ignore
from agent_service.io_type_utils import IOType
from regression_test.test_regression import (
    TestExecutionPlanner,
    get_output,
    validate_plan,
)


class TestStockScreener(TestExecutionPlanner):
    def test_stock_screener_spy(self):
        prompt = (
            "Can you find good buying opportunities in SPY as of April 2024? Make sure PE > 20 "
            "and PE < 50, news is positive, and their earnings mention Generative AI."
        )

        def validate_output(prompt: str, output: IOType):
            output_stock_ids = get_output(output=output)
            self.assertGreater(len(output_stock_ids), 0)

        self.prompt_test(
            prompt=prompt, validate_plan=validate_plan, validate_output=validate_output
        )

    def test_stock_screener_qqq(self):
        prompt = (
            "Find companies in QQQ as of Feb 2024 with a PE < 30 and PE > 10 that mention "
            "greenshoots in their most recent earnings."
        )

        def validate_output(prompt: str, output: IOType):
            output_stock_ids = get_output(output=output)
            self.assertGreater(len(output_stock_ids), 0)

        self.prompt_test(
            prompt=prompt, validate_plan=validate_plan, validate_output=validate_output
        )

    def test_best_gen_ai_stocks(self):
        prompt = "Give me your best Generative AI buying ideas on a weekly basis"

        def validate_output(prompt: str, output: IOType):
            output_stock_ids = get_output(output=output)
            self.assertGreater(len(output_stock_ids), 0)

        self.prompt_test(
            prompt=prompt, validate_plan=validate_plan, validate_output=validate_output
        )
