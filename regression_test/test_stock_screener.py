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
            "Can you find good buying opportunities in SPY? Make sure PE > 20 "
            "and PE < 30, news is positive, and their earnings mention Generative AI."
        )

        def validate_output(prompt: str, output: IOType):
            output_stock_ids = get_output(output=output)
            self.assertGreater(len(output_stock_ids), 0)

        self.prompt_test(
            prompt=prompt, validate_plan=validate_plan, validate_output=validate_output
        )
