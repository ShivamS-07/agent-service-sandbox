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
            "Find companies in QQQ with a PE < 30 and PE > 10 in Feb 2024 that mention "
            "greenshoots in their Q1 earnings."
        )

        def validate_output(prompt: str, output: IOType):
            output_stock_ids = get_output(output=output)
            self.assertEqual(len(output_stock_ids), 1)
            self.assertEqual(output_stock_ids[0].gbi_id, 713)

        self.prompt_test(
            prompt=prompt,
            validate_plan=validate_plan,
            validate_output=validate_output,
            raise_plan_validation_error=True,
        )

    def test_best_gen_ai_stocks(self):
        prompt = "Give me your best Generative AI buying ideas for first week of July 2024"

        def validate_output(prompt: str, output: IOType):
            output_stock_ids = get_output(output=output)
            self.assertEqual(len(output_stock_ids), 5)
            actual_gbi_ids = [output_stock_id.gbi_id for output_stock_id in output_stock_ids]
            actual_gbi_ids.sort()
            expected_gbi_ids = [149, 155, 714, 1694, 10096]
            self.assertEqual(actual_gbi_ids, expected_gbi_ids)

        self.prompt_test(
            prompt=prompt,
            validate_plan=validate_plan,
            validate_output=validate_output,
            raise_plan_validation_error=True,
        )
