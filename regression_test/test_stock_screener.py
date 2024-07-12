# type: ignore
from agent_service.io_type_utils import IOType
from regression_test.test_regression import (
    OutputValidationError,
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

    def test_stock_screener_qqq_feb_2024(self):
        prompt = (
            "Find companies in QQQ with a PE < 30 and PE > 10 in Feb 2024 that mention "
            "greenshoots in their Q1 earnings."
        )

        def validate_output(prompt: str, output: IOType):
            output_stock_ids = get_output(output=output)
            self.assertGreater(len(output_stock_ids), 0)
            if len(output_stock_ids) != 1:
                raise OutputValidationError(
                    f"Number of output stock id's are {len(output_stock_ids)} rather than 1"
                )
            actual_gbi_ids = [output_stock_id.gbi_id for output_stock_id in output_stock_ids]
            actual_gbi_ids.sort()
            expected_gbi_ids = [713]
            if actual_gbi_ids != expected_gbi_ids:
                raise OutputValidationError(
                    f"Output expected id's are {actual_gbi_ids} rather than {expected_gbi_ids}"
                )

        self.prompt_test(
            prompt=prompt,
            validate_plan=validate_plan,
            validate_output=validate_output,
            raise_plan_validation_error=True,
        )

    def test_best_gen_ai_stocks_july_2024(self):
        prompt = "Give me your best Generative AI buying ideas for first week of July 2024"

        def validate_output(prompt: str, output: IOType):
            output_stock_ids = get_output(output=output)
            self.assertGreater(len(output_stock_ids), 0)
            if len(output_stock_ids) != 5:
                raise OutputValidationError(
                    f"Number of output stock id's are {len(output_stock_ids)} rather than 5"
                )
            actual_gbi_ids = [output_stock_id.gbi_id for output_stock_id in output_stock_ids]
            actual_gbi_ids.sort()
            expected_gbi_ids = [149, 155, 714, 1694, 10096]
            if actual_gbi_ids != expected_gbi_ids:
                raise OutputValidationError(
                    f"Output expected id's are {actual_gbi_ids} rather than {expected_gbi_ids}"
                )

        self.prompt_test(
            prompt=prompt,
            validate_plan=validate_plan,
            validate_output=validate_output,
        )

    def test_stock_screener_qqq(self):
        prompt = (
            "Find companies in QQQ with a PE < 30 and PE > 10 that mention "
            "greenshoots in their most recent earnings."
        )

        def validate_output(prompt: str, output: IOType):
            output_stock_ids = get_output(output=output)
            self.assertGreater(len(output_stock_ids), 0)

        self.prompt_test(
            prompt=prompt,
            validate_plan=validate_plan,
            validate_output=validate_output,
            raise_plan_validation_error=True,
        )

    def test_best_gen_ai_stocks(self):
        prompt = "Give me your best Generative AI buying ideas on a weekly basis"

        def validate_output(prompt: str, output: IOType):
            output_stock_ids = get_output(output=output)
            self.assertTrue(output_stock_ids)

        self.prompt_test(
            prompt=prompt,
            validate_plan=validate_plan,
            validate_output=validate_output,
            raise_plan_validation_error=True,
        )
