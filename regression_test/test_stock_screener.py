# type: ignore
from agent_service.io_type_utils import IOType
from regression_test.test_regression import TestExecutionPlanner, get_output, skip_in_ci


class TestStockScreener(TestExecutionPlanner):
    @skip_in_ci
    def test_stock_screener_spy(self):
        prompt = (
            "Can you find good buying opportunities in SPY as of April 2024? Make sure PE > 20 "
            "and PE < 50, news is positive, and their earnings mention Generative AI."
        )

        def validate_output(prompt: str, output: IOType):
            output_stock_ids = get_output(output=output)
            self.assertGreater(len(output_stock_ids), 0, msg="No stocks returned")

        self.prompt_test(
            prompt=prompt,
            validate_output=validate_output,
            required_tools=[
                "get_statistic_data_for_companies",
                "get_all_news_developments_about_companies",
                "get_earnings_call_summaries",
            ],
        )

    @skip_in_ci
    def test_stock_screener_qqq_feb_2024(self):
        prompt = (
            "Find companies in QQQ with a PE < 30 and PE > 10 in Feb 2024 that mention "
            "greenshoots in their Q1 earnings."
        )

        def validate_output(prompt: str, output: IOType):
            output_stock_ids = get_output(output=output)
            actual_gbi_ids = [output_stock_id.gbi_id for output_stock_id in output_stock_ids]
            actual_gbi_ids.sort()
            expected_gbi_ids = [713]
            self.assertEqual(
                actual_gbi_ids,
                expected_gbi_ids,
                "Output stocks don't match",
            )

        self.prompt_test(
            prompt=prompt,
            validate_output=validate_output,
            required_tools=["get_statistic_data_for_companies", "get_earnings_call_summaries"],
            raise_plan_validation_error=True,
            required_sample_plans=[],
        )

    @skip_in_ci
    def test_best_gen_ai_stocks_july_2024(self):
        prompt = "Give me your best Generative AI buying ideas for first week of July 2024"

        def validate_output(prompt: str, output: IOType):
            output_stock_ids = get_output(output=output)
            actual_gbi_ids = [output_stock_id.gbi_id for output_stock_id in output_stock_ids]
            actual_gbi_ids.sort()
            expected_gbi_ids = [149, 155, 714, 1694, 10096]

            self.assertEqual(
                actual_gbi_ids,
                expected_gbi_ids,
                "Output stocks don't match",
            )

        self.prompt_test(
            prompt=prompt,
            validate_output=validate_output,
            required_tools=["get_macroeconomic_themes"],
        )

    def test_stock_screener_qqq(self):
        prompt = (
            "Find companies in QQQ with a PE < 30 and PE > 10 that mention "
            "greenshoots in their most recent earnings."
        )

        def validate_output(prompt: str, output: IOType):
            output_stock_ids = get_output(output=output)
            self.assertGreater(len(output_stock_ids), 0, msg="No stocks returned")

        self.prompt_test(
            prompt=prompt,
            validate_output=validate_output,
            required_tools=["get_statistic_data_for_companies", "get_earnings_call_summaries"],
            raise_plan_validation_error=True,
        )

    def test_best_gen_ai_stocks(self):
        prompt = "Give me your best Generative AI buying ideas on a weekly basis"

        def validate_output(prompt: str, output: IOType):
            output_stock_ids = get_output(output=output)
            self.assertGreater(len(output_stock_ids), 0, msg="No stocks returned")

        self.prompt_test(
            prompt=prompt,
            validate_output=validate_output,
            required_tools=["get_macroeconomic_themes"],
            raise_plan_validation_error=True,
        )
