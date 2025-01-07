# type: ignore
import unittest

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
                "get_latest_news_for_companies",
                "get_earnings_call_summaries",
            ],
        )

    @skip_in_ci
    def test_stock_screener_qqq_feb_2024(self):
        prompt = (
            "Find companies in QQQ with a PE < 25 and PE > 20 in Feb 2024 that mention "
            "greenshoots in their Q1 earnings."
        )

        def validate_output(prompt: str, output: IOType):
            output_stock_ids = get_output(output=output)
            actual_gbi_ids = [output_stock_id.gbi_id for output_stock_id in output_stock_ids]
            actual_gbi_ids.sort()
            expected_gbi_ids = set([713])  # AMAT
            self.assertTrue(
                expected_gbi_ids.issubset(set(actual_gbi_ids)),
                f"Some expected stocks missing {expected_gbi_ids=}, {actual_gbi_ids=}",
            )

        self.prompt_test(
            prompt=prompt,
            validate_output=validate_output,
            required_tools=["get_statistic_data_for_companies", "get_earnings_call_summaries"],
            raise_plan_validation_error=True,
            required_sample_plans=[],
        )

    @unittest.skip("takes 12+ minutes to compute this, after we removed the theme tool")
    def test_best_gen_ai_stocks_july_2024(self):
        prompt = "Give me your best Generative AI theme buying ideas for first week of July 2024"

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

    @unittest.skip("Often results in no stocks, need to figure out better way to test")
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

    @skip_in_ci
    def test_stock_screener_qqq_canaary(self):
        prompt = "Find companies in QQQ that mention revenue in their most recent earnings."

        def validate_output(prompt: str, output: IOType):
            output_stock_ids = get_output(output=output)
            self.assertGreater(len(output_stock_ids), 0, msg="No stocks returned")

        self.prompt_test(
            prompt=prompt,
            validate_output=validate_output,
            required_tools=["filter_and_rank_stocks_by_profile"],
            raise_plan_validation_error=True,
        )

    @skip_in_ci
    def test_growth_garp1(self):
        prompt = """Small and medium cap biotechnology companies that
        have made a major technology breakthrough in the past year
        that match growth at a reasonable price. You’ll need to look in the r2k
        """

        def validate_output(prompt: str, output: IOType):
            # how do I ONLY do the tool check?
            # an not even execute the plan
            return

        self.prompt_test(
            prompt=prompt,
            validate_output=validate_output,
            required_tools=["growth_filter", "value_filter"],
            disallowed_tools=["get_risk_exposure_for_stocks"],
            raise_plan_validation_error=True,
            only_validate_plan=True,
        )

    @unittest.skip("This test is sometimes failing")
    def test_growth_garp2(self):
        prompt = """I want growth at a reasonable price.
        Strong balance sheet. High volume. Market cap above 500M.
        Strong management teams. Technology companies.
        The company should ideally have made a major technology breakthrough over the past year.
        """

        def validate_output(prompt: str, output: IOType):
            # how do I ONLY do the tool check?
            return

        self.prompt_test(
            prompt=prompt,
            validate_output=validate_output,
            required_tools=["growth_filter", "value_filter"],
            disallowed_tools=[],
            raise_plan_validation_error=True,
            only_validate_plan=True,
        )
