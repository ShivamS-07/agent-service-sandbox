# type: ignore
from agent_service.io_type_utils import IOType
from agent_service.io_types.text import Text
from regression_test.test_regression import TestExecutionPlanner, get_output, skip_in_ci
from regression_test.util import compare_with_expected_text


class TestMarketCommentary(TestExecutionPlanner):
    @skip_in_ci
    def test_market_commentary_jan_2024(self):
        prompt = """
        Write a commentary on market performance for the last week of Jan 2024.
        Your response should be in one paragraph less than 200 words.
        """

        def validate_output(prompt: str, output: IOType):
            output_text = get_output(output=output)
            expected_text = (
                "In the last week of January 2024, the S&P 500 experienced a slight decline, "
                "closing with a cumulative return of -1.06%. "
                "This downturn was primarily driven by the underperformance of the Information "
                "Technology sector, which saw a return of -0.83%. "
                "Key contributors to this decline included major tech companies like AAPL and MSFT, "
                "which faced challenges such as disappointing "
                "earnings reports and regulatory pressures in Europe. Conversely, "
                "the Health Care sector provided some positive momentum, "
                "with companies like LLY and UNH showing gains due to strong earnings "
                "and positive market sentiment. "
                "Despite these mixed performances, the overall market sentiment remained cautious, "
                "reflecting broader economic "
                "uncertainties and investor concerns about future growth prospects. "
            )
            self.loop.run_until_complete(
                compare_with_expected_text(
                    llm=self.llm,
                    output_text=output_text,
                    prompt=prompt,
                    expected_text=expected_text,
                )
            )

        self.prompt_test(
            prompt=prompt,
            validate_output=validate_output,
            required_tools=["get_commentary_inputs", "write_commentary"],
        )

    @skip_in_ci
    def test_market_commentary_past_week(self):
        prompt = (
            "Write a commentary on SP500 performance over the last week. "
            "Set macroeconomic=False."
            "Your response should be in one paragraph less than 200 words."
        )

        def validate_output(prompt: str, output: IOType):
            output_text = get_output(output=output)
            (
                self.assertTrue(isinstance(output_text, Text)),
                f"Expected type: Text, Actual type: {type(output_text)}",
            )
            self.assertTrue(output_text.val), "Expected non empty string"

        self.prompt_test(
            prompt=prompt,
            validate_output=validate_output,
            required_tools=["get_commentary_inputs", "write_commentary"],
            raise_plan_validation_error=True,
        )
