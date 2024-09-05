# type: ignore
import unittest

from agent_service.io_type_utils import IOType
from regression_test.test_regression import TestExecutionPlanner, get_output, skip_in_ci
from regression_test.util import validate_text


class TestHypothesis(TestExecutionPlanner):
    @skip_in_ci
    def test_tsla_hypothesis(self):
        prompt = "Tesla stock increases when they announce a new autopilot feature"

        def validate_output(prompt: str, output: IOType):
            output_text = get_output(output=output)
            self.loop.run_until_complete(
                validate_text(llm=self.llm, output_text=output_text, prompt=prompt)
            )

        self.prompt_test(
            prompt=prompt,
            validate_output=validate_output,
            required_tools=[
                "get_all_news_developments_about_companies",
                "test_and_summarize_hypothesis_with_news_developments",
            ],
        )

    @unittest.skip("Unstable test")
    def test_pharmaceutical_sector_growth(self):
        prompt = "Pharmaceutical industry growth"

        def validate_output(prompt: str, output: IOType):
            output_text = get_output(output=output[1])
            self.loop.run_until_complete(
                validate_text(llm=self.llm, output_text=output_text, prompt=prompt)
            )

        self.prompt_test(
            prompt=prompt,
            validate_output=validate_output,
            required_tools=["sector_identifier_lookup", "sector_filter"],
        )
