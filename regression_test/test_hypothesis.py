# type: ignore
from agent_service.io_type_utils import IOType
from regression_test.test_regression import (
    TestExecutionPlanner,
    get_output,
    validate_plan,
)
from regression_test.util import validate_and_compare_text


class TestHypothesis(TestExecutionPlanner):
    def test_tsla_hypothesis(self):
        prompt = "Tesla stock increases when they announce a new autopilot feature"

        def validate_output(prompt: str, output: IOType):
            output_text = get_output(output=output)
            self.loop.run_until_complete(
                validate_and_compare_text(llm=self.llm, output_text=output_text, prompt=prompt)
            )

        self.prompt_test(
            prompt=prompt, validate_plan=validate_plan, validate_output=validate_output
        )

    def test_pharmaceutical_sector_growth(self):
        prompt = "Pharmaceutical industry growth"

        def validate_output(prompt: str, output: IOType):
            output_text = get_output(output=output[1])
            self.loop.run_until_complete(
                validate_and_compare_text(llm=self.llm, output_text=output_text, prompt=prompt)
            )

        self.prompt_test(
            prompt=prompt, validate_plan=validate_plan, validate_output=validate_output
        )
