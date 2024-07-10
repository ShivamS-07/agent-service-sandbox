# type: ignore
from agent_service.io_type_utils import IOType
from regression_test.test_regression import (
    TestExecutionPlanner,
    get_output,
    validate_plan,
)
from regression_test.util import validate_and_compare_text


class TestMarketCommentary(TestExecutionPlanner):
    def test_market_commentary(self):
        prompt = "Write a market commentary of everything that has happened over the past week?"

        def validate_output(prompt: str, output: IOType):
            output_text = get_output(output=output)
            self.loop.run_until_complete(
                validate_and_compare_text(llm=self.llm, output_text=output_text, prompt=prompt)
            )

        self.prompt_test(
            prompt=prompt, validate_plan=validate_plan, validate_output=validate_output
        )
