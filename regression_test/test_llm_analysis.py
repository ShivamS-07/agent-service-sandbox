# type: ignore
from typing import DefaultDict, List

from agent_service.io_type_utils import IOType
from regression_test.test_regression import TestExecutionPlanner, skip_in_ci


class TestLLMAnalysis(TestExecutionPlanner):
    @skip_in_ci
    def test_text_comparison(self):
        prompt = "Compare the discussion of AI in the last two TSLA earnings calls"

        def validate_output(prompt: str, output: IOType):
            return

        def validate_tool_args(execution_log: DefaultDict[str, List[dict]]):
            return

        self.prompt_test(
            prompt=prompt,
            validate_output=validate_output,
            required_tools=["compare_texts"],
            required_sample_plans=["55aa8f29-27df-4e67-a896-1051753ec0f3"],
            validate_tool_args=validate_tool_args,
        )

    @skip_in_ci
    def test_answer_question(self):
        prompt = "What countries does Netflix operate in?"

        def validate_output(prompt: str, output: IOType):
            return

        def validate_tool_args(execution_log: DefaultDict[str, List[dict]]):
            return

        self.prompt_test(
            prompt=prompt,
            validate_output=validate_output,
            required_tools=["answer_question_with_text_data"],
            validate_tool_args=validate_tool_args,
        )
