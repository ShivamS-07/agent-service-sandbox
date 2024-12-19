# type: ignore
from typing import DefaultDict, List

from agent_service.io_type_utils import IOType
from regression_test.test_regression import TestExecutionPlanner, get_output, skip_in_ci


class TestWebSearch(TestExecutionPlanner):
    @skip_in_ci
    def test_general_web_search(self) -> None:
        prompt = "What is the gambling situation in Australia?"

        def validate_output(prompt: str, output: IOType) -> None:
            output_text = get_output(output=output)
            self.assertGreater(len(output_text.val), 0)

        def validate_tool_args(execution_log: DefaultDict[str, List[dict]]) -> None:
            tool_name = "summarize_texts"
            args = execution_log[tool_name][0]
            text_input = args["texts"]
            self.assertTrue(len(text_input) > 0)

        self.prompt_test(
            prompt=prompt,
            validate_output=validate_output,
            required_tools=[
                "general_web_search",
            ],
            validate_tool_args=validate_tool_args,
        )

    def test_general_stock_web_search(self) -> None:
        prompt = "Get me news on apple, microsoft and google"

        def validate_output(prompt: str, output: IOType) -> None:
            output_text = get_output(output=output)
            self.assertGreater(len(output_text.val), 0)

        def validate_tool_args(execution_log: DefaultDict[str, List[dict]]) -> None:
            tool_name = "get_latest_news_for_companies"
            args = execution_log[tool_name][0]
            stock_ids = args["stock_ids"]
            self.assertTrue(len(stock_ids) == 3)

        self.prompt_test(
            prompt=prompt,
            validate_output=validate_output,
            required_tools=[
                "get_latest_news_for_companies",
            ],
            validate_tool_args=validate_tool_args,
        )
