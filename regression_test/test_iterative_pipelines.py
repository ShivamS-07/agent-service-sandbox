# type: ignore
from typing import DefaultDict, List

from agent_service.io_type_utils import IOType
from regression_test.test_regression import TestExecutionPlanner, skip_in_ci


class TestIterativePipelines(TestExecutionPlanner):
    @skip_in_ci
    def test_ideas_pipelines(self):
        prompt = (
            "Read recent news on the topic of AI, and identify 3 major trends."
            " For each trend, discuss any mention of the trends in earnings calls for companies in the ARTI ETF."
            " Separately, identify any stock in the ARTI ETF which might be benefit from each trend,"
            " based on the earnings calls."
        )

        def validate_output(prompt: str, output: IOType) -> bool:
            return

        def validate_tool_args(execution_log: DefaultDict[str, List[dict]]) -> bool:
            return

        self.prompt_test(
            prompt=prompt,
            validate_output=validate_output,
            required_tools=[
                "brainstorm_ideas_from_text",
                "per_idea_summarize_texts",
                "per_idea_filter_stocks_by_profile_match",
            ],
            required_sample_plans=["be3320c4-be89-4ae5-bf6c-ce6a341cfb59"],
            validate_tool_args=validate_tool_args,
        )

    @skip_in_ci
    def test_stock_group_pipelines(self):
        prompt = (
            "For stocks in the TSX, group by sector, and write a summary of earnings call sentiment for each sector."
            " Next, create a graph of average performance over the last year for each sector."
        )

        def validate_output(prompt: str, output: IOType) -> bool:
            return

        def validate_tool_args(execution_log: DefaultDict[str, List[dict]]) -> bool:
            return

        self.prompt_test(
            prompt=prompt,
            validate_output=validate_output,
            required_tools=[
                "create_stock_groups_from_table",
                "per_stock_group_summarize_texts",
                "per_stock_group_transform_table",
            ],
            validate_tool_args=validate_tool_args,
        )

    @skip_in_ci
    def test_per_stock_summarize(self):
        prompt = (
            "For each of MSFT, NVDA, and TSLA, write a summary of their AI news for the last week"
        )

        def validate_output(prompt: str, output: IOType) -> bool:
            return

        def validate_tool_args(execution_log: DefaultDict[str, List[dict]]) -> bool:
            return

        self.prompt_test(
            prompt=prompt,
            validate_output=validate_output,
            required_tools=["per_stock_summarize_texts"],
            validate_tool_args=validate_tool_args,
        )

    @skip_in_ci
    def test_per_stock_competitors(self):
        prompt = (
            "Get competitors for each of AMD, V, and GM."
            " For each stock discuss any initiatives of competitors that might affect its business"
        )

        def validate_output(prompt: str, output: IOType) -> bool:
            return

        def validate_tool_args(execution_log: DefaultDict[str, List[dict]]) -> bool:
            return

        self.prompt_test(
            prompt=prompt,
            validate_output=validate_output,
            required_tools=[
                "per_stock_get_general_peers",
                "collapse_stock_groups_to_stock_list",
                "per_stock_group_summarize_texts",
            ],
            validate_tool_args=validate_tool_args,
        )
