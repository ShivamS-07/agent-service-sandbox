# type: ignore
from typing import DefaultDict, List

from agent_service.canned_prompts.canned_prompts import CANNED_PROMPTS
from agent_service.io_type_utils import IOType
from regression_test.test_regression import TestExecutionPlanner, get_output, skip_in_ci


def get_canned_prompt_text(prompt_id: str) -> str:
    for canned_prompt in CANNED_PROMPTS:
        if canned_prompt["id"] == prompt_id:
            return canned_prompt["prompt"]
    raise Exception(f"Could not find canned prompt with {prompt_id=}")


class TestHomeScreenQueries(TestExecutionPlanner):
    @skip_in_ci
    def test_home_screen_q1_commentary(self):
        prompt = get_canned_prompt_text(prompt_id="write_commentary")

        def validate_output(prompt: str, output: IOType):
            output_text = get_output(output=output)
            self.assertTrue(len(output_text.val.strip()) > 0)

        def validate_tool_args(execution_log: DefaultDict[str, List[dict]]):
            # Tool - get_date_range
            tool_name = "get_date_range"
            self.assertTrue(tool_name in execution_log.keys())
            arg_name = "date_range_str"
            ref_tool_arg = execution_log[tool_name][0][arg_name]
            exp_tool_arg = "last month"
            self.assertEqual(
                ref_tool_arg,
                exp_tool_arg,
                f"Args tool {tool_name} - {arg_name}",
            )

        max_runs = 3
        for idx in range(max_runs):
            print(f"Run {idx+1} of {max_runs}\nprompt={prompt}")
            self.prompt_test(
                prompt=prompt,
                validate_output=validate_output,
                required_tools=[
                    "get_date_range",
                    "write_commentary",
                    "prepare_output",
                ],
                validate_tool_args=validate_tool_args,
            )

    @skip_in_ci
    def test_home_screen_q2_top_tech(self):
        prompt = get_canned_prompt_text(prompt_id="identify_top_tech_stocks")

        def validate_output(prompt: str, output: IOType):
            return

        def validate_tool_args(execution_log: DefaultDict[str, List[dict]]):
            # Tool - prepare_output -> There are 3 sections
            tool_name = "prepare_output"
            self.assertTrue(tool_name in execution_log.keys())
            self.assertTrue(len(execution_log[tool_name]) == 3)

        max_runs = 3
        for idx in range(max_runs):
            print(f"Run {idx+1} of {max_runs}\nprompt={prompt}")
            self.prompt_test(
                prompt=prompt,
                validate_output=validate_output,
                required_tools=[
                    "get_stock_universe",
                    "sector_identifier_lookup",
                    "sector_filter",
                    "get_statistic_data_for_companies",
                    "transform_table",
                    "get_stock_identifier_list_from_table",
                    "get_date_range",
                    "get_universe_performance",
                    "get_all_text_data_for_stocks",
                    "summarize_texts",
                    "get_all_news_developments_about_companies",
                    "prepare_output",
                ],
                validate_tool_args=validate_tool_args,
            )

    @skip_in_ci
    def test_home_screen_q3_COST_developments(self):
        prompt = get_canned_prompt_text(prompt_id="summarize_costco")

        def validate_output(prompt: str, output: IOType):
            output_text = get_output(output=output)
            self.assertTrue(len(output_text.val.strip()) > 0)

        def validate_tool_args(execution_log: DefaultDict[str, List[dict]]):
            # Tool - get_date_range
            tool_name = "get_date_range"
            self.assertTrue(tool_name in execution_log.keys())
            arg_name = "date_range_str"
            ref_tool_arg = execution_log[tool_name][0][arg_name]
            exp_tool_arg = "last month"
            self.assertEqual(
                ref_tool_arg,
                exp_tool_arg,
                f"Args tool {tool_name} - {arg_name}",
            )
            # Tool - stock_identifier_lookup
            tool_name = "stock_identifier_lookup"
            self.assertTrue(tool_name in execution_log.keys())
            arg_name = "stock_name"
            ref_tool_arg = execution_log[tool_name][0][arg_name]
            exp_tool_arg = "COST"
            self.assertEqual(
                ref_tool_arg,
                exp_tool_arg,
                f"Args tool {tool_name} - {arg_name}",
            )

        max_runs = 3
        for idx in range(max_runs):
            print(f"Run {idx+1} of {max_runs}\nprompt={prompt}")
            self.prompt_test(
                prompt=prompt,
                validate_output=validate_output,
                required_tools=[
                    "stock_identifier_lookup",
                    "get_date_range",
                    "get_earnings_call_summaries",
                    "get_10k_10q_sec_filings",
                    "add_lists",
                    "summarize_texts",
                    "prepare_output",
                ],
                validate_tool_args=validate_tool_args,
            )

    @skip_in_ci
    def test_home_screen_q4_netlfix_earnings_call(self):
        prompt = get_canned_prompt_text(prompt_id="summarize_netflix")

        def validate_output(prompt: str, output: IOType):
            output_text = get_output(output=output)
            self.assertTrue(len(output_text.val.strip()) > 0)

        def validate_tool_args(execution_log: DefaultDict[str, List[dict]]):
            # Tool - get_date_range
            tool_name = "get_date_range"
            self.assertTrue(tool_name in execution_log.keys())
            arg_name = "date_range_str"
            ref_tool_arg = execution_log[tool_name][0][arg_name]
            exp_tool_arg = "last quarter"
            self.assertEqual(
                ref_tool_arg,
                exp_tool_arg,
                f"Args tool {tool_name} - {arg_name}",
            )
            # Tool - stock_identifier_lookup
            tool_name = "stock_identifier_lookup"
            self.assertTrue(tool_name in execution_log.keys())
            arg_name = "stock_name"
            ref_tool_arg = execution_log[tool_name][0][arg_name]
            exp_tool_arg = "NFLX"
            self.assertEqual(
                ref_tool_arg,
                exp_tool_arg,
                f"Args tool {tool_name} - {arg_name}",
            )

        max_runs = 3
        for idx in range(max_runs):
            print(f"Run {idx+1} of {max_runs}\nprompt={prompt}")
            self.prompt_test(
                prompt=prompt,
                validate_output=validate_output,
                required_tools=[
                    "stock_identifier_lookup",
                    "get_date_range",
                    "get_earnings_call_summaries",
                    "get_all_news_developments_about_companies",
                    "add_lists",
                    "summarize_texts",
                    "prepare_output",
                ],
                validate_tool_args=validate_tool_args,
            )

    @skip_in_ci
    def test_home_screen_q5_filter(self):
        prompt = get_canned_prompt_text(prompt_id="spy_pe_news_filter")

        def validate_output(prompt: str, output: IOType):
            return

        def validate_tool_args(execution_log: DefaultDict[str, List[dict]]):
            return

        max_runs = 1
        for idx in range(max_runs):
            print(f"Run {idx+1} of {max_runs}\nprompt={prompt}")
            self.prompt_test(
                prompt=prompt,
                validate_output=validate_output,
                required_tools=[],
                validate_tool_args=validate_tool_args,
            )

    @skip_in_ci
    def test_home_screen_q6_hypothesis_leader(self):
        prompt = "Is Nintendo a leader with millennial gamers?"

        def validate_output(prompt: str, output: IOType):
            output_text = get_output(output=output)
            self.assertTrue(len(output_text.val.strip()) > 0)

        def validate_tool_args(execution_log: DefaultDict[str, List[dict]]):
            # Tool - stock_identifier_lookup
            tool_name = "stock_identifier_lookup"
            self.assertTrue(tool_name in execution_log.keys())
            arg_name = "stock_name"
            ref_tool_arg = execution_log[tool_name][0][arg_name]
            exp_tool_arg = "Nintendo"
            self.assertEqual(
                ref_tool_arg,
                exp_tool_arg,
                f"Args tool {tool_name} - {arg_name}",
            )
            # Tool - prepare_output -> There are 3 sections
            tool_name = "prepare_output"
            self.assertTrue(tool_name in execution_log.keys())
            self.assertTrue(len(execution_log[tool_name]) == 3)

        max_runs = 3
        for idx in range(max_runs):
            print(f"Run {idx+1} of {max_runs}\nprompt={prompt}")
            self.prompt_test(
                prompt=prompt,
                validate_output=validate_output,
                required_tools=[
                    "get_stock_universe",
                    "get_company_descriptions",
                    "stock_identifier_lookup",
                    "filter_stocks_by_product_or_service",
                    "get_all_text_data_for_stocks",
                    "get_success_criteria",
                    "analyze_hypothesis_with_categories",
                    "generate_summary_for_hypothesis_with_categories",
                    "prepare_output",
                ],
                validate_tool_args=validate_tool_args,
            )
