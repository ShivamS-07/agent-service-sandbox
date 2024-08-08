# type: ignore
from typing import DefaultDict, List

from agent_service.io_type_utils import IOType
from regression_test.test_regression import TestExecutionPlanner, get_output, skip_in_ci


class TestHomeScreenQueries(TestExecutionPlanner):
    @skip_in_ci
    def test_home_screen_q1_commentary(self):
        prompt = (
            "Write a commentary on market performance over the last month. Back"
            " observations up with data. Format the commentary to make it easy"
            " to read."
        )

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
        prompt = (
            "Identify the top 5 performing stocks in the technology sector over"
            " the past year, focusing on companies with a market cap above"
            " $10B. Provide a summary of their recent financial performance and"
            " any significant news"
        )

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
        prompt = (
            "Summarize all the major developments for COST over the past year."
            " Focus your analysis on corporate filings and earnings calls. Show"
            " the developments in point form as a timeline with dates. Bold"
            " anything important. For each development mention if it is"
            " positive or negative and why it is significant to COST"
        )

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
        prompt = (
            "Give me all information available about NFLX that will affect its"
            " upcoming earnings call, use only top news sources, refer to past"
            " earnings transcripts. Show answer in bullet points."
        )

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
        prompt = (
            "Look at the bottom 100 stocks in SPY based on their percentage"
            " gain over the past year. Then filter that down to companies with"
            " a PE > 10 and PE < 25. Then filter those companies down to good"
            " buys with positive news over the past 3 months. Then add a deep"
            " dive into the top 3 ideas"
        )

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
