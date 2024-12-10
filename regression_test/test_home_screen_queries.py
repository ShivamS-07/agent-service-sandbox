# type: ignore
import logging
from typing import DefaultDict, List

from agent_service.canned_prompts.canned_prompts import CANNED_PROMPTS
from agent_service.io_type_utils import IOType
from regression_test.test_regression import TestExecutionPlanner, get_output, skip_in_ci

logger = logging.getLogger(__name__)


def get_canned_prompt_text(prompt_id: str) -> str:
    for canned_prompt in CANNED_PROMPTS:
        if canned_prompt["id"] == prompt_id:
            return canned_prompt["prompt"]
    raise Exception(f"Could not find canned prompt with {prompt_id=}")


class TestHomeScreenQueries(TestExecutionPlanner):
    # we run this test 3 times to catch any randomness
    # use duplicated test so that they can be run in parallel
    @skip_in_ci
    def test_home_screen_q1_commentary2(self):
        logger.warning("Run 2 of test_home_screen_q1_commentary")
        self.test_home_screen_q1_commentary()

    @skip_in_ci
    def test_home_screen_q1_commentary3(self):
        logger.warning("Run 3 of test_home_screen_q1_commentary")
        self.test_home_screen_q1_commentary()

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

        max_runs = 1
        for idx in range(max_runs):
            logger.warning(f"Run {idx + 1} of {max_runs}\nprompt={prompt}")
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

    # we run this test 3 times to catch any randomness
    # use duplicated test so that they can be run in parallel
    @skip_in_ci
    def test_home_screen_q2_top_tech2(self):
        logger.warning("Run 2 of test_home_screen_q2_top_tech")
        self.test_home_screen_q2_top_tech()

    @skip_in_ci
    def test_home_screen_q2_top_tech3(self):
        logger.warning("Run 3 of test_home_screen_q2_top_tech")
        self.test_home_screen_q2_top_tech()

    @skip_in_ci
    def test_home_screen_q2_top_tech(self):
        # changed from the original:
        # prompt = get_canned_prompt_text(prompt_id="identify_top_tech_stocks")
        """
        Identify the top 5 performing stocks in the technology sector over the past year,
        focusing on companies with a market cap above $10B.
        provide a summary of their recent financial performance and any significant news.
        """

        # changed to this to make it consistent and repeatable
        prompt = """Identify the top 5 performing stocks in the technology
        sector in 2023, focusing on companies with a market cap above $10B.
        provide a summary of their recent financial performance and any significant news.
        """

        def validate_output(prompt: str, output: IOType):
            return

        def validate_tool_args(execution_log: DefaultDict[str, List[dict]]):
            # Tool - prepare_output -> There are 3 sections
            tool_name = "prepare_output"
            self.assertTrue(tool_name in execution_log.keys())
            self.assertTrue(len(execution_log[tool_name]) == 3)

        # we run it 3 times above in parallel instead of this loop
        max_runs = 1
        for idx in range(max_runs):
            logger.warning(f"Run {idx + 1} of {max_runs}\nprompt={prompt}")
            self.prompt_test(
                prompt=prompt,
                validate_output=validate_output,
                required_tools=[
                    "get_stock_universe",
                    "sector_identifier_lookup",
                    "gics_sector_industry_filter",
                    "get_statistic_data_for_companies",
                    "transform_table",
                    "get_stock_identifier_list_from_table",
                    "get_all_news_developments_about_companies",
                    "get_earnings_call_summaries",
                    "get_10k_10q_sec_filings",
                    "get_default_text_data_for_stocks",
                    "summarize_texts",
                    "prepare_output",
                ],
                validate_tool_args=validate_tool_args,
            )

    # we run this test 3 times to catch any randomness
    # use duplicated test so that they can be run in parallel
    @skip_in_ci
    def test_home_screen_q3_COST_developments2(self):
        logger.warning("Run 2 of test_home_screen_q3_COST_developments")
        self.test_home_screen_q3_COST_developments()

    @skip_in_ci
    def test_home_screen_q3_COST_developments3(self):
        logger.warning("Run 3 of test_home_screen_q3_COST_developments")
        self.test_home_screen_q3_COST_developments()

    @skip_in_ci
    def test_home_screen_q3_COST_developments(self):
        # this should be renamed to:
        # test_home_screen_q3_AAPL_developments(self):
        # This original is needlessly slow for testing purposes
        """
        Summarize all the major developments for COST over the past year.
        Focus your analysis on corporate filings and earnings calls.
        Show the developments in point form as a timeline with dates.
        Bold anything important. For each development mention if it
        is positive or negative and why it is significant to COST.
        """

        # Use last quarter instead
        # switch to apple that actually has data on dev.
        prompt = """
        Summarize all the major developments for Apple over the last quarter.
        Focus your analysis on corporate filings and earnings calls.
        Show the developments in point form as a timeline with dates.
        Bold anything important. For each development mention if it
        is positive or negative and why it is significant to AAPL.
        """

        def validate_output(prompt: str, output: IOType):
            output_text = get_output(output=output)
            self.assertTrue(len(output_text.val.strip()) > 0)
            self.assertTrue("\n- " in output_text.val, msg="not bullet points in markdown")
            self.assertTrue("**" in output_text.val, msg="no bold in markdown")

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
            exp_tool_arg = "COST"
            self.assertEqual(
                ref_tool_arg,
                exp_tool_arg,
                f"Args tool {tool_name} - {arg_name}",
            )

        # duplicated the test 2 times so we can run this in parallel instead
        max_runs = 1
        for idx in range(max_runs):
            logger.warning(f"Run {idx + 1} of {max_runs}\nprompt={prompt}")
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

    # we run this test 3 times to catch any randomness
    # use duplicated test so that they can be run in parallel
    @skip_in_ci
    def test_home_screen_q4_netlfix_earnings_call2(self):
        logger.warning("Run 2 of test_home_screen_q4_netlfix_earnings_call")
        self.test_home_screen_q4_netlfix_earnings_call()

    @skip_in_ci
    def test_home_screen_q4_netlfix_earnings_call3(self):
        logger.warning("Run 3 of test_home_screen_q4_netlfix_earnings_call")
        self.test_home_screen_q4_netlfix_earnings_call()

    @skip_in_ci
    def test_home_screen_q4_netlfix_earnings_call(self):
        # actually AAPL now
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

        max_runs = 1
        for idx in range(max_runs):
            logger.warning(f"Run {idx + 1} of {max_runs}\nprompt={prompt}")
            self.prompt_test(
                prompt=prompt,
                validate_output=validate_output,
                required_tools=[
                    "stock_identifier_lookup",
                    "get_earnings_call_summaries",
                    "get_latest_news_for_companies",
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
            print(f"Run {idx + 1} of {max_runs}\nprompt={prompt}")
            self.prompt_test(
                prompt=prompt,
                validate_output=validate_output,
                required_tools=[],
                validate_tool_args=validate_tool_args,
            )

    # we run this test 3 times to catch any randomness
    # use duplicated test so that they can be run in parallel
    @skip_in_ci
    def test_home_screen_q6_hypothesis_leader2(self):
        logger.warning("Run 2 of test_home_screen_q6_hypothesis_leader")
        self.test_home_screen_q6_hypothesis_leader()

    @skip_in_ci
    def test_home_screen_q6_hypothesis_leader3(self):
        logger.warning("Run 3 of test_home_screen_q6_hypothesis_leader")
        self.test_home_screen_q6_hypothesis_leader()

    @skip_in_ci
    def test_home_screen_q6_hypothesis_leader(self):
        prompt = (
            "Is Nintendo a leader with millennial gamers amongst the following"
            " companies, microsoft, sony, taketwo, capcom, electronic arts, and"
            " ubisoft?"
        )

        def validate_output(prompt: str, output: IOType):
            output_text = get_output(output=output)
            self.assertTrue(len(output_text.val.strip()) > 0)

        def validate_tool_args(execution_log: DefaultDict[str, List[dict]]):
            # Tool - stock_identifier_lookup
            tool_name = "stock_identifier_lookup"
            ref_tool_arg = []
            if tool_name in execution_log.keys():
                arg_name = "stock_name"
                ref_tool_arg = [execution_log[tool_name][0][arg_name]]
            # Tool - multi_stock_identifier_lookup (only care about total univ)
            tool_name = "multi_stock_identifier_lookup"
            self.assertTrue(tool_name in execution_log.keys())
            arg_name = "stock_names"
            ref_tool_arg.extend(execution_log[tool_name][0][arg_name])
            ref_tool_arg = set(ref_tool_arg)
            exp_tool_arg = set(
                [
                    "nintendo",
                    "microsoft",
                    "sony",
                    "taketwo",
                    "capcom",
                    "electronic arts",
                    "ubisoft",
                ]
            )
            self.assertSetEqual(
                ref_tool_arg,
                exp_tool_arg,
                f"Args tool {tool_name} - {arg_name}",
            )
            # Tool - prepare_output -> There are 3 sections
            tool_name = "prepare_output"
            self.assertTrue(tool_name in execution_log.keys())
            self.assertTrue(len(execution_log[tool_name]) == 3)

        max_runs = 1
        for idx in range(max_runs):
            logger.warning(f"Run {idx + 1} of {max_runs}\nprompt={prompt}")
            self.prompt_test(
                prompt=prompt,
                validate_output=validate_output,
                required_tools=[
                    # "stock_identifier_lookup",
                    "multi_stock_identifier_lookup",
                    "get_company_descriptions",
                    "get_all_news_developments_about_companies",
                    "get_earnings_call_summaries",
                    "get_10k_10q_sec_filings",
                    "get_default_text_data_for_stocks",
                    "get_criteria_for_competitive_analysis",
                    "do_competitive_analysis",
                    "generate_summary_for_competitive_analysisprepare_output",
                ],
                validate_tool_args=validate_tool_args,
            )
