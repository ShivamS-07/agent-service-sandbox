# type: ignore
import unittest
from typing import DefaultDict, List

from agent_service.io_type_utils import IOType
from agent_service.io_types.stock import StockID
from agent_service.io_types.text import (
    Text,
)
from regression_test.test_regression import TestExecutionPlanner, get_output, skip_in_ci


# todo: simonlee9504 - add back fiscal earnings tests
class FiscalEarningsDate:
    pass


class TestEarnings(TestExecutionPlanner):
    @skip_in_ci
    def test_get_latest_earnings_summary(self) -> None:
        prompt = "Summarize latest earnings report from Apple"

        def validate_output(prompt: str, output: IOType) -> None:
            output_text = get_output(output=output)
            if isinstance(output_text, list) and len(output_text) > 0:
                output_text = output_text[0]
            self.assertTrue(isinstance(output_text, Text))
            self.assertGreater(len(output_text.val), 0)

        def validate_tool_args(execution_log: DefaultDict[str, List[dict]]) -> None:
            tool_name = "get_earnings_call_summaries"

            args = execution_log[tool_name][0]
            stock_input = args["stock_ids"]

            # validate using default daterange when prompted for latest filings
            self.assertTrue("date_range" not in args or args["date_range"] is None)
            # expected args passed
            self.assertTrue(len(stock_input) == 1)

        self.prompt_test(
            prompt=prompt,
            validate_output=validate_output,
            required_tools=[
                "get_date_range",
                "get_earnings_call_summaries",
            ],
            validate_tool_args=validate_tool_args,
            raise_output_validation_error=True,
        )

    @skip_in_ci
    def test_get_latest_full_earnings_transcript(self) -> None:
        prompt = "Retrieve the latest full earnings call transcript from Apple"

        def validate_output(prompt: str, output: IOType) -> None:
            output_text = get_output(output=output)
            if isinstance(output_text, list) and len(output_text) > 0:
                output_text = output_text[0]
            self.assertTrue(isinstance(output_text, Text))
            self.assertGreater(len(output_text.val), 0)

        def validate_tool_args(execution_log: DefaultDict[str, List[dict]]) -> None:
            tool_name = "get_earnings_call_full_transcripts"

            args = execution_log[tool_name][0]
            stock_input = args["stock_ids"]

            # validate using default daterange when prompted for latest filings
            self.assertTrue("date_range" not in args or args["date_range"] is None)
            # expected args passed
            self.assertTrue(len(stock_input) == 1)

        self.prompt_test(
            prompt=prompt,
            validate_output=validate_output,
            required_tools=[
                "get_date_range",
                "get_earnings_call_full_transcripts",
            ],
            validate_tool_args=validate_tool_args,
            raise_output_validation_error=True,
        )

    @skip_in_ci
    def test_get_earnings_summary_with_daterange(self) -> None:
        prompt = "Summarize the Q1 2023 earnings report from Apple"

        def validate_output(prompt: str, output: IOType) -> None:
            output_text = get_output(output=output)
            if isinstance(output_text, list) and len(output_text) > 0:
                output_text = output_text[0]
            self.assertTrue(isinstance(output_text, Text))
            self.assertGreater(len(output_text.val), 0)

        def validate_tool_args(execution_log: DefaultDict[str, List[dict]]) -> None:
            tool_name = "get_earnings_call_summaries"

            args = execution_log[tool_name][0]
            stock_input = args["stock_ids"]
            daterange_input = args["date_range"]

            # validate daterange passed in
            self.assertTrue(daterange_input is not None)
            # expected args passed
            self.assertTrue(len(stock_input) == 1)

        self.prompt_test(
            prompt=prompt,
            validate_output=validate_output,
            required_tools=[
                "get_date_range",
                "get_earnings_call_summaries",
            ],
            validate_tool_args=validate_tool_args,
            raise_output_validation_error=True,
        )

    @skip_in_ci
    def test_get_earnings_summary_for_last_year(self) -> None:
        prompt = "Summarize earnings reports for Apple from last year"

        def validate_output(prompt: str, output: IOType) -> None:
            output_text = get_output(output=output)
            if isinstance(output_text, list) and len(output_text) > 0:
                output_text = output_text[0]
            self.assertTrue(isinstance(output_text, Text))
            self.assertGreater(len(output_text.val), 0)

        def validate_tool_args(execution_log: DefaultDict[str, List[dict]]) -> None:
            tool_name = "get_earnings_call_summaries"

            args = execution_log[tool_name][0]
            stock_input = args["stock_ids"]
            daterange_input = args["date_range"]

            # validate daterange passed in
            self.assertTrue(daterange_input is not None)
            # expected args passed
            self.assertTrue(len(stock_input) == 1)

        self.prompt_test(
            prompt=prompt,
            validate_output=validate_output,
            required_tools=[
                "get_date_range",
                "get_earnings_call_summaries",
            ],
            validate_tool_args=validate_tool_args,
            raise_output_validation_error=True,
        )

    # todo: QL24-3528
    @unittest.skip("skipping until we explicitly support last x in earnings")
    @skip_in_ci
    def test_get_earnings_summary_last_x(self) -> None:
        prompt = "Summarize last 4 earnings reports for Apple"

        def validate_output(prompt: str, output: IOType) -> None:
            output_text = get_output(output=output)
            if isinstance(output_text, list) and len(output_text) > 0:
                output_text = output_text[0]
            self.assertTrue(isinstance(output_text, Text))
            self.assertGreater(len(output_text.val), 0)

        def validate_tool_args(execution_log: DefaultDict[str, List[dict]]) -> None:
            tool_name = "get_earnings_call_summaries"

            args = execution_log[tool_name][0]
            stock_input = args["stock_ids"]
            daterange_input = args["date_range"]

            # validate daterange passed in
            self.assertTrue(daterange_input is not None)
            # expected args passed
            self.assertTrue(len(stock_input) == 1)

        self.prompt_test(
            prompt=prompt,
            validate_output=validate_output,
            required_tools=[
                "get_date_range",
                "get_earnings_call_summaries",
            ],
            validate_tool_args=validate_tool_args,
            raise_output_validation_error=True,
        )

    @skip_in_ci
    def test_get_earnings_summary_multiple(self) -> None:
        prompt = "Summarize Q3 2023, Q2 2023, Q1 2023 earnings reports for Apple"

        def validate_output(prompt: str, output: IOType) -> None:
            output_text = get_output(output=output)
            if isinstance(output_text, list) and len(output_text) > 0:
                output_text = output_text[0]
            self.assertTrue(isinstance(output_text, Text))
            self.assertGreater(len(output_text.val), 0)

        def validate_tool_args(execution_log: DefaultDict[str, List[dict]]) -> None:
            tool_name = "get_earnings_call_summaries"

            args = execution_log[tool_name][0]
            stock_input = args["stock_ids"]
            daterange_input = args["date_range"]

            # validate daterange passed in
            self.assertTrue(daterange_input is not None)
            # expected args passed
            self.assertTrue(len(stock_input) == 1)

        self.prompt_test(
            prompt=prompt,
            validate_output=validate_output,
            required_tools=[
                "get_date_range",
                "get_earnings_call_summaries",
            ],
            validate_tool_args=validate_tool_args,
            raise_output_validation_error=True,
        )

    @unittest.skip("Fiscal support not implemented")
    @skip_in_ci
    def test_get_earnings_summary_fiscal_single(self) -> None:
        prompt = "Summarize fiscal Q1 2023 earnings report for Apple"

        def validate_output(prompt: str, output: IOType) -> None:
            output_text = get_output(output=output)
            if isinstance(output_text, list) and len(output_text) > 0:
                output_text = output_text[0]
            self.assertTrue(isinstance(output_text, Text))
            self.assertGreater(len(output_text.val), 0)

        def validate_tool_args(execution_log: DefaultDict[str, List[dict]]) -> None:
            tool_name = "get_earnings_call_summaries_by_fiscal_calendar_for_stock_id"

            args = execution_log[tool_name][0]
            stock_input = args["stock_id"]
            fiscal_earnings_dates = args["fiscal_earnings_dates"]

            # validate fiscal dates
            self.assertTrue(fiscal_earnings_dates is not None)
            self.assertIsInstance(fiscal_earnings_dates, list)
            self.assertTrue(len(fiscal_earnings_dates) > 0)
            fiscal_date = fiscal_earnings_dates[0]
            self.assertIsInstance(fiscal_date, FiscalEarningsDate)
            self.assertEqual(fiscal_date.year, 2023)
            self.assertEqual(fiscal_date.quarter, 1)

            # expected args passed
            self.assertIsInstance(stock_input, StockID)
            self.assertTrue(stock_input.gbi_id, 714)

        self.prompt_test(
            prompt=prompt,
            validate_output=validate_output,
            required_tools=[
                "get_fiscal_earnings_dates"
                "get_earnings_call_summaries_by_fiscal_calendar_for_stock_id",
            ],
            validate_tool_args=validate_tool_args,
            raise_output_validation_error=True,
        )

    @unittest.skip("Fiscal support not implemented")
    @skip_in_ci
    def test_get_earnings_summary_fiscal_multiple(self) -> None:
        prompt = "Summarize fiscal 3Q23, 2Q23, 1Q23 earnings report for Apple"

        def validate_output(prompt: str, output: IOType) -> None:
            output_text = get_output(output=output)
            if isinstance(output_text, list) and len(output_text) > 0:
                output_text = output_text[0]
            self.assertTrue(isinstance(output_text, Text))
            self.assertGreater(len(output_text.val), 0)

        def validate_tool_args(execution_log: DefaultDict[str, List[dict]]) -> None:
            tool_name = "get_earnings_call_summaries_by_fiscal_calendar_for_stock_id"

            args = execution_log[tool_name][0]
            stock_input = args["stock_id"]
            fiscal_earnings_dates = args["fiscal_earnings_dates"]

            # validate fiscal dates
            self.assertTrue(fiscal_earnings_dates is not None)
            self.assertIsInstance(fiscal_earnings_dates, list)
            self.assertTrue(len(fiscal_earnings_dates) == 3)
            fiscal_date = fiscal_earnings_dates[0]
            self.assertIsInstance(fiscal_date, FiscalEarningsDate)
            self.assertEqual(fiscal_date.year, 2023)

            # expected args passed
            self.assertIsInstance(stock_input, StockID)
            self.assertTrue(stock_input.gbi_id, 714)

        self.prompt_test(
            prompt=prompt,
            validate_output=validate_output,
            required_tools=[
                "get_fiscal_earnings_dates"
                "get_earnings_call_summaries_by_fiscal_calendar_for_stock_id",
            ],
            validate_tool_args=validate_tool_args,
            raise_output_validation_error=True,
        )
