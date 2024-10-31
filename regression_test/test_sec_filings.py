# type: ignore
from typing import DefaultDict, List

from agent_service.io_type_utils import IOType
from agent_service.io_types.text import (
    StockOtherSecFilingText,
    StockSecFilingText,
    Text,
)
from agent_service.utils.sec.constants import FILE_10K, FILE_10Q
from regression_test.test_regression import TestExecutionPlanner, get_output, skip_in_ci


class TestSecFilings(TestExecutionPlanner):
    # skip in CI since SEC filings prompts usually take a long time
    @skip_in_ci
    def test_get_10k_10q_sec_filings(self) -> None:
        # tests basic SEC retrieval

        prompt = "Summarize the latest SEC filings from AAPL"

        def validate_output(prompt: str, output: IOType) -> None:
            output_text = get_output(output=output)
            self.assertIsInstance(output_text, Text)
            self.assertGreater(len(output_text.val), 0)

        def validate_tool_args(execution_log: DefaultDict[str, List[dict]]) -> None:
            tool_name = "get_10k_10q_sec_filings"

            args = execution_log[tool_name][0]
            stock_inputs = args["stock_ids"]

            # validate stock in input
            self.assertEqual(len(stock_inputs), 1)

            # validate using default daterange when prompted for latest filings
            self.assertTrue("date_range" not in args or args["date_range"] is None)
            # expected args passed
            self.assertTrue("must_include_10q" not in args or args["must_include_10q"] is True)
            self.assertTrue("must_include_10k" not in args or args["must_include_10k"] is True)

            tool_name = "summarize_texts"
            args = execution_log[tool_name][0]
            self.assertTrue("texts" in args and len(args["texts"]) >= 2)
            self.assertTrue(
                len(
                    [
                        sec_filing.form_type
                        for sec_filing in args["texts"]
                        if isinstance(sec_filing, StockSecFilingText)
                        and sec_filing.form_type == FILE_10K
                    ]
                )
                != 0
            )

        self.prompt_test(
            prompt=prompt,
            validate_output=validate_output,
            required_tools=[
                "get_10k_10q_sec_filings",
            ],
            validate_tool_args=validate_tool_args,
            raise_output_validation_error=True,
        )

    @skip_in_ci
    def test_get_10q_sec_filings(self) -> None:
        # tests SEC retrieval only 10-Q

        prompt = "Summarize only the 10-Q SEC filings from AAPL within the last 8 months"

        def validate_output(prompt: str, output: IOType) -> None:
            output_text = get_output(output=output)
            self.assertIsInstance(output_text, Text)
            self.assertGreater(len(output_text.val), 0)

        def validate_tool_args(execution_log: DefaultDict[str, List[dict]]) -> None:
            tool_name = "get_10k_10q_sec_filings"

            args = execution_log[tool_name][0]
            stock_inputs = args["stock_ids"]

            # validate stock in input
            self.assertEqual(len(stock_inputs), 1)
            self.assertTrue("date_range" in args)

            # expected args passed
            self.assertTrue("must_include_10q" not in args or args["must_include_10q"] is True)
            self.assertTrue("must_include_10k" in args and args["must_include_10k"] is False)

            tool_name = "summarize_texts"
            args = execution_log[tool_name][0]
            self.assertTrue("texts" in args and len(args["texts"]) >= 2)
            self.assertTrue(
                len(
                    [
                        sec_filing.form_type
                        for sec_filing in args["texts"]
                        if isinstance(sec_filing, StockSecFilingText)
                        and sec_filing.form_type == FILE_10K
                    ]
                )
                == 0
            )

        self.prompt_test(
            prompt=prompt,
            validate_output=validate_output,
            required_tools=[
                "get_10k_10q_sec_filings",
                "summarize_texts",
            ],
            validate_tool_args=validate_tool_args,
            raise_output_validation_error=True,
        )

    @skip_in_ci
    def test_get_10k_sec_filings(self) -> None:
        # tests SEC retrieval only 10-K

        prompt = "Summarize only the 10-K SEC filings from AAPL within the last 1.5 years"

        def validate_output(prompt: str, output: IOType) -> None:
            output_text = get_output(output=output)
            self.assertIsInstance(output_text, Text)
            self.assertGreater(len(output_text.val), 0)

        def validate_tool_args(execution_log: DefaultDict[str, List[dict]]) -> None:
            tool_name = "get_10k_10q_sec_filings"

            args = execution_log[tool_name][0]
            stock_inputs = args["stock_ids"]

            # validate stock in input
            self.assertEqual(len(stock_inputs), 1)
            self.assertTrue("date_range" in args)

            # expected args passed
            self.assertTrue("must_include_10q" in args and args["must_include_10q"] is False)
            self.assertTrue("must_include_10k" not in args or args["must_include_10k"] is True)

            tool_name = "summarize_texts"
            args = execution_log[tool_name][0]
            self.assertTrue("texts" in args and len(args["texts"]) >= 1)
            self.assertTrue(
                len(
                    [
                        sec_filing.form_type
                        for sec_filing in args["texts"]
                        if isinstance(sec_filing, StockSecFilingText)
                        and sec_filing.form_type == FILE_10Q
                    ]
                )
                == 0
            )

        self.prompt_test(
            prompt=prompt,
            validate_output=validate_output,
            required_tools=[
                "get_10k_10q_sec_filings",
                "summarize_texts",
            ],
            validate_tool_args=validate_tool_args,
            raise_output_validation_error=True,
        )

    @skip_in_ci
    def test_get_non_10k_10q_sec_filings(self) -> None:
        # tests other filing type SEC retrieval, exclude 10-K, 10-Q

        prompt = "Summarize 8-K filings from AAPL within the last year, do not include 10K or 10Q filings"

        def validate_output(prompt: str, output: IOType) -> None:
            output_text = get_output(output=output)
            self.assertIsInstance(output_text, Text)
            self.assertGreater(len(output_text.val), 0)

        def validate_tool_args(execution_log: DefaultDict[str, List[dict]]) -> None:
            tool_name = "get_sec_filings_with_type"

            args = execution_log[tool_name][0]
            stock_inputs = args["stock_ids"]

            # validate stock in input
            self.assertEqual(len(stock_inputs), 1)
            self.assertTrue("date_range" in args)

            tool_name = "summarize_texts"
            args = execution_log[tool_name][0]
            self.assertTrue("texts" in args and len(args["texts"]) >= 1)
            self.assertTrue(
                len(
                    [
                        sec_filing.form_type
                        for sec_filing in args["texts"]
                        if isinstance(sec_filing, StockOtherSecFilingText)
                        and (sec_filing.form_type == FILE_10Q or sec_filing.form_type == FILE_10K)
                    ]
                )
                == 0
            )
            self.assertTrue(
                len(
                    [
                        sec_filing.form_type
                        for sec_filing in args["texts"]
                        if isinstance(sec_filing, StockOtherSecFilingText)
                        and sec_filing.form_type == "8-K"
                    ]
                )
                >= 1
            )

        self.prompt_test(
            prompt=prompt,
            validate_output=validate_output,
            required_tools=[
                "sec_filings_type_lookup",
                "get_sec_filings_with_type",
                "summarize_texts",
            ],
            validate_tool_args=validate_tool_args,
            raise_output_validation_error=True,
        )

    @skip_in_ci
    def test_get_10q_8k_sec_filings(self) -> None:
        # tests other filing type SEC retrieval, exclude 10-K, 10-Q

        prompt = "Summarize 10-Q, 8-K SEC filings for the last year from AAPL"

        def validate_output(prompt: str, output: IOType) -> None:
            output_text = get_output(output=output)
            self.assertIsInstance(output_text, Text)
            self.assertGreater(len(output_text.val), 0)

        def validate_tool_args(execution_log: DefaultDict[str, List[dict]]) -> None:
            tool_name = "get_sec_filings_with_type"

            args = execution_log[tool_name][0]
            stock_inputs = args["stock_ids"]

            # validate stock in input
            self.assertEqual(len(stock_inputs), 1)
            self.assertTrue("date_range" in args)

            tool_name = "summarize_texts"
            args = execution_log[tool_name][0]
            self.assertTrue("texts" in args and len(args["texts"]) >= 1)
            self.assertTrue(
                len(
                    [
                        sec_filing.form_type
                        for sec_filing in args["texts"]
                        if isinstance(sec_filing, StockSecFilingText)
                        and (sec_filing.form_type == FILE_10Q)
                    ]
                )
                >= 1
            )
            self.assertTrue(
                len(
                    [
                        sec_filing.form_type
                        for sec_filing in args["texts"]
                        if isinstance(sec_filing, StockSecFilingText)
                        and (sec_filing.form_type == FILE_10K)
                    ]
                )
                == 0
            )
            self.assertTrue(
                len(
                    [
                        sec_filing.form_type
                        for sec_filing in args["texts"]
                        if isinstance(sec_filing, StockOtherSecFilingText)
                        and sec_filing.form_type == "8-K"
                    ]
                )
                >= 1
            )

        self.prompt_test(
            prompt=prompt,
            validate_output=validate_output,
            required_tools=[
                "sec_filings_type_lookup",
                "get_sec_filings_with_type",
                "summarize_texts",
            ],
            validate_tool_args=validate_tool_args,
            raise_output_validation_error=True,
        )
