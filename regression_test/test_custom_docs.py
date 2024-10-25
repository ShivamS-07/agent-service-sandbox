# type: ignore
from typing import DefaultDict, List

from agent_service.io_type_utils import IOType
from agent_service.io_types.text import Text
from regression_test.test_regression import TestExecutionPlanner, get_output, skip_in_ci


class TestCustomDocuments(TestExecutionPlanner):
    # TODO: jzhao - add regression test for a hypothesis query/profiler query that
    # pulls custom docs
    # custom doc user - dev only so disabling for now.
    # CUSTOM_DOC_DEV_TEST_USER = "515b61f7-38af-4826-ad32-0900b3b1b7d4"

    @skip_in_ci
    def test_custom_documents_by_topic(self) -> None:
        # two tests for the basic operation of custom docs tools

        prompt = (
            "From the 21 most relevant custom documents about climate change, summarize"
            " information about climate change and any adjacent macro effects in 2023"
        )

        def validate_output(prompt: str, output: IOType) -> None:
            output_text = get_output(output=output)
            self.assertIsInstance(output_text, Text)
            self.assertGreater(len(output_text.val), 0)

        def validate_tool_args(execution_log: DefaultDict[str, List[dict]]) -> None:
            tool_name = "get_user_custom_documents_by_topic"

            args = execution_log[tool_name][0]
            # validate the custom doc tool did not interpret 2023 as a date range filter
            self.assertTrue("date_range" not in args or args["date_range"] is None)
            # expected args passed
            self.assertEqual(args["top_n"], 21)
            self.assertTrue("climate" in args["topic"].lower())

        self.prompt_test(
            prompt=prompt,
            validate_output=validate_output,
            required_tools=[
                "get_user_custom_documents_by_topic",
            ],
            validate_tool_args=validate_tool_args,
        )

    @skip_in_ci
    def test_custom_documents_by_stock(self) -> None:
        prompt = (
            "Summarize all Q1 information from only custom documents about NVDA and WBA"
            " published in the last year"
        )

        def validate_output(prompt: str, output: IOType) -> None:
            output_text = get_output(output=output)
            self.assertIsInstance(output_text, Text)
            self.assertGreater(len(output_text.val), 0)

        def validate_tool_args(execution_log: DefaultDict[str, List[dict]]) -> None:
            tool_name = "get_user_custom_documents"

            args = execution_log[tool_name][0]
            stock_inputs = args["stock_ids"]

            # validate 2 stocks in input
            self.assertEqual(len(stock_inputs), 2)

            # validate the custom doc tool interpreted the last year filter correctly
            self.assertTrue("date_range" in args)
            self.assertAlmostEqual(
                (args["date_range"].end_date - args["date_range"].start_date).days,
                365,
                delta=22,
            )

        self.prompt_test(
            prompt=prompt,
            validate_output=validate_output,
            required_tools=[
                "get_user_custom_documents",
            ],
            validate_tool_args=validate_tool_args,
        )

    @skip_in_ci
    def test_custom_documents_by_filename(self) -> None:
        # Mostly, I am aiming to test the agent's ability to parse out naive file names.
        # To be removed once file picker is in place.
        prompt = (
            "From my custom docs titled:\n"
            # single quote
            "'20240425_BWG_Strategy_NVDA_Semi_AMD_Focused_Q-A_4_18_24_Master.pdf'\n"
            # doublequote
            '"Federal fund pours $66.5M into firms for greener concrete - The Logic.pdf"\n'
            # french quotes
            "« gme.html »\n"
            # unquoted
            "and also AMD_10Q.pdf and STZ_10Q.pdf,"
            "summarize the key points and present to me in a bullet point format."
        )

        def validate_output(prompt: str, output: IOType) -> None:
            output_text = get_output(output=output)
            self.assertIsInstance(output_text, Text)
            self.assertGreater(len(output_text.val), 0)

        def validate_tool_args(execution_log: DefaultDict[str, List[dict]]) -> None:
            tool_name = "get_user_custom_documents_by_filename"

            args = execution_log[tool_name][0]
            file_names = set(args["file_names"])

            # validate all mentinoed files in input
            exected_files = {
                "20240425_BWG_Strategy_NVDA_Semi_AMD_Focused_Q-A_4_18_24_Master.pdf",
                "Federal fund pours $66.5M into firms for greener concrete - The Logic.pdf",
                "gme.html",
                "AMD_10Q.pdf",
                "STZ_10Q.pdf",
            }
            self.assertSetEqual(
                file_names,
                exected_files,
            )

        self.prompt_test(
            prompt=prompt,
            validate_output=validate_output,
            required_tools=[
                "get_user_custom_documents_by_filename",
            ],
            validate_tool_args=validate_tool_args,
        )
