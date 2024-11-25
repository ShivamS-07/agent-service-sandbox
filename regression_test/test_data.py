# type: ignore
from datetime import date
from typing import DefaultDict, List

from agent_service.io_type_utils import IOType, TableColumnType
from agent_service.io_types.text import Text
from regression_test.test_regression import TestExecutionPlanner, get_output, skip_in_ci
from regression_test.util import (
    compare_with_expected_text,
    validate_line_graph,
    validate_table_and_get_columns,
    validate_text,
)


class TestData(TestExecutionPlanner):
    @skip_in_ci
    def test_relative_strength_2023(self):
        prompt = """
        Show me
        ```{"type": "variable", "id": "rsi_close_14", "label": "Relative Strength Index (Close, 14 days)"}```
        for NVDA, AMD, INTL and GOOG over the year 2023.
        """

        def validate_output(prompt: str, output: IOType):
            output_line_graph = get_output(output=output)
            validate_line_graph(output_line_graph=output_line_graph)
            data_points = {
                dataset.dataset_id.symbol: dataset.points for dataset in output_line_graph.data
            }
            nvda_data_points = data_points["NVDA"]
            amd_data_points = data_points["AMD"]
            intl_data_points = data_points["INTL"]
            goog_data_points = data_points["GOOG"]

            expected_number_of_data_points = [259, 261]
            number_of_data_points_mismatch_err_msg = "Number of data points for {stock} don't match"
            data_point_mismatch_err_msg = "Data point for NVDA on {date} does not match"
            self.assertIn(
                len(nvda_data_points),
                expected_number_of_data_points,
                msg=number_of_data_points_mismatch_err_msg.format(stock="NVDA"),
            )

            self.assertAlmostEqual(
                next(
                    point
                    for point in nvda_data_points
                    if point.x_val == date(year=2023, month=5, day=1)
                ).y_val,
                65.9,
                delta=1.0,
                msg=data_point_mismatch_err_msg.format(date=date(year=2023, month=5, day=1)),
            )
            self.assertIn(
                len(amd_data_points),
                expected_number_of_data_points,
                msg=number_of_data_points_mismatch_err_msg.format(stock="AMD"),
            )
            self.assertIn(
                len(intl_data_points),
                expected_number_of_data_points,
                msg=number_of_data_points_mismatch_err_msg.format(stock="INTL"),
            )
            self.assertAlmostEqual(
                next(
                    point
                    for point in nvda_data_points
                    if point.x_val == date(year=2023, month=9, day=11)
                ).y_val,
                46.1,
                delta=1.0,
                msg=data_point_mismatch_err_msg.format(date=date(year=2023, month=9, day=11)),
            )
            self.assertIn(
                len(goog_data_points),
                expected_number_of_data_points,
                msg=number_of_data_points_mismatch_err_msg.format(stock="GOOG"),
            )

        def validate_tool_args(execution_log: DefaultDict[str, List[dict]]):
            tool_name = "get_statistic_data_for_companies"
            actual_reference = execution_log[tool_name][0]["statistic_reference"]
            expected_reference = "Relative Strength Index"
            self.assertTrue(
                expected_reference in actual_reference,
                f"The {actual_reference=} for tool {tool_name} does not include {expected_reference=}",
            )

        self.prompt_test(
            prompt=prompt,
            validate_output=validate_output,
            required_tools=["get_statistic_data_for_companies"],
            validate_tool_args=validate_tool_args,
        )

    @skip_in_ci
    def test_plot_tsla_price_jan_to_march(self):
        prompt = "plot tsla price from Jan 2024 to March 2024"

        def validate_output(prompt: str, output: IOType):
            output_line_graph = get_output(output)
            validate_line_graph(output_line_graph=output_line_graph)
            actual_points = output_line_graph.data[0].points
            self.assertEqual(len(actual_points), 65, msg="Number of data points do not match")
            expected_max_price = 248.48
            expected_max_price_date = date(year=2024, month=1, day=1)

            expected_min_price = 162.50
            expected_min_price_date = date(year=2024, month=3, day=14)

            actual_min_price_point = min(actual_points, key=lambda x: x.y_val)
            actual_max_price_point = max(actual_points, key=lambda x: x.y_val)
            self.assertEqual(
                expected_min_price_date,
                actual_min_price_point.x_val,
                msg="Date with minimum price doesn't match",
            )
            self.assertEqual(
                expected_max_price_date,
                actual_max_price_point.x_val,
                msg="Date with maximum price doesn't match",
            )
            self.assertAlmostEqual(
                expected_min_price,
                actual_min_price_point.y_val,
                places=2,
                msg="Minimum price doesn't match",
            )
            self.assertAlmostEqual(
                expected_max_price,
                actual_max_price_point.y_val,
                places=2,
                msg="Maximum price doesn't match",
            )

        self.prompt_test(
            prompt=prompt,
            validate_output=validate_output,
            required_tools=["get_statistic_data_for_companies"],
        )

    @skip_in_ci
    def test_intersection_of_qqq_xlv_jan_2024(self):
        prompt = "Find the intersection of QQQ and XLV on Jan 1, 2024"

        def validate_output(prompt: str, output: IOType):
            output_stock_ids = get_output(output=output)
            actual_stock_ids = sorted(
                [output_stock_id.gbi_id for output_stock_id in output_stock_ids]
            )
            expected_stock_ids = [
                722,
                4605,
                5176,
                5177,
                5756,
                8700,
                12279,
                12993,
                15958,
                58434,
                610881,
            ]
            self.assertEqual(actual_stock_ids, expected_stock_ids, "Output stocks don't match")

        self.prompt_test(
            prompt=prompt,
            validate_output=validate_output,
            required_tools=["get_stock_universe", "intersect_lists"],
        )

    @skip_in_ci
    def test_top_mcap_april_2024(self):
        prompt = "top 10 by market cap today, and then graph their market caps over the month of April 2024"

        def validate_output(prompt: str, output: IOType):
            output_line_graph = get_output(output=output[1])
            validate_line_graph(output_line_graph=output_line_graph)
            self.assertEqual(
                len(output_line_graph.data), 10, msg="Number of stocks in graph don't match"
            )
            self.assertEqual(
                len(output_line_graph.data[0].points),
                22,
                msg="Number of data points for a stock don't match",
            )

        self.prompt_test(
            prompt=prompt,
            validate_output=validate_output,
            required_tools=["get_stock_universe", "get_statistic_data_for_companies"],
        )

    @skip_in_ci
    def test_consumer_discretionary_sector_stocks(self):
        # this sector was chosen since there aren't many similar sector names
        prompt = "Find stocks in the Consumer Discretionary sector in the QQQ on Jan 10, 2024"

        def validate_tool_args(execution_log: DefaultDict[str, List[dict]]) -> None:
            tool_name = "gics_sector_industry_filter"

            args = execution_log[tool_name][0]
            sector_id = args["sector_id"]

            # validate sector_id in input
            self.assertEqual(sector_id.sec_id, 25)

        def validate_output(prompt: str, output: IOType):
            output_stock_ids = get_output(output=output)

            expected_ids = [
                149,
                6976,
                7547,
                8712,
                9351,
                14438,
                19839,
                19864,
                25508,
                40168,
                498886,
                498965,
            ]
            self.assertEqual(
                sorted([stock_id.gbi_id for stock_id in output_stock_ids]),
                expected_ids,
                msg="Output stocks don't match",
            )

        self.prompt_test(
            prompt=prompt,
            validate_tool_args=validate_tool_args,
            validate_output=validate_output,
            required_tools=["sector_identifier_lookup", "gics_sector_industry_filter"],
        )

    @skip_in_ci
    def test_education_sector_etfs(self):
        # this sector was chosen since there aren't many similar sector names
        prompt = (
            "Find ETFs with at least 10% holdings in the Education Services sector on Jan 10, 2024"
        )

        def validate_tool_args(execution_log: DefaultDict[str, List[dict]]) -> None:
            tool_name = "gics_sector_industry_filter"

            args = execution_log[tool_name][0]
            sector_id = args["sector_id"]
            etf_min_sector_threshold = args["etf_min_sector_threshold"]

            # validate sector_id in input
            self.assertEqual(sector_id.sec_id, 25302010)
            self.assertEqual(etf_min_sector_threshold, 10.0)

        def validate_output(prompt: str, output: IOType):
            output_etf_ids = get_output(output=output)

            expected_ids = [18661, 26394, 413777, 430983, 499215, 508828, 523654]
            self.assertEqual(
                sorted([etf_id.gbi_id for etf_id in output_etf_ids]),
                expected_ids,
                msg="Output stocks don't match",
            )

        self.prompt_test(
            prompt=prompt,
            validate_tool_args=validate_tool_args,
            validate_output=validate_output,
            required_tools=["sector_identifier_lookup", "gics_sector_industry_filter"],
            raise_output_validation_error=True,
            raise_plan_validation_error=True,
        )

    @skip_in_ci
    def test_pe_nvda_feb_2024(self):
        prompt = "Show me the PE of NVDA over month of Feb 2024?"

        def validate_output(prompt: str, output: IOType):
            output_stock_table = get_output(output)
            date_column, pe_column = validate_table_and_get_columns(
                output_stock_table=output_stock_table,
                column_types=[TableColumnType.DATE, TableColumnType.FLOAT],
            )
            self.assertEqual(len(date_column.data), 21, "Number of dates don't match")
            self.assertEqual(len(pe_column.data), 21, "Number of data points don't match ")
            self.assertAlmostEqual(
                pe_column.data[3],
                89.9,
                delta=1.0,
                msg=f"Data point doesn't match on {date_column.data[3]}",
            )
            self.assertAlmostEqual(
                pe_column.data[7],
                95.2,
                delta=1.0,
                msg=f"Data point doesn't match on {date_column.data[7]}",
            )

        self.prompt_test(
            prompt=prompt,
            validate_output=validate_output,
            required_tools=["get_statistic_data_for_companies"],
        )

    @skip_in_ci
    def test_graph_pe_2023(self):
        prompt = "Graph the PE of health care stocks in QQQ over the year 2023"

        def validate_output(prompt: str, output: IOType):
            output_line_graph = get_output(output)
            validate_line_graph(output_line_graph=output_line_graph)
            self.assertGreaterEqual(len(output_line_graph.data[0].points), 260)

        self.prompt_test(
            prompt=prompt,
            validate_output=validate_output,
            required_tools=["get_statistic_data_for_companies"],
        )

    @skip_in_ci
    def test_open_close_spread_jan_2024(self):
        prompt = "Calculate the spread between open and close for AAPL over the month of Jan 2024"

        def validate_output(prompt: str, output: IOType):
            output_stock_table = get_output(output)
            date_column = validate_table_and_get_columns(
                output_stock_table=output_stock_table, column_types=[TableColumnType.DATE]
            )[0]
            self.assertEqual(len(date_column.data), 23, msg="Number of data points don't match")

        self.prompt_test(
            prompt=prompt,
            validate_output=validate_output,
            required_tools=["get_statistic_data_for_companies"],
        )

    @skip_in_ci
    def test_machine_learning_news_summary(self):
        prompt = (
            "Can you give me a single summary of news published in the last week of June 2024 about machine "
            "learning at Meta, Apple, and Microsoft?"
        )

        def validate_output(prompt: str, output: IOType):
            output_text = get_output(output)
            self.loop.run_until_complete(
                validate_text(llm=self.llm, output_text=output_text, prompt=prompt)
            )

        self.prompt_test(
            prompt=prompt,
            validate_output=validate_output,
            required_tools=["get_all_news_developments_about_companies"],
        )

    @skip_in_ci
    def test_notify_big_big_developments_June_2024(self):
        prompt = (
            "Scan corporate filings for LIPO until June 2024 and notify me of any big developments or "
            "changes to cash flow"
        )

        def validate_output(prompt: str, output: IOType):
            output_text = get_output(output)
            expected_text = (
                "Lipella Pharmaceuticals Inc. (LIPO) has experienced several significant developments and "
                "changes to its cash flow up until June 2024. The company has been actively progressing "
                "its lead product candidates, LP-10 for hemorrhagic cystitis (HC) and LP-310 for oral "
                "lichen planus (OLP), through various stages of clinical trials. Notably, LP-10 completed "
                "a phase 2a clinical trial, and LP-310 received FDA IND approval and orphan drug "
                "designation in late 2023. Financially, Lipella has been primarily funded through grants "
                "and equity financing. The company received a significant NIH grant in 2022, "
                "which was extended in 2023, providing a total of $1,353,000. This grant has been a key "
                "source of revenue, with $225,000 recognized in the first half of 2023. Additionally, "
                "Lipella completed an initial public offering (IPO) in December 2022, "
                "raising approximately $5 million. Operating expenses have increased substantially, "
                "primarily due to research and development (R&D) activities and costs associated with "
                "becoming a public company. R&D expenses rose due to clinical trial activities and stock "
                "option expenses, while general and administrative expenses increased due to higher costs "
                "for insurance, legal, and accounting services. The company has also engaged in financing "
                "activities, including a private placement in October 2023, raising approximately $2 "
                "million. Despite these efforts, Lipella continues to face challenges related to cash "
                "flow, with ongoing operating losses and a need for additional funding to support its "
                "clinical development programs and operational costs."
            )
            self.loop.run_until_complete(
                compare_with_expected_text(
                    llm=self.llm,
                    output_text=output_text,
                    prompt=prompt,
                    expected_text=expected_text,
                )
            )

        self.prompt_test(
            prompt=prompt,
            validate_output=validate_output,
            required_tools=["get_10k_10q_sec_filings"],
        )

    @skip_in_ci
    def test_notify_big_big_developments(self):
        prompt = (
            "Scan corporate filings for LIPO and notify me of any big developments or "
            "changes to cash flow"
        )

        def validate_output(prompt: str, output: IOType):
            output_text = get_output(output)
            self.assertTrue(
                isinstance(output_text, Text)
            ), f"Expected type: Text, Actual type: {type(output_text)}."
            self.assertTrue(output_text.val), "Expected non empty string"

        self.prompt_test(
            prompt=prompt,
            validate_output=validate_output,
            required_tools=["get_10k_10q_sec_filings"],
            raise_plan_validation_error=True,
        )

    @skip_in_ci
    def test_nvidia_general_peers(self):
        prompt = "Find me Nvidia's major competitors"

        def validate_output(prompt: str, output: IOType):
            output_stock_ids = get_output(output=output)

            expected_ids = {124, 5766, 8707}  # {AMD, INTEL, QUALCOMM}
            self.assertTrue(
                expected_ids.issubset(set([etf_id.gbi_id for etf_id in output_stock_ids])),
                msg="Competitors do not contain obvious ones",
            )

        self.prompt_test(
            prompt=prompt,
            validate_output=validate_output,
            required_tools=["get_general_peers"],
            raise_output_validation_error=True,
            raise_plan_validation_error=True,
        )
