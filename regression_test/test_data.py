# type: ignore
from datetime import date

from agent_service.io_type_utils import IOType, TableColumnType
from agent_service.io_types.text import Text
from regression_test.test_regression import TestExecutionPlanner, get_output
from regression_test.util import (
    compare_with_expected_text,
    validate_line_graph,
    validate_table_and_get_columns,
    validate_text,
)


class TestData(TestExecutionPlanner):
    def test_relative_strength_2023(self):
        prompt = "Show me Relative Strength Index for NVDA, AMD, INTL and GOOG over the year 2023"

        def validate_output(prompt: str, output: IOType):
            output_line_graph = get_output(output=output)
            validate_line_graph(output_line_graph=output_line_graph)
            nvda_points = output_line_graph.data[0].points
            amd_points = output_line_graph.data[1].points
            intl_points = output_line_graph.data[2].points
            goog_points = output_line_graph.data[3].points
            self.assertEqual(len(nvda_points), 260)
            self.assertAlmostEqual(
                next(
                    point for point in nvda_points if point.x_val == date(year=2023, month=5, day=1)
                ).y_val,
                65.8765487671,
                places=3,
            )
            self.assertEqual(len(amd_points), 260)
            self.assertEqual(len(intl_points), 260)
            self.assertAlmostEqual(
                next(
                    point
                    for point in nvda_points
                    if point.x_val == date(year=2023, month=9, day=11)
                ).y_val,
                46.0734291077,
                places=3,
            )
            self.assertEqual(len(goog_points), 260)

        self.prompt_test(
            prompt=prompt,
            validate_output=validate_output,
            required_tools=["get_statistic_data_for_companies"],
        )

    def test_plot_tsla_price_jan_to_march(self):
        prompt = "plot tsla price from Jan 2024 to March 2024"

        def validate_output(prompt: str, output: IOType):
            output_line_graph = get_output(output)
            validate_line_graph(output_line_graph=output_line_graph)
            actual_points = output_line_graph.data[0].points
            self.assertTrue(len(actual_points), 65)
            expected_max_price = 248.47999572753906
            expected_max_price_date = date(year=2024, month=1, day=1)

            expected_min_price = 162.5
            expected_min_price_date = date(year=2024, month=3, day=14)

            actual_min_price_point = min(actual_points, key=lambda x: x.y_val)
            actual_max_price_point = max(actual_points, key=lambda x: x.y_val)
            self.assertEqual(expected_min_price_date, actual_min_price_point.x_val)
            self.assertEqual(expected_max_price_date, actual_max_price_point.x_val)
            self.assertAlmostEqual(expected_min_price, actual_min_price_point.y_val, places=3)
            self.assertAlmostEqual(expected_max_price, actual_max_price_point.y_val, places=3)

        self.prompt_test(
            prompt=prompt,
            validate_output=validate_output,
            required_tools=["get_statistic_data_for_companies"],
        )

    def test_intersection_of_qqq_xlv_jan_2024(self):
        prompt = "Find the intersection of QQQ and XLV on Jan 1, 2024"

        def validate_output(prompt: str, output: IOType):
            output_stock_ids = get_output(output=output)
            self.assertEqual(len(output_stock_ids), 11)
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
            self.assertEqual(actual_stock_ids, expected_stock_ids)

        self.prompt_test(
            prompt=prompt,
            validate_output=validate_output,
            required_tools=["get_stock_universe", "intersect_lists"],
        )

    def test_top_mcap_april_2024(self):
        prompt = "top 10 by market cap today, and then graph their market caps over the month of April 2024"

        def validate_output(prompt: str, output: IOType):
            output_line_graph = get_output(output=output[1])
            validate_line_graph(output_line_graph=output_line_graph)
            self.assertEqual(len(output_line_graph.data), 10)
            self.assertEqual(len(output_line_graph.data[0].points), 2)

        self.prompt_test(
            prompt=prompt,
            validate_output=validate_output,
            required_tools=["get_stock_universe", "get_statistic_data_for_companies"],
        )

    def test_tech_sector_stocks(self):
        prompt = "Find stocks in the technology sector on Jan 10, 2024"

        def validate_output(prompt: str, output: IOType):
            output_stock_ids = get_output(output=output)

            self.assertEqual(len(output_stock_ids), 65)
            expected_ids = [
                124,
                155,
                691,
                713,
                714,
                716,
                719,
                723,
                1144,
                1694,
                2271,
                2849,
                3428,
                4043,
                4083,
                4569,
                5112,
                5136,
                5721,
                5757,
                5766,
                6344,
                6384,
                6387,
                6960,
                6961,
                6963,
                7504,
                7528,
                7551,
                7555,
                8154,
                8707,
                9292,
                10817,
                10865,
                10931,
                11554,
                11595,
                11635,
                12293,
                12299,
                12985,
                13831,
                14424,
                14426,
                15014,
                15315,
                18851,
                18854,
                19729,
                21466,
                22901,
                26805,
                27375,
                28309,
                28385,
                29336,
                29372,
                30055,
                30940,
                31120,
                31767,
                35692,
                514112,
            ]
            self.assertEqual(
                sorted([stock_id.gbi_id for stock_id in output_stock_ids]), expected_ids
            )

        self.prompt_test(
            prompt=prompt, validate_output=validate_output, required_tools=["sector_filter"]
        )

    def test_pe_nvda_feb_2024(self):
        prompt = "Show me the PE of NVDA over month of Feb 2024?"

        def validate_output(prompt: str, output: IOType):
            output_stock_table = get_output(output)
            date_column, pe_column = validate_table_and_get_columns(
                output_stock_table=output_stock_table,
                column_types=[TableColumnType.DATE, TableColumnType.FLOAT],
            )
            self.assertEqual(len(date_column.data), 21)
            self.assertEqual(len(pe_column.data), 21)
            self.assertAlmostEqual(pe_column.data[3], 89.21580505371094, places=3)
            self.assertAlmostEqual(pe_column.data[7], 94.47933197021484, places=3)

        self.prompt_test(
            prompt=prompt,
            validate_output=validate_output,
            required_tools=["get_statistic_data_for_companies"],
        )

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

    def test_open_close_spread_jan_2024(self):
        prompt = "Calculate the spread between open and close for AAPL over the month of Jan 2024"

        def validate_output(prompt: str, output: IOType):
            output_stock_table = get_output(output)
            date_column = validate_table_and_get_columns(
                output_stock_table=output_stock_table, column_types=[TableColumnType.DATE]
            )[0]
            self.assertEqual(len(date_column.data), 23)

        self.prompt_test(
            prompt=prompt,
            validate_output=validate_output,
            required_tools=["get_statistic_data_for_companies"],
        )

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

    def test_notify_big_big_developments_June_2024(self):
        prompt = (
            "Scan all corporate filings for LIPO until June 2024 and notify me of any big developments or "
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

    def test_notify_big_big_developments(self):
        prompt = (
            "Scan all corporate filings for LIPO and notify me of any big developments or "
            "changes to cash flow"
        )

        def validate_output(prompt: str, output: IOType):
            output_text = get_output(output)
            self.assertTrue(isinstance(output_text, Text)), "Output is not of type Text"
            self.assertTrue(output_text.val), "Expected non empty string"

        self.prompt_test(
            prompt=prompt,
            validate_output=validate_output,
            raise_plan_validation_error=True,
            required_tools=["get_10k_10q_sec_filings"],
        )
