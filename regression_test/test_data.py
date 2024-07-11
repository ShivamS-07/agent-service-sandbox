# type: ignore
from agent_service.io_type_utils import IOType, TableColumnType
from regression_test.test_regression import (
    TestExecutionPlanner,
    get_output,
    validate_plan,
)
from regression_test.util import (
    compare_with_expected_text,
    validate_line_graph,
    validate_table_and_get_columns,
    validate_text,
)


class TestData(TestExecutionPlanner):
    def test_relative_strength(self):
        prompt = "Show me Relative Strength Index for NVDA, AMD, INTL and GOOG over the year 2023"

        def validate_output(prompt: str, output: IOType):
            output_line_graph = get_output(output=output)
            validate_line_graph(output_line_graph=output_line_graph)
            self.assertGreaterEqual(len(output_line_graph.data[0].points), 150)
            self.assertLessEqual(len(output_line_graph.data[0].points), 400)
            self.assertEqual(len(output_line_graph.data), 4)

        self.prompt_test(
            prompt=prompt, validate_plan=validate_plan, validate_output=validate_output
        )

    def test_plot_tsla_price(self):
        prompt = "plot tsla price from Jan 2024 to March 2024"

        def validate_output(prompt: str, output: IOType):
            output_line_graph = get_output(output)
            validate_line_graph(output_line_graph=output_line_graph)

        self.prompt_test(
            prompt=prompt, validate_plan=validate_plan, validate_output=validate_output
        )

    def test_intersection_of_qqq_xlv(self):
        prompt = "Find the intersection of QQQ and XLV on Jan 1, 2024"

        def validate_output(prompt: str, output: IOType):
            output_stock_ids = get_output(output=output)
            self.assertGreater(len(output_stock_ids), 0)

        self.prompt_test(
            prompt=prompt, validate_plan=validate_plan, validate_output=validate_output
        )

    def test_top_mcap(self):
        prompt = "top 10 by market cap today, and then graph their market caps over the month of April 2024"

        def validate_output(prompt: str, output: IOType):
            output_line_graph = get_output(output=output[1])
            validate_line_graph(output_line_graph=output_line_graph)
            self.assertEqual(len(output_line_graph.data), 10)
            self.assertGreaterEqual(len(output_line_graph.data[0].points), 10)

        self.prompt_test(
            prompt=prompt, validate_plan=validate_plan, validate_output=validate_output
        )

    def test_sector_stocks(self):
        prompt = "Find stocks in the technology sector"

        def validate_output(prompt: str, output: IOType):
            self.assertGreater(len(output), 0)

        self.prompt_test(
            prompt=prompt, validate_plan=validate_plan, validate_output=validate_output
        )

    def test_pe_nvda(self):
        prompt = "Show me the PE of NVDA over month of Feb 2024?"

        def validate_output(prompt: str, output: IOType):
            output_stock_table = get_output(output)
            date_column, pe_column = validate_table_and_get_columns(
                output_stock_table=output_stock_table,
                column_types=[TableColumnType.DATE, TableColumnType.FLOAT],
            )
            self.assertGreater(len(date_column.data), 0)
            self.assertGreater(len(pe_column.data), 0)

        self.prompt_test(
            prompt=prompt, validate_plan=validate_plan, validate_output=validate_output
        )

    def test_graph_pe(self):
        prompt = "Graph the PE of health care stocks in QQQ over the year 2023"

        def validate_output(prompt: str, output: IOType):
            output_line_graph = get_output(output)
            validate_line_graph(output_line_graph=output_line_graph)
            self.assertGreaterEqual(len(output_line_graph.data[0].points), 150)
            self.assertLessEqual(len(output_line_graph.data[0].points), 400)

        self.prompt_test(
            prompt=prompt, validate_plan=validate_plan, validate_output=validate_output
        )

    def test_open_close_spread(self):
        prompt = "Calculate the spread between open and close for AAPL over the month of Jan 2024"

        def validate_output(prompt: str, output: IOType):
            output_stock_table = get_output(output)
            date_column = validate_table_and_get_columns(
                output_stock_table=output_stock_table, column_types=[TableColumnType.DATE]
            )[0]
            self.assertGreaterEqual(len(date_column.data), 10)
            self.assertLessEqual(len(date_column.data), 50)

        self.prompt_test(
            prompt=prompt, validate_plan=validate_plan, validate_output=validate_output
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
            prompt=prompt, validate_plan=validate_plan, validate_output=validate_output
        )

    def test_notify_big_big_developments(self):
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
            validate_plan=validate_plan,
            validate_output=validate_output,
            raise_plan_validation_error=True,
        )
