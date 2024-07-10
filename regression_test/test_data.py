# type: ignore
from agent_service.io_type_utils import IOType, TableColumnType
from regression_test.test_regression import (
    TestExecutionPlanner,
    get_output,
    validate_plan,
)
from regression_test.util import (
    validate_and_compare_text,
    validate_line_graph,
    validate_table_and_get_columns,
)


class TestData(TestExecutionPlanner):
    def test_relative_strength(self):
        prompt = "Show me Relative Strength Index for NVDA, AMD, INTL and GOOG over the past year"

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
        prompt = "plot tsla price"

        def validate_output(prompt: str, output: IOType):
            output_line_graph = get_output(output)
            validate_line_graph(output_line_graph=output_line_graph)

        self.prompt_test(
            prompt=prompt, validate_plan=validate_plan, validate_output=validate_output
        )

    def test_intersection_of_qqq_xlv(self):
        prompt = "Find the intersection of QQQ and XLV"

        def validate_output(prompt: str, output: IOType):
            output_stock_ids = get_output(output=output)
            self.assertGreater(len(output_stock_ids), 0)

        self.prompt_test(
            prompt=prompt, validate_plan=validate_plan, validate_output=validate_output
        )

    def test_top_mcap(self):
        prompt = "top 10 by market cap today, and then graph their market caps over the last month"

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
        prompt = "Show me the PE of NVDA?"

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

    def test_mcap_nvda(self):
        prompt = "Show me the market cap of NVDA?"

        def validate_output(prompt: str, output: IOType):
            output_stock_table = get_output(output)
            date_column, mcap_column = validate_table_and_get_columns(
                output_stock_table=output_stock_table,
                column_types=[TableColumnType.DATE, TableColumnType.CURRENCY],
            )
            self.assertGreater(len(mcap_column.data), 0)

        self.prompt_test(
            prompt=prompt, validate_plan=validate_plan, validate_output=validate_output
        )

    def test_graph_pe(self):
        prompt = "Graph the PE of health care stocks in QQQ over the past year"

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
            "Can you give me a single summary of news published in the last week about machine "
            "learning at Meta, Apple, and Microsoft?"
        )

        def validate_output(prompt: str, output: IOType):
            output_text = get_output(output)
            self.loop.run_until_complete(
                validate_and_compare_text(llm=self.llm, output_text=output_text, prompt=prompt)
            )

        self.prompt_test(
            prompt=prompt, validate_plan=validate_plan, validate_output=validate_output
        )
