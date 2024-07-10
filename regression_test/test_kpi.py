# type: ignore
from agent_service.io_type_utils import IOType
from regression_test.test_regression import (
    TestExecutionPlanner,
    get_output,
    validate_plan,
)
from regression_test.util import (
    validate_and_compare_text,
    validate_table_and_get_columns,
)


class TestKPI(TestExecutionPlanner):
    def test_exploration_expense(self):
        prompt = "Show Exploration Expense for XOM"

        def validate_output(prompt: str, output: IOType):
            output_stock_table = get_output(output=output)
            validate_table_and_get_columns(output_stock_table=output_stock_table, column_types=[])

        self.prompt_test(
            prompt=prompt, validate_plan=validate_plan, validate_output=validate_output
        )

    def test_main_kpi_compare(self):
        prompt = "Compare how the main KPI for Microsoft have been discussed in the first 2 earning's calls of 2024"

        def validate_output(prompt: str, output: IOType):
            output_text = get_output(output)
            self.loop.run_until_complete(
                validate_and_compare_text(llm=self.llm, output_text=output_text, prompt=prompt)
            )

        self.prompt_test(
            prompt=prompt, validate_plan=validate_plan, validate_output=validate_output
        )
