# type: ignore
from agent_service.io_type_utils import IOType, TableColumnType
from regression_test.test_regression import (
    TestExecutionPlanner,
    get_output,
    validate_plan,
)
from regression_test.util import validate_table_and_get_columns


class TestPortfolioMonitoring(TestExecutionPlanner):
    def test_portfolio_monitoring(self):
        prompt = "Notify me any time a stock in my portfolio announces earnings"

        def validate_output(prompt: str, output: IOType):
            output_stock_table = get_output(output=output)
            validate_table_and_get_columns(
                output_stock_table=output_stock_table, column_types=[TableColumnType.DATE]
            )[0]

        self.prompt_test(
            prompt=prompt, validate_plan=validate_plan, validate_output=validate_output
        )