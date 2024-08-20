import unittest

import pandas as pd

from agent_service.io_type_utils import TableColumnType
from agent_service.io_types.table import TableColumnMetadata
from agent_service.tools.product_comparison.helpers import update_dataframe


class TestProductCompare(unittest.TestCase):
    def test_update_dataframe(self):
        columns = []

        # the data list will contain lists where keys and values are all strings
        data_list = [
            {
                "rank": "4",
                "capacity": "123 kwh",
                "speed": "20.6 km/h",
                "model": "Car 1",
                "auto": "n/a",
                "charge": "2 hours",
                "price": "16,000 dollars",
            },
            {
                "rank": "5",
                "capacity": "13 kwh",
                "speed": "24 km/h",
                "model": "2 Car",
                "auto": "n/a",
                "charge": "30 minutes",
                "price": "24,000 dollars",
            },
            {
                "rank": "2",
                "capacity": "113 kwh",
                "speed": "25.7 km/h",
                "model": "3 Car",
                "auto": "n/a",
                "charge": "1 day",
                "price": "29,000 dollars",
            },
            {
                "rank": "3",
                "capacity": "n/a",
                "speed": "n/a",
                "model": "5 Car 3",
                "auto": "n/a",
                "charge": "3 days",
                "price": "56,000 dollars",
            },
            {
                "rank": "1",
                "capacity": "n/a",
                "speed": "20.6 km/h",
                "model": "Car 67",
                "auto": "n/a",
                "charge": "2.5 hour",
                "price": "145,000 dollars",
            },
        ]

        column_titles = data_list[0].keys()
        for title in column_titles:
            columns.append(TableColumnMetadata(label=title, col_type=TableColumnType.STRING))
        df = pd.DataFrame(data_list)

        # test that we can detect there is a common suffix in the speed column
        speed_col_before = TableColumnMetadata(label="speed", col_type=TableColumnType.STRING)
        self.assertEqual(columns[2], speed_col_before)

        # update the information in the dataframe based on the column contents
        columns[2] = update_dataframe(
            "speed", ["20.6 km/h", "24 km/h", "25.7 km/h", "n/a", "20.6 km/h"], df
        )

        speed_col_after = TableColumnMetadata(
            label="speed", unit="km/h", col_type=TableColumnType.FLOAT_WITH_UNIT
        )
        self.assertEqual(speed_col_after, columns[2])

        # test that we cannot detect a common suffix in the model column
        model_col_before = TableColumnMetadata(label="model", col_type=TableColumnType.STRING)
        self.assertEqual(columns[3], model_col_before)

        # update the information in the dataframe based on the column contents
        columns[3] = update_dataframe("model", ["Car 1", "2 Car", "3 Car", "5 Car 3", "Car 67"], df)

        model_col_after = TableColumnMetadata(label="model", col_type=TableColumnType.STRING)
        self.assertEqual(columns[3], model_col_after)

        expected_data_list = [
            {
                "rank": "4",
                "capacity": "123 kwh",
                "speed": 20.6,
                "model": "Car 1",
                "auto": "n/a",
                "charge": "2 hours",
                "price": "16,000 dollars",
            },
            {
                "rank": "5",
                "capacity": "13 kwh",
                "speed": 24,
                "model": "2 Car",
                "auto": "n/a",
                "charge": "30 minutes",
                "price": "24,000 dollars",
            },
            {
                "rank": "2",
                "capacity": "113 kwh",
                "speed": 25.7,
                "model": "3 Car",
                "auto": "n/a",
                "charge": "1 day",
                "price": "29,000 dollars",
            },
            {
                "rank": "3",
                "capacity": "n/a",
                "speed": "n/a",
                "model": "5 Car 3",
                "auto": "n/a",
                "charge": "3 days",
                "price": "56,000 dollars",
            },
            {
                "rank": "1",
                "capacity": "n/a",
                "speed": 20.6,
                "model": "Car 67",
                "auto": "n/a",
                "charge": "2.5 hour",
                "price": "145,000 dollars",
            },
        ]

        expected_df = pd.DataFrame(expected_data_list)
        self.assertEqual(df["speed"].tolist(), expected_df["speed"].tolist())
        self.assertEqual(df["model"].tolist(), expected_df["model"].tolist())
