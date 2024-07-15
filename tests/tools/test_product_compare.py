import unittest

from agent_service.io_type_utils import TableColumnType
from agent_service.io_types.table import TableColumnMetadata
from agent_service.tools.product_compare import column_treatment


class TestProductCompare(unittest.TestCase):
    def test_column_treatment(self):
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
            columns.append(column_treatment(title, data_list))

        rank_col = TableColumnMetadata(label="rank", col_type=TableColumnType.INTEGER)
        capacity_col = TableColumnMetadata(
            label="capacity", unit="kwh", col_type=TableColumnType.INTEGER_WITH_UNIT
        )
        speed_col = TableColumnMetadata(
            label="speed", unit="km/h", col_type=TableColumnType.FLOAT_WITH_UNIT
        )
        model_col = TableColumnMetadata(label="model", col_type=TableColumnType.STRING)
        auto_col = TableColumnMetadata(label="auto", col_type=TableColumnType.STRING)
        charge_col = TableColumnMetadata(label="charge", col_type=TableColumnType.STRING)
        price_col = TableColumnMetadata(
            label="price", unit="dollars", col_type=TableColumnType.INTEGER_WITH_UNIT
        )

        self.assertEqual(columns[0], rank_col)
        self.assertEqual(columns[1], capacity_col)
        self.assertEqual(columns[2], speed_col)
        self.assertEqual(columns[3], model_col)
        self.assertEqual(columns[4], auto_col)
        self.assertEqual(columns[5], charge_col)
        self.assertEqual(columns[6], price_col)

        expected_data_list = [
            {
                "rank": 4,
                "capacity": 123,
                "speed": 20.6,
                "model": "Car 1",
                "auto": "n/a",
                "charge": "2 hours",
                "price": 16000,
            },
            {
                "rank": 5,
                "capacity": 13,
                "speed": 24,
                "model": "2 Car",
                "auto": "n/a",
                "charge": "30 minutes",
                "price": 24000,
            },
            {
                "rank": 2,
                "capacity": 113,
                "speed": 25.7,
                "model": "3 Car",
                "auto": "n/a",
                "charge": "1 day",
                "price": 29000,
            },
            {
                "rank": 3,
                "capacity": "n/a",
                "speed": "n/a",
                "model": "5 Car 3",
                "auto": "n/a",
                "charge": "3 days",
                "price": 56000,
            },
            {
                "rank": 1,
                "capacity": "n/a",
                "speed": 20.6,
                "model": "Car 67",
                "auto": "n/a",
                "charge": "2.5 hour",
                "price": 145000,
            },
        ]

        self.assertEqual(data_list, expected_data_list)
