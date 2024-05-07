import datetime
import unittest

import pandas as pd

from agent_service.tools.io_type_utils import IO_TYPE_NAME_KEY, IOTypeSerializer
from agent_service.tools.io_types import (
    ListofLists,
    Mapping,
    StockTable,
    StockTimeseriesTable,
)


class TestIOType(unittest.TestCase):
    def test_primitive_serialization(self):
        # Tuple of input/output
        cases = [
            (27, 27),
            ("test", "test"),
            (2.14, 2.14),
            (True, True),
            (
                datetime.date(2024, 1, 1),
                {IO_TYPE_NAME_KEY: "Date", "val": datetime.date(2024, 1, 1)},
            ),
            (
                datetime.datetime(2024, 1, 1),
                {IO_TYPE_NAME_KEY: "DateTime", "val": datetime.datetime(2024, 1, 1)},
            ),
            ([1, 2, 3], [1, 2, 3]),
            ([1.1, 2.2, 3.3], [1.1, 2.2, 3.3]),
            (["a", "b", "c"], ["a", "b", "c"]),
            ([True, False], [True, False]),
            ([1, 1.1, "a", True], [1, 1.1, "a", True]),
        ]
        for arg, expected in cases:
            res = IOTypeSerializer.dump_io_type_dict(val=arg)
            self.assertEqual(res, expected)
            loaded = IOTypeSerializer.load_io_type_dict(res)
            self.assertEqual(loaded, arg)
            json = IOTypeSerializer.dump_io_type_json(val=arg)
            loaded = IOTypeSerializer.load_io_type_json(json)
            self.assertEqual(loaded, arg)

    def test_complex_serialization(self):
        cases = [
            ListofLists(val=[[1, 2, 3], [4, 5, 6], ["a", 1, True]]),
            Mapping(val={"1": 2, "a": "b"}),
            Mapping(val={"1": [1, 2, 3, "a", True], "2": [2.4]}),
        ]

        for arg in cases:
            res = IOTypeSerializer.dump_io_type_dict(val=arg)
            loaded = IOTypeSerializer.load_io_type_dict(res)
            self.assertEqual(type(loaded), type(arg))
            self.assertEqual(loaded, arg)

            res = IOTypeSerializer.dump_io_type_json(val=arg)
            loaded = IOTypeSerializer.load_io_type_json(res)
            self.assertEqual(type(loaded), type(arg))
            self.assertEqual(loaded, arg)

    def test_dataframe_serialization(self):
        df = pd.DataFrame([[1, 2, 3], [4, 5, 6]], index=["a", "b"], columns=["x", "y", "z"])

        cases = [StockTimeseriesTable(val=df), StockTable(val=df)]
        for arg in cases:
            res = IOTypeSerializer.dump_io_type_dict(val=arg)
            loaded = IOTypeSerializer.load_io_type_dict(res)
            self.assertEqual(type(loaded), type(arg))
            pd.testing.assert_frame_equal(loaded.val, arg.val)  # type: ignore

            json = IOTypeSerializer.dump_io_type_json(val=arg)
            loaded = IOTypeSerializer.load_io_type_json(json)
            self.assertEqual(type(loaded), type(arg))
            pd.testing.assert_frame_equal(loaded.val, arg.val)  # type: ignore
