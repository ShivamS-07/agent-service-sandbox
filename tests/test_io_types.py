import datetime
import unittest

import pandas as pd

from agent_service.tools.io_type_utils import ComplexIOBase
from agent_service.tools.io_types import (
    ListofLists,
    Mapping,
    StockTable,
    StockTimeseriesTable,
)
from agent_service.types import ChatContext, Message


class TestIOType(unittest.TestCase):
    def test_complex_serialization(self):
        cases = [
            ListofLists(val=[[1, 2, 3], [4, 5, 6], ["a", 1, True]]),
            Mapping(val={"1": 2, "a": "b"}),
            Mapping(val={"1": [1, 2, 3, "a", True], "2": [2.4]}),
            # Super nested!
            Mapping(
                val={
                    "1": Mapping(
                        val={"1": Mapping(val={"1": [1, 2, 3, "a", True], "2": [2.4]}), "2": [2.4]}
                    ),
                    "2": ListofLists(val=[[1, 2, 3], [4, 5, 6], ["a", 1, True]]),
                }
            ),
        ]

        for arg in cases:
            res = arg.model_dump()
            loaded = ComplexIOBase.load(res)
            self.assertEqual(type(loaded), type(arg))
            self.assertEqual(loaded, arg)

            res_j = arg.model_dump_json()
            loaded = ComplexIOBase.load_json(res_j)
            self.assertEqual(type(loaded), type(arg))
            self.assertEqual(loaded, arg)

    def test_dataframe_serialization(self):
        df = pd.DataFrame([[1, 2, 3], [4, 5, 6]], index=["a", "b"], columns=["x", "y", "z"])

        cases = [StockTimeseriesTable(val=df), StockTable(val=df)]
        for arg in cases:
            res = arg.model_dump()
            loaded = ComplexIOBase.load(res)
            self.assertEqual(type(loaded), type(arg))
            pd.testing.assert_frame_equal(loaded.val, arg.val)  # type: ignore

            res_j = arg.model_dump_json()
            loaded = ComplexIOBase.load_json(res_j)
            self.assertEqual(type(loaded), type(arg))
            pd.testing.assert_frame_equal(loaded.val, arg.val)  # type: ignore

    def test_container_types(self):
        content = Mapping(
            val={
                "1": Mapping(
                    val={"1": Mapping(val={"1": [1, 2, 3, "a", True], "2": [2.4]}), "2": [2.4]}
                ),
                "2": ListofLists(val=[[1, 2, 3], [4, 5, 6], ["a", 1, True]]),
            }
        )
        m = Message(content=content, is_user=True, timestamp=datetime.datetime.now())
        res = m.model_dump()
        loaded = Message(**res)
        self.assertEqual(loaded, m)

        res_j = m.model_dump_json()
        loaded = Message.model_validate_json(res_j)
        self.assertEqual(loaded, m)

        c = ChatContext(messages=[m, m, m])
        res = c.model_dump()
        loaded = ChatContext(**res)
        self.assertEqual(loaded, c)
