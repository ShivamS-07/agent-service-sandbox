import unittest
from typing import Dict, List, Union

import pandas as pd

from agent_service.io_type_utils import (
    ComplexIOBase,
    IOType,
    check_type_is_io_type,
    check_type_is_valid,
    dump_io_type,
    io_type,
    load_io_type,
)
from agent_service.io_types.table import Table, TableColumn, TableColumnType
from agent_service.types import ChatContext, Message


@io_type
class TestComplex1(ComplexIOBase):
    val: int
    another: str = "3"


@io_type
class TestComplexType(ComplexIOBase):
    val: int
    another: str = "3"
    x: TestComplex1 = TestComplex1(val=2)


class TestIOType(unittest.TestCase):
    def test_collection_serialization(self):
        table = TestComplexType(val=2)
        cases = [
            # Super nested!
            {
                "1": {"1": {"1": [1, 2, 3, "a", True], "2": [2.4]}, "2": [2.4]},
                "2": [[1, 2, 3], [4, 5, 6], ["a", 1, True]],
                "3": table,
            },
            [[1, 2, 3], [4, 5, 6], ["a", 1, True], table],
            {"1": 2, "a": "b"},
            {"1": [1, 2, 3, "a", True], "2": [2.4]},
        ]

        for arg in cases:
            res = dump_io_type(arg)
            loaded = load_io_type(res)
            self.assertEqual(type(loaded), type(arg))
            self.assertEqual(loaded, arg)

    def test_dataframe_serialization(self):
        df = pd.DataFrame(
            [[1, 2, 3], [4, 5, 6]],
            index=pd.Index(["a", "b"], name="Test Index"),
            columns=["x", "y", "z"],
        )

        cases = [
            [Table(data=df, columns=[TableColumn(label="A", col_type=TableColumnType.INTEGER)]), 1],
            Table(data=df, columns=[TableColumn(label="B", col_type=TableColumnType.STRING)]),
            Table(data=df, columns=[TableColumn(label="C", col_type=TableColumnType.FLOAT)]),
        ]
        for arg in cases:
            res = dump_io_type(arg)
            loaded = load_io_type(res)
            self.assertEqual(type(loaded), type(arg))
            if isinstance(arg, list):
                pd.testing.assert_frame_equal(loaded[0].data, arg[0].data)
            else:
                pd.testing.assert_frame_equal(loaded.data, arg.data)  # type: ignore

    def test_container_types(self):
        table = TestComplexType(val=2)
        content: Dict[str, IOType] = {
            "1": {"1": {"1": [1, 2, 3, "a", True], "2": [2.4]}, "2": [2.4]},
            "2": [[1, 2, 3], [4, 5, 6], ["a", 1, True]],
            "3": table,
        }
        m = Message(message=content, is_user_message=True)
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

        c = ChatContext(messages=[m, m, m])
        res_j = c.model_dump_json()
        loaded = ChatContext.model_validate_json(res_j)
        self.assertEqual(loaded, c)

    def test_check_type_is_io_type(self):
        class BadClass:
            x: int

        # type and expected output
        cases = [
            (int, True),
            (str, True),
            (bool, True),
            (float, True),
            (List[int], True),
            (List[Union[int, str, bool]], True),
            (List[Union[int, str, bytes]], False),
            (List, False),
            (Dict[int, List[List[TestComplexType]]], True),
            (TestComplexType, True),
            (Dict[int, List[List[BadClass]]], False),
            (BadClass, False),
            (Dict[bytes, List[List[TestComplexType]]], False),
        ]

        for typ, expected in cases:
            self.assertEqual(check_type_is_io_type(typ), expected)

    def test_check_type_is_valid(self):
        class BadClass:
            x: int

        cases = [
            (int, int, True),
            (str, str, True),
            (bool, bool, True),
            (List[int], List[int], True),
            (List[int], Dict[int, str], False),
            (List[int], List[Union[int, str]], True),
            (List[Union[int, str]], List[Union[int, str]], True),
            (Dict[int, Union[int, str]], Dict[int, Union[int, str]], True),
            (Dict[int, int], Dict[int, Union[int, str]], True),
            (TestComplexType, TestComplexType, True),
            (TestComplexType, BadClass, False),
            (BadClass, TestComplexType, False),
            (Union[int, str], Union[int, str, float], True),
            (int, Union[int, str], True),
            (Union[int, str], Union[int, str], True),
        ]

        for typ1, typ2, expected in cases:
            self.assertEqual(check_type_is_valid(typ1, typ2), expected, f"{typ1} and {typ2} error")
