import unittest
from typing import Any, Dict, List, Union

import pandas as pd

from agent_service.io_type_utils import (
    ComplexIOBase,
    HistoryEntry,
    IOType,
    TableColumnType,
    check_type_is_io_type,
    check_type_is_valid,
    dump_io_type,
    io_type,
    load_io_type,
)
from agent_service.io_types.table import StockTable, Table, TableColumnMetadata
from agent_service.io_types.text import StockText, Text, TextCitation, WebText
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

    def __hash__(self) -> int:
        return self.val

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, self.__class__):
            return self.val == other.val
        return NotImplemented


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
            # Tuples (which aren't technically supported but sort of need to be
            # now for backwards compatibility)
            [(1, True), ("x", False)],
            ("hello", "goodbye"),
            (
                "test",
                TestComplexType(
                    title="",
                    val=3,
                    history=[
                        HistoryEntry(
                            explanation="test", citations=[TextCitation(source_text=Text())]
                        )
                    ],
                ),
            ),
        ]

        for arg in cases:
            res = dump_io_type(arg)
            loaded = load_io_type(res)
            self.assertEqual(type(loaded), type(arg))
            self.assertEqual(loaded, arg)

    def test_table_serialization(self):
        df = pd.DataFrame(
            [[1, 2, 3], [4, 5, 6]],
            columns=["x", "y", "z"],
        )

        cases = [
            [
                Table.from_df_and_cols(
                    data=df,
                    columns=[TableColumnMetadata(label="x", col_type=TableColumnType.INTEGER)],
                ),
                1,
            ],
            Table.from_df_and_cols(
                data=df, columns=[TableColumnMetadata(label="y", col_type=TableColumnType.STRING)]
            ),
            Table.from_df_and_cols(
                data=df, columns=[TableColumnMetadata(label="z", col_type=TableColumnType.FLOAT)]
            ),
        ]
        for arg in cases:
            res = dump_io_type(arg)
            loaded = load_io_type(res)
            self.assertEqual(type(loaded), type(arg))

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
            (int | str, int | str | float, True),
            (int | str, float | bool, False),
            (int, Union[int, str], True),
            (int, int | str, True),
            (Union[int, str], Union[int, str], True),
            (int | str, int | str, True),
            # Nested generics
            (list[int], list[int | str], True),
            (list[int | str], list[int], False),
            (Union[StockTable, Table], Table, True),
            (Union[WebText, StockText], Text, True),
        ]

        for typ1, typ2, expected in cases:
            with self.subTest(msg=f"Actual: {typ1}, Expected: {typ2}"):
                self.assertEqual(
                    check_type_is_valid(typ1, typ2), expected, f"{typ1} and {typ2} error"
                )

    def test_union_with_history(self):
        set1 = {
            TestComplexType(val=1, history=[HistoryEntry(explanation="Test1", title="")]),
            TestComplexType(val=2),
            TestComplexType(val=3),
        }
        set2 = {
            TestComplexType(val=1, history=[HistoryEntry(explanation="Test2", title="")]),
            TestComplexType(val=6, history=[HistoryEntry(explanation="Test1", title="")]),
            TestComplexType(val=4),
        }
        result = TestComplexType.union_sets(set1, set2)
        self.assertEqual(
            result,
            {
                TestComplexType(
                    val=1,
                    history=[
                        HistoryEntry(explanation="Test1", title=""),
                        HistoryEntry(explanation="Test2", title=""),
                    ],
                ),
                TestComplexType(val=2),
                TestComplexType(val=3),
                TestComplexType(val=6, history=[HistoryEntry(explanation="Test1", title="")]),
                TestComplexType(val=4),
            },
        )

    def test_intersection_with_history(self):
        set1 = {
            TestComplexType(val=1, history=[HistoryEntry(explanation="Test1", title="")]),
            TestComplexType(val=2),
            TestComplexType(val=3),
        }
        set2 = {
            TestComplexType(val=1, history=[HistoryEntry(explanation="Test2", title="")]),
            TestComplexType(val=6, history=[HistoryEntry(explanation="Test1", title="")]),
            TestComplexType(val=4),
            TestComplexType(val=3),
        }
        result = TestComplexType.intersect_sets(set1, set2)
        self.assertEqual(
            result,
            {
                TestComplexType(
                    val=1,
                    history=[
                        HistoryEntry(explanation="Test1", title=""),
                        HistoryEntry(explanation="Test2", title=""),
                    ],
                ),
                TestComplexType(val=3),
            },
        )
