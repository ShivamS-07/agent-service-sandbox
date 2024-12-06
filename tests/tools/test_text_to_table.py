import unittest
from typing import Optional

from parameterized import param, parameterized

from agent_service.tools.text_to_table import extract_number_with_unit_from_text


class TestTextToTable(unittest.TestCase):

    @parameterized.expand(
        [
            param(input_str="Nothing", to_int=False, expected_val=None, expected_unit=None),
            param(input_str="123", to_int=True, expected_val=123, expected_unit=None),
            param(input_str="123.0", to_int=False, expected_val=123.0, expected_unit=None),
            param(input_str="123,325.0", to_int=False, expected_val=123325.0, expected_unit=None),
            param(input_str="123 USD", to_int=True, expected_val=123, expected_unit="USD"),
            param(input_str="123 OBDF", to_int=True, expected_val=123, expected_unit=None),
            param(input_str="$123.0", to_int=False, expected_val=123.0, expected_unit="USD"),
            param(
                input_str="$123,325.0 USD", to_int=False, expected_val=123325.0, expected_unit="USD"
            ),
        ]
    )
    def test_extract_number_with_unit_from_text(
        self, input_str: str, to_int: bool, expected_val: int | float, expected_unit: Optional[str]
    ):
        val, unit = extract_number_with_unit_from_text(val=input_str, return_int=to_int)
        self.assertEqual(val, expected_val)
        self.assertEqual(unit, expected_unit)
