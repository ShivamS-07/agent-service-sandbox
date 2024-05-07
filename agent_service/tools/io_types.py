from typing import Dict, List, Union

import pandas as pd

from agent_service.tools.io_type_utils import ComplexIOBase, SimpleType, io_type


@io_type
class StockTimeseriesTable(ComplexIOBase):
    # A dataframe with date row index and GBI ID columns.
    val: pd.DataFrame


@io_type
class StockTable(ComplexIOBase):
    # A dataframe with GBI ID row index and arbitrary columns.
    val: pd.DataFrame


@io_type
class ListofLists(ComplexIOBase):
    # List of lists of (int, float, bool, str)
    val: List[List[SimpleType]]


@io_type
class Mapping(ComplexIOBase):
    val: Dict[str, Union[SimpleType, List[SimpleType]]]
