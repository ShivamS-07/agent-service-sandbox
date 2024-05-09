import pandas as pd

from agent_service.io_type_utils import ComplexIOBase, io_type


@io_type
class StockTimeseriesTable(ComplexIOBase):
    # A dataframe with date row index and GBI ID columns.
    val: pd.DataFrame


@io_type
class StockTable(ComplexIOBase):
    # A dataframe with GBI ID row index and arbitrary columns.
    val: pd.DataFrame
