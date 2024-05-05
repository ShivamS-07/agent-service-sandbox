from typing import Literal

import pandas as pd

from agent_service.tools.io_type_utils import ComplexIOBase, IOTypeEnum


class StockTimeseriesTable(ComplexIOBase):
    # Must define this so that pydantic can dump and load from json automatically.
    io_type: Literal[IOTypeEnum.STOCK_TIMESERIES] = IOTypeEnum.STOCK_TIMESERIES

    # A dataframe with date row index and GBI ID columns.
    val: pd.DataFrame
