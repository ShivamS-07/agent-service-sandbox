import pandas as pd

from agent_service.io_type_utils import ComplexIOBase, io_type


@io_type
class Table(ComplexIOBase):
    # A dataframe wrapper
    val: pd.DataFrame

    def to_gpt_input(self) -> str:
        return f"[Table with {self.val.shape[0]} rows and {self.val.shape[0]} columns]"


@io_type
class TimeSeriesTable(Table):
    # A dataframe with date row index.
    val: pd.DataFrame


@io_type
class StockTimeSeriesTable(TimeSeriesTable):
    # A dataframe with date row index and GBI ID columns.
    pass


@io_type
class StockTable(Table):
    # A dataframe with GBI ID row index and arbitrary columns.
    pass


@io_type
class Graph(ComplexIOBase):
    # TODO: figure out how we are going to represent a graph, now just a table
    val: Table

    def to_gpt_input(self) -> str:
        return f"[Graph based on table with {self.val.val.shape[0]} rows and {self.val.val.shape[0]} columns]"


@io_type
class TimeSeriesLineGraph(Graph):
    pass
