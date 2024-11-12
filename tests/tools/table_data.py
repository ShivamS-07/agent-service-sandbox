import datetime

import pandas as pd

from agent_service.io_type_utils import HistoryEntry, TableColumnType
from agent_service.io_types.stock import StockID
from agent_service.io_types.table import Table, TableColumnMetadata

STOCK1 = StockID(gbi_id=72, symbol="AAPL", isin="", company_name="")
STOCK2 = StockID(gbi_id=76, symbol="TSLA", isin="", company_name="")
STOCK3 = StockID(gbi_id=78, symbol="MSFT", isin="", company_name="")
STOCK4 = StockID(
    gbi_id=112,
    symbol="GOOG",
    isin="",
    company_name="",
    history=[HistoryEntry(explanation="Test 1", title="Test")],
)
STOCK5 = StockID(gbi_id=124, symbol="IBM", isin="", company_name="")
STOCK6 = StockID(gbi_id=149, symbol="IDK", isin="", company_name="")
STOCK4_alt = StockID(
    gbi_id=112,
    symbol="GOOG",
    isin="",
    company_name="",
    history=[HistoryEntry(explanation="Test 2")],
)


TEST_STOCK_DATE_TABLE1 = Table.from_df_and_cols(
    columns=[
        TableColumnMetadata(label="Date", col_type=TableColumnType.DATE, unit=None),
        TableColumnMetadata(label="Security", col_type=TableColumnType.STOCK, unit=None),
        TableColumnMetadata(label="Close Price", col_type=TableColumnType.FLOAT, unit=None),
    ],
    data=pd.DataFrame(
        data={
            "Date": {
                0: datetime.date(2024, 5, 15),
                1: datetime.date(2024, 5, 16),
                2: datetime.date(2024, 5, 17),
                3: datetime.date(2024, 5, 18),
                4: datetime.date(2024, 5, 19),
                5: datetime.date(2024, 5, 15),
                6: datetime.date(2024, 5, 16),
                7: datetime.date(2024, 5, 17),
                8: datetime.date(2024, 5, 18),
                9: datetime.date(2024, 5, 19),
                10: datetime.date(2024, 5, 15),
                11: datetime.date(2024, 5, 16),
                12: datetime.date(2024, 5, 17),
                13: datetime.date(2024, 5, 18),
                14: datetime.date(2024, 5, 19),
            },
            "Security": {
                0: STOCK4,
                1: STOCK4,
                2: STOCK4,
                3: STOCK4,
                4: STOCK1,
                5: STOCK2,
                6: STOCK2,
                7: STOCK2,
                8: STOCK2,
                9: STOCK2,
                10: STOCK3,
                11: STOCK3,
                12: STOCK3,
                13: STOCK3,
                14: STOCK3,
            },
            "Close Price": {
                0: 221.81,
                1: 221.85,
                2: 222.12,
                3: 222.12,
                4: 222.12,
                5: 78.9,
                6: 78.74,
                7: 80.54,
                8: 80.54,
                9: 80.54,
                10: 241.7,
                11: 241.32,
                12: 242.82,
                13: 242.82,
                14: 242.82,
            },
        }
    ),
)

TEST_STOCK_DATE_TABLE2 = Table.from_df_and_cols(
    columns=[
        TableColumnMetadata(label="Date", col_type=TableColumnType.DATE, unit=None),
        TableColumnMetadata(label="Security", col_type=TableColumnType.STOCK, unit=None),
        TableColumnMetadata(label="Open Price", col_type=TableColumnType.FLOAT, unit=None),
    ],
    data=pd.DataFrame(
        data={
            "Date": {
                0: datetime.date(2024, 5, 16),
                1: datetime.date(2024, 5, 17),
                2: datetime.date(2024, 5, 18),
                3: datetime.date(2024, 5, 19),
                4: datetime.date(2024, 5, 16),
                5: datetime.date(2024, 5, 17),
                6: datetime.date(2024, 5, 18),
                7: datetime.date(2024, 5, 19),
                8: datetime.date(2024, 5, 16),
                9: datetime.date(2024, 5, 17),
                10: datetime.date(2024, 5, 18),
                11: datetime.date(2024, 5, 19),
            },
            "Security": {
                0: STOCK4,
                1: STOCK4,
                2: STOCK4,
                3: STOCK4,
                4: STOCK5,
                5: STOCK5,
                6: STOCK5,
                7: STOCK5,
                8: STOCK6,
                9: STOCK6,
                10: STOCK6,
                11: STOCK6,
            },
            "Open Price": {
                0: 21.21,
                1: 21.1,
                2: 21.1,
                3: 21.1,
                4: 160.92,
                5: 168.43,
                6: 168.43,
                7: 168.43,
                8: 185.6,
                9: 183.76,
                10: 183.76,
                11: 183.76,
            },
        }
    ),
)

TEST_STOCK_QTR_TABLE = Table.from_df_and_cols(
    columns=[
        TableColumnMetadata(label="Quarter", col_type=TableColumnType.QUARTER, unit=None),
        TableColumnMetadata(label="Security", col_type=TableColumnType.STOCK, unit=None),
        TableColumnMetadata(label="Close Price", col_type=TableColumnType.FLOAT, unit=None),
    ],
    data=pd.DataFrame(
        data={
            "Quarter": {
                0: "2024Q1",
                1: "2024Q2",
                2: "2024Q3",
                3: "2024Q4",
                4: "2024Q1",
                5: "2024Q2",
                6: "2024Q3",
                7: "2024Q4",
                8: "2024Q1",
                9: "2024Q2",
                10: "2024Q3",
                11: "2024Q4",
            },
            "Security": {
                0: STOCK4,
                1: STOCK4,
                2: STOCK4,
                3: STOCK4,
                4: STOCK5,
                5: STOCK5,
                6: STOCK5,
                7: STOCK5,
                8: STOCK6,
                9: STOCK6,
                10: STOCK6,
                11: STOCK6,
            },
            "Close Price": {
                0: 21.21,
                1: 21.1,
                2: 21.1,
                3: 21.1,
                4: 160.92,
                5: 168.43,
                6: 168.43,
                7: 168.43,
                8: 185.6,
                9: 183.76,
                10: 183.76,
                11: 183.76,
            },
        }
    ),
)

TEST_STOCK_YEAR_TABLE = Table.from_df_and_cols(
    columns=[
        TableColumnMetadata(label="Year", col_type=TableColumnType.YEAR, unit=None),
        TableColumnMetadata(label="Security", col_type=TableColumnType.STOCK, unit=None),
        TableColumnMetadata(label="Open Price", col_type=TableColumnType.FLOAT, unit=None),
    ],
    data=pd.DataFrame(
        data={
            "Year": {
                0: "2021",
                1: "2022",
                2: "2023",
                3: "2024",
                4: "2021",
                5: "2022",
                6: "2023",
                7: "2024",
                8: "2021",
                9: "2022",
                10: "2023",
                11: "2024",
            },
            "Security": {
                0: STOCK4,
                1: STOCK4,
                2: STOCK4,
                3: STOCK4,
                4: STOCK5,
                5: STOCK5,
                6: STOCK5,
                7: STOCK5,
                8: STOCK6,
                9: STOCK6,
                10: STOCK6,
                11: STOCK6,
            },
            "Open Price": {
                0: 21.21,
                1: 21.1,
                2: 21.1,
                3: 21.1,
                4: 160.92,
                5: 168.43,
                6: 168.43,
                7: 168.43,
                8: 185.6,
                9: 183.76,
                10: 183.76,
                11: 183.76,
            },
        }
    ),
)

TEST_STOCK_MONTH_TABLE = Table.from_df_and_cols(
    columns=[
        TableColumnMetadata(label="Month", col_type=TableColumnType.MONTH, unit=None),
        TableColumnMetadata(label="Security", col_type=TableColumnType.STOCK, unit=None),
        TableColumnMetadata(label="Open Price", col_type=TableColumnType.FLOAT, unit=None),
    ],
    data=pd.DataFrame(
        data={
            "Month": {
                0: "2021-01",
                1: "2022-02",
                2: "2023-03",
                3: "2024-04",
                4: "2021-01",
                5: "2022-02",
                6: "2023-03",
                7: "2024-04",
                8: "2021-01",
                9: "2022-02",
                10: "2023-03",
                11: "2024-04",
            },
            "Security": {
                0: STOCK4,
                1: STOCK4,
                2: STOCK4,
                3: STOCK4,
                4: STOCK5,
                5: STOCK5,
                6: STOCK5,
                7: STOCK5,
                8: STOCK6,
                9: STOCK6,
                10: STOCK6,
                11: STOCK6,
            },
            "Open Price": {
                0: 21.21,
                1: 21.1,
                2: 21.1,
                3: 21.1,
                4: 160.92,
                5: 168.43,
                6: 168.43,
                7: 168.43,
                8: 185.6,
                9: 183.76,
                10: 183.76,
                11: 183.76,
            },
        }
    ),
)

TEST_STRING_DATE_TABLE1 = Table.from_df_and_cols(
    columns=[
        TableColumnMetadata(label="Date", col_type=TableColumnType.DATE, unit=None),
        TableColumnMetadata(label="Security", col_type=TableColumnType.STRING, unit=None),
        TableColumnMetadata(label="Close Price", col_type=TableColumnType.FLOAT, unit=None),
    ],
    data=pd.DataFrame(
        data={
            "Date": {
                0: datetime.date(2024, 5, 15),
                1: datetime.date(2024, 5, 16),
                2: datetime.date(2024, 5, 17),
                3: datetime.date(2024, 5, 18),
                4: datetime.date(2024, 5, 19),
                5: datetime.date(2024, 5, 15),
                6: datetime.date(2024, 5, 16),
                7: datetime.date(2024, 5, 17),
                8: datetime.date(2024, 5, 18),
                9: datetime.date(2024, 5, 19),
                10: datetime.date(2024, 5, 15),
                11: datetime.date(2024, 5, 16),
                12: datetime.date(2024, 5, 17),
                13: datetime.date(2024, 5, 18),
                14: datetime.date(2024, 5, 19),
            },
            "Security": {
                0: STOCK4.symbol,
                1: STOCK4.symbol,
                2: STOCK4.symbol,
                3: STOCK4.symbol,
                4: STOCK1.symbol,
                5: STOCK2.symbol,
                6: STOCK2.symbol,
                7: STOCK2.symbol,
                8: STOCK2.symbol,
                9: STOCK2.symbol,
                10: STOCK3.symbol,
                11: STOCK3.symbol,
                12: STOCK3.symbol,
                13: STOCK3.symbol,
                14: STOCK3.symbol,
            },
            "Close Price": {
                0: 221.81,
                1: 221.85,
                2: 222.12,
                3: 222.12,
                4: 222.12,
                5: 78.9,
                6: 78.74,
                7: 80.54,
                8: 80.54,
                9: 80.54,
                10: 241.7,
                11: 241.32,
                12: 242.82,
                13: 242.82,
                14: 242.82,
            },
        }
    ),
)

TEST_STRING_DATE_TABLE2 = Table.from_df_and_cols(
    columns=[
        TableColumnMetadata(label="Date", col_type=TableColumnType.DATE, unit=None),
        TableColumnMetadata(label="Security", col_type=TableColumnType.STRING, unit=None),
        TableColumnMetadata(label="Open Price", col_type=TableColumnType.FLOAT, unit=None),
    ],
    data=pd.DataFrame(
        data={
            "Date": {
                0: datetime.date(2024, 5, 16),
                1: datetime.date(2024, 5, 17),
                2: datetime.date(2024, 5, 18),
                3: datetime.date(2024, 5, 19),
                4: datetime.date(2024, 5, 16),
                5: datetime.date(2024, 5, 17),
                6: datetime.date(2024, 5, 18),
                7: datetime.date(2024, 5, 19),
                8: datetime.date(2024, 5, 16),
                9: datetime.date(2024, 5, 17),
                10: datetime.date(2024, 5, 18),
                11: datetime.date(2024, 5, 19),
            },
            "Security": {
                0: STOCK4.symbol,
                1: STOCK4.symbol,
                2: STOCK4.symbol,
                3: STOCK4.symbol,
                4: STOCK5.symbol,
                5: STOCK5.symbol,
                6: STOCK5.symbol,
                7: STOCK5.symbol,
                8: STOCK6.symbol,
                9: STOCK6.symbol,
                10: STOCK6.symbol,
                11: STOCK6.symbol,
            },
            "Open Price": {
                0: 21.21,
                1: 21.1,
                2: 21.1,
                3: 21.1,
                4: 160.92,
                5: 168.43,
                6: 168.43,
                7: 168.43,
                8: 185.6,
                9: 183.76,
                10: 183.76,
                11: 183.76,
            },
        }
    ),
)

TEST_STOCK_TABLE1 = Table.from_df_and_cols(
    columns=[
        TableColumnMetadata(label="Security", col_type=TableColumnType.STOCK),
        TableColumnMetadata(label="News Summary", col_type=TableColumnType.STRING),
    ],
    data=pd.DataFrame(
        data={
            "Security": {
                0: STOCK4,
                1: STOCK5,
                2: STOCK6,
            },
            "News Summary": {
                0: "blah1",
                1: "blah2",
                2: "blah3",
            },
        }
    ),
)

TEST_STOCK_TABLE2 = Table.from_df_and_cols(
    columns=[
        TableColumnMetadata(label="Security", col_type=TableColumnType.STOCK),
        TableColumnMetadata(label="Earnings Summary", col_type=TableColumnType.STRING),
    ],
    data=pd.DataFrame(
        data={
            "Security": {
                0: STOCK4_alt,
                1: STOCK5,
            },
            "Earnings Summary": {
                0: "blah1",
                1: "blah2",
            },
        }
    ),
)
