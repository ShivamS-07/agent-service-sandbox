import datetime

import pandas as pd

from agent_service.io_types.table import Table, TableColumn, TableColumnType

TEST_STOCK_DATE_TABLE1 = Table(
    columns=[
        TableColumn(label="Date", col_type=TableColumnType.DATE, unit=None),
        TableColumn(label="Security", col_type=TableColumnType.STOCK, unit=None),
        TableColumn(label="Close Price", col_type=TableColumnType.FLOAT, unit=None),
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
                0: 72,
                1: 72,
                2: 72,
                3: 72,
                4: 72,
                5: 76,
                6: 76,
                7: 76,
                8: 76,
                9: 76,
                10: 78,
                11: 78,
                12: 78,
                13: 78,
                14: 78,
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

TEST_STOCK_DATE_TABLE2 = Table(
    columns=[
        TableColumn(label="Date", col_type=TableColumnType.DATE, unit=None),
        TableColumn(label="Security", col_type=TableColumnType.STOCK, unit=None),
        TableColumn(label="Open Price", col_type=TableColumnType.FLOAT, unit=None),
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
                0: 112,
                1: 112,
                2: 112,
                3: 112,
                4: 124,
                5: 124,
                6: 124,
                7: 124,
                8: 149,
                9: 149,
                10: 149,
                11: 149,
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

TEST_STOCK_TABLE1 = Table(
    columns=[
        TableColumn(label="Security", col_type=TableColumnType.STOCK),
        TableColumn(label="News Summary", col_type=TableColumnType.STRING),
    ],
    data=pd.DataFrame(
        data={
            "Security": {
                0: 112,
                1: 124,
                2: 149,
            },
            "News Summary": {
                0: "blah1",
                1: "blah2",
                2: "blah3",
            },
        }
    ),
)

TEST_STOCK_TABLE2 = Table(
    columns=[
        TableColumn(label="Security", col_type=TableColumnType.STOCK),
        TableColumn(label="Earnings Summary", col_type=TableColumnType.STRING),
    ],
    data=pd.DataFrame(
        data={
            "Security": {
                0: 112,
                1: 124,
            },
            "Earnings Summary": {
                0: "blah1",
                1: "blah2",
            },
        }
    ),
)
