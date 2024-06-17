import datetime
from typing import Optional, Tuple

from google.protobuf.timestamp_pb2 import Timestamp


def get_now_utc() -> datetime.datetime:
    return datetime.datetime.now(tz=datetime.timezone.utc)


def get_year_quarter_for_date(date: datetime.date) -> Tuple[int, int]:
    quarter = (date.month - 1) // 3 + 1
    return (date.year, quarter)


def date_to_pb_timestamp(dt: Optional[datetime.date]) -> Timestamp:
    ts = Timestamp()
    if not dt:
        return ts

    ts.FromDatetime(datetime.datetime(dt.year, dt.month, dt.day))
    return ts
