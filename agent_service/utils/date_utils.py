import datetime
from typing import Optional, Tuple

import pytz
from dateutil.parser import parse as date_parse
from google.protobuf.timestamp_pb2 import Timestamp

DAYS_LOOKUP = {"D": 1, "W": 7, "M": 30, "Y": 365}


def get_now_utc(strip_tz: bool = False) -> datetime.datetime:
    now = datetime.datetime.now(tz=datetime.timezone.utc)
    if strip_tz:
        now = now.replace(tzinfo=None)
    return now


def get_year_quarter_for_date(date: datetime.date) -> Tuple[int, int]:
    quarter = (date.month - 1) // 3 + 1
    return (date.year, quarter)


def date_to_pb_timestamp(dt: Optional[datetime.date]) -> Timestamp:
    ts = Timestamp()
    if not dt:
        return ts

    ts.FromDatetime(datetime.datetime(dt.year, dt.month, dt.day))
    return ts


def convert_horizon_to_days(horizon: str) -> int:
    return int(horizon[:-1]) * DAYS_LOOKUP[horizon[-1]]


def convert_horizon_to_date(horizon: str) -> datetime.date:
    days = convert_horizon_to_days(horizon)
    return datetime.date.today() - datetime.timedelta(days=days)


def timezoneify(dt: datetime.datetime) -> datetime.datetime:
    if not dt.tzinfo:
        return dt.replace(tzinfo=datetime.timezone.utc)
    return dt


def parse_date_str_in_utc(date_str: str) -> datetime.datetime:
    return date_parse(date_str).astimezone(pytz.utc)
