import datetime
from typing import Tuple


def get_now_utc() -> datetime.datetime:
    return datetime.datetime.now(tz=datetime.timezone.utc)


def get_year_quarter_for_date(date: datetime.date) -> Tuple[int, int]:
    quarter = (date.month - 1) // 3 + 1
    return (date.year, quarter)
