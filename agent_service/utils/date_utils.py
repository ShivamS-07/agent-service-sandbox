import datetime
import logging
import time
from typing import List, Optional, Tuple, cast

import pytz
from dateutil.parser import parse as date_parse
from google.protobuf.timestamp_pb2 import Timestamp

logger = logging.getLogger(__name__)

DAYS_LOOKUP = {"D": 1, "W": 7, "M": 30, "Q": 90, "Y": 365}

QUARTER_START_DATES = {(1, 1): "Q1", (4, 1): "Q2", (7, 1): "Q3", (10, 1): "Q4"}
QUARTER_END_DATES = {(3, 31): "Q1", (6, 30): "Q2", (9, 30): "Q3", (12, 31): "Q4"}

real_datetime_class = datetime.datetime
real_date_class = datetime.date
real_date_new = datetime.date.__new__

# set us up to override the current date/time
real_now = datetime.datetime.now
real_utcnow = datetime.datetime.utcnow
real_today = datetime.date.today

_mock_current_time: Optional[datetime.datetime] = None
_use_mock_time = False

WARN_NTH = 5
_warn_counter = 0


def warn_every_nth(n: int, s: str) -> None:
    global _warn_counter
    if _warn_counter % n == 0:
        print(s)
        logger.warning(s)

    _warn_counter += 1


# to make this useful you must import datetime and MockDate
# before any modules you want to override their date usage
# like so: datetime.date = MockDate  # type:ignore
# or call enable_mock_time()
class _MockDate(real_date_class):
    @classmethod
    def today(cls) -> datetime.date:  # type:ignore
        if _use_mock_time:
            val = get_now_utc().date()
            s = f"WARNING USING MOCK TIME today() {val}"
            warn_every_nth(WARN_NTH, s)
            return val

        return real_today()

    def __new__(cls, year, month, day) -> datetime.date:  # type:ignore
        # we only needed to override the today() static/class function
        # we actually still want to create plain date objects though
        tmp_date = real_date_new(real_date_class, year, month, day)
        return tmp_date


MockDate = _MockDate


def enable_mock_time() -> None:
    s = "WARNING ENABLING MOCK TIME"
    warn_every_nth(1, s)
    global _use_mock_time
    _use_mock_time = True
    global MockDate
    MockDate = _MockDate
    datetime.date = MockDate  # type: ignore


def disable_mock_time() -> None:
    s = "WARNING DISABLING MOCK TIME"
    warn_every_nth(1, s)

    global _use_mock_time
    _use_mock_time = False
    datetime.datetime = real_datetime_class  # type: ignore
    datetime.date = real_date_class  # type: ignore

    global MockDate
    MockDate = real_date_class  # type: ignore


# this is needed to make time still 'flow' when using mocked time
orig_time_counter = time.perf_counter()


def set_mock_time(dt: datetime.datetime) -> None:
    global orig_time_counter
    orig_time_counter = time.perf_counter()

    global _mock_current_time
    _mock_current_time = dt
    s = f"set_mock_time: {_mock_current_time}"
    warn_every_nth(1, s)


# DG: originally I monkeypatched datetime.datetime.now/utcnow
# but various libraries didnt like it, I could not work around boto's complaint because
# it was rejecting on the server side
# when receiving any auth tokens created too far in the "past"
# fortunately most of our code uses get_now_utc() below
def increment_mock_time(td: Optional[datetime.timedelta] = None) -> None:
    global _mock_current_time

    if _mock_current_time is None:
        set_mock_time(real_now())

    # simulate time flowing
    if td is None:
        global orig_time_counter
        current_time_counter = time.perf_counter()
        seconds = current_time_counter - orig_time_counter

        # pretend at least 1 microsecond passes between calls
        td = datetime.timedelta(seconds=seconds, microseconds=1)

        orig_time_counter = current_time_counter

    _mock_current_time = cast(datetime.datetime, _mock_current_time)
    _mock_current_time = _mock_current_time + td


def mock_now() -> datetime.datetime:  # type:ignore
    curr_time = _mock_current_time

    if curr_time is None:
        curr_time = real_now()
        set_mock_time(curr_time)

    # simulate time flowing
    increment_mock_time()

    s = f"WARNING USING MOCK TIME mock_now() {curr_time}"
    warn_every_nth(WARN_NTH, s)
    return curr_time


def mock_utcnow() -> datetime.datetime:
    now = mock_now()

    # close enough assuming now is EST
    utcnow = now + datetime.timedelta(hours=4)
    return utcnow


def get_now_utc(strip_tz: bool = False) -> datetime.datetime:
    if _use_mock_time:
        now = mock_utcnow()
    else:
        now = datetime.datetime.now(tz=datetime.timezone.utc)
    if strip_tz:
        now = now.replace(tzinfo=None)
    return now


def get_next_quarter(quarter: str) -> str:
    if quarter[-1] != "4":
        return quarter[:-1] + str(int(quarter[-1]) + 1)
    else:
        return str(int(quarter[:4]) + 1) + "Q1"


def get_prev_quarter(quarter: str) -> str:
    if quarter[-1] != "1":
        return quarter[:-1] + str(int(quarter[-1]) - 1)
    else:
        return str(int(quarter[:4]) - 1) + "Q4"


def get_year_quarter_for_date(date: datetime.date) -> Tuple[int, int]:
    quarter = (date.month - 1) // 3 + 1
    return (date.year, quarter)


def get_month_day(date: datetime.date) -> Tuple[int, int]:
    return (date.month, date.day)


def is_valid_quarter_range(start_date: datetime.date, end_date: datetime.date) -> bool:
    return (
        start_date < end_date
        and get_month_day(start_date) in QUARTER_START_DATES
        and get_month_day(end_date) in QUARTER_END_DATES
    )


def get_quarters_for_time_range(start_date: datetime.date, end_date: datetime.date) -> List[str]:
    # Note there are hardcoded dependencies on this in earnings tools
    if not is_valid_quarter_range(start_date, end_date):
        return []
    start_quarter = str(start_date.year) + QUARTER_START_DATES[get_month_day(start_date)]
    end_quarter = str(end_date.year) + QUARTER_END_DATES[get_month_day(end_date)]
    quarters = [start_quarter]
    curr_quarter = start_quarter
    while curr_quarter != end_quarter:
        curr_quarter = get_next_quarter(curr_quarter)
        quarters.append(curr_quarter)
    return quarters


def date_to_pb_timestamp(dt: Optional[datetime.date]) -> Timestamp:
    ts = Timestamp()
    if not dt:
        return ts

    ts.FromDatetime(datetime.datetime(dt.year, dt.month, dt.day))
    return ts


def convert_horizon_to_days(horizon: str) -> int:
    return int(horizon[:-1]) * DAYS_LOOKUP[horizon[-1]]


def convert_horizon_to_date(
    horizon: str, end_date: Optional[datetime.date] = None
) -> datetime.date:
    days = convert_horizon_to_days(horizon)
    if end_date is None:
        end_date = datetime.date.today()
    return end_date - datetime.timedelta(days=days)


def timezoneify(dt: datetime.datetime) -> datetime.datetime:
    if not dt.tzinfo:
        return dt.replace(tzinfo=datetime.timezone.utc)
    return dt


def parse_date_str_in_utc(date_str: str) -> datetime.datetime:
    return date_parse(date_str).astimezone(pytz.utc)
