import datetime


def get_now_utc() -> datetime.datetime:
    return datetime.datetime.now(tz=datetime.timezone.utc)
