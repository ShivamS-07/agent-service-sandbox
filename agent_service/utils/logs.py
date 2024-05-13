import functools
import logging
import sys
import time
from typing import Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


def init_logging(log_handler: logging.Handler, log_level: int = logging.INFO) -> None:
    """Initialize logging with the specified log handler."""
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # add thread id to confirm parallel threads are running
    formatter = logging.Formatter(
        fmt="%(asctime)s %(process)8d %(thread)d %(levelname)8s  %(name)s : %(message)s"
    )
    log_handler.setFormatter(formatter)
    root_logger.addHandler(log_handler)
    logger.info("Initialized logging")


def init_stdout_logging(log_level: int = logging.INFO) -> None:
    handler = logging.StreamHandler(sys.stdout)
    init_logging(handler, log_level)


def async_perf_logger(func: Callable[..., T], level: int = logging.INFO) -> Callable[..., T]:
    @functools.wraps(func)
    async def wrapper_timer(*args, **kwargs) -> T:  # type: ignore
        start_time = time.perf_counter()
        value: T = await func(*args, **kwargs)  # type: ignore
        end_time = time.perf_counter()
        logger.log(level, f"{func.__qualname__}: {end_time - start_time}s")
        return value

    return wrapper_timer  # type: ignore
