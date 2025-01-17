import functools
import logging
import sys
import time
from typing import Callable, Optional, ParamSpec, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


def init_logging(
    log_handler: logging.Handler,
    log_level: int = logging.INFO,
    disable_prefect_logging: bool = True,
    formatter: Optional[logging.Formatter] = None,
) -> None:
    """Initialize logging with the specified log handler."""
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # add thread id to confirm parallel threads are running
    if not formatter:
        formatter = logging.Formatter(
            fmt="%(asctime)s %(process)8d %(thread)d %(levelname)8s  %(name)s %(filename)s:%(lineno)d : %(message)s"
        )
    log_handler.setFormatter(formatter)
    if disable_prefect_logging:
        # Remove the prefect handler
        root_logger.handlers = []
    root_logger.addHandler(log_handler)
    logger.info("Initialized logging")


def init_stdout_logging(
    log_level: int = logging.INFO,
    disable_prefect_logging: bool = True,
) -> None:
    handler = logging.StreamHandler(sys.stdout)
    init_logging(handler, log_level, disable_prefect_logging=disable_prefect_logging)


def init_test_logging(
    log_level: int = logging.INFO,
    disable_prefect_logging: bool = True,
) -> None:
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(fmt="%(asctime)s %(levelname)8s  %(name)s : %(message)s")
    init_logging(
        handler, log_level, disable_prefect_logging=disable_prefect_logging, formatter=formatter
    )


P = ParamSpec("P")


def async_perf_logger(func: Callable[P, T], level: int = logging.INFO) -> Callable[P, T]:
    @functools.wraps(func)
    async def wrapper_timer(*args, **kwargs) -> T:  # type: ignore
        start_time = time.perf_counter()
        value: T = await func(*args, **kwargs)  # type: ignore
        end_time = time.perf_counter()
        logger.log(level, f"{func.__qualname__}: {end_time - start_time}s")
        return value

    return wrapper_timer  # type: ignore
