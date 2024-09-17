import inspect
import logging
from typing import Any

logger = logging.getLogger(__name__)


def _super_print(*args: Any) -> None:
    print(*args)
    logger.warning(f"{args}")


def _check_if_test() -> bool:

    # check the stack to see if any unit-test like directories exist
    for frame in inspect.stack():
        for key in [
            "service/tests/",
            "service/regression_test/",
            "/unittest/",
            "/pytest/",
            "/_pytest/",
        ]:
            if key in frame.filename:
                _super_print("\n\n================ TESTING DETECTED!!! ====================")
                _super_print("key:", key, "found in frame:", frame)
                _super_print("================ TESTING DETECTED!!! ====================\n\n")

                return True
    logging.info("This is not a test")
    return False


RUNNING_IN_UNIT_TEST = _check_if_test()
