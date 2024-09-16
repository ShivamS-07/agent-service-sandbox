import inspect
import logging
import sys
from typing import Any

logger = logging.getLogger(__name__)


def _super_print(*args: Any) -> None:
    print(*args)
    logger.warning(f"{args}")


def _check_if_test() -> bool:

    for key in ["unittest_parallel", "pytest"]:
        if key in sys.modules:
            _super_print("\n\n================ TESTING DETECTED!!! ====================")
            _super_print("module", key, "detected")
            _super_print("================ TESTING DETECTED!!! ====================\n\n")
            return True

    for frame in inspect.stack():
        for key in ["service/tests/", "service/regression_test", "unittest", "pytest"]:
            if key in frame.filename:
                _super_print("\n\n================ TESTING DETECTED!!! ====================")
                _super_print("key:", key, "found in frame:", frame)
                _super_print("================ TESTING DETECTED!!! ====================\n\n")

                return True
    logging.info("This is not a test")
    return False


RUNNING_IN_UNIT_TEST = _check_if_test()
