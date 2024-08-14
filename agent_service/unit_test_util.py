import inspect
import logging
import sys


def _check_if_test() -> bool:

    for key in ["unittest_parallel"]:
        if key in sys.modules:
            print("\n\n================ TESTING DETECTED!!! ====================")
            print("module", key, "detected")
            print("================ TESTING DETECTED!!! ====================\n\n")
            return True

    for frame in inspect.stack():
        for key in ["service/tests/", "service/regression_test", "unittest", "pytest"]:
            if key in frame.filename:
                print("\n\n================ TESTING DETECTED!!! ====================")
                print("key:", key, "found in frame:", frame)
                print("================ TESTING DETECTED!!! ====================\n\n")

                return True
    logging.info("This is not a test")
    return False


RUNNING_IN_UNIT_TEST = _check_if_test()
