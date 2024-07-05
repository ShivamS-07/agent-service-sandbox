import inspect


def _check_if_test() -> bool:
    for frame in inspect.stack():
        if "unittest" in frame.filename or "pytest" in frame.filename:
            return True
    return False


RUNNING_IN_UNIT_TEST = _check_if_test()
