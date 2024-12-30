import unittest
import uuid

from agent_service.utils.enablement_function_registry import ENABLEMENT_FUNCTION_REGISTRY


class TestEnablement(unittest.TestCase):
    def test_sample_plans_dont_crash(self):
        for enabler_func in ENABLEMENT_FUNCTION_REGISTRY.values():
            enabler_func({"user_id": str(uuid.uuid4())})


if __name__ == "__main__":
    unittest.main()
