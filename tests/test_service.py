import datetime
import logging
import unittest
from unittest.mock import patch
from uuid import uuid4

from dateutil.parser import parse
from fastapi.testclient import TestClient
from httpx import Response

from agent_service.endpoints.authz_helper import User
from agent_service.utils.environment import EnvironmentUtils
from agent_service.utils.postgres import DEFAULT_AGENT_NAME, get_psql
from application import DUMMY_RESP_FROM_GPT, application

logger = logging.getLogger(__name__)


class TestAgentService(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

        cls.header = {"Authorization": "test token"}

        EnvironmentUtils.is_testing = True
        cls.pg = get_psql(skip_commit=True)

        cls.patches = []

        with patch(
            "agent_service.endpoints.authz_helper.get_keyid_to_key_map"
        ) as mock_get_keyid_to_key_map:
            mock_get_keyid_to_key_map.return_value = {}

            logger.info("Launching Fastapi server")
            cls.client = TestClient(application)

    def setUp(self) -> None:
        self.extract_user_patch = patch(
            "agent_service.endpoints.authz_helper.extract_user_from_jwt"
        ).start()
        self.extract_user_patch.return_value = ""
        self.extract_user_patch.start()

    def tearDown(self) -> None:
        self.extract_user_patch.stop()
        for active_patch in self.patches:
            active_patch.stop()
        self.pg.close()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.client.close()

    ############################################################################
    # Helper methods that could be shared by subclasses
    ############################################################################
    def create_random_user(self) -> str:
        user_id = str(uuid4())

        sql = """
            INSERT INTO user_service.users (id) VALUES (%s)
        """
        self.pg.generic_write(sql, (user_id,))
        return user_id

    def post(self, requesting_user_id: str, url: str, body: dict = {}) -> Response:
        self.extract_user_patch.return_value = User(
            user_id=requesting_user_id, auth_token="test token"
        )
        resp = self.client.request("POST", url, json=body, headers=self.header)
        self.assert_resp_ok(resp)
        return resp

    def get(self, requesting_user_id: str, url: str, body: dict = {}) -> Response:
        self.extract_user_patch.return_value = User(
            user_id=requesting_user_id, auth_token="test token"
        )
        resp = self.client.request("GET", url, json=body, headers=self.header)
        self.assert_resp_ok(resp)
        return resp

    def delete(self, requesting_user_id: str, url: str, body: dict = {}) -> Response:
        self.extract_user_patch.return_value = User(
            user_id=requesting_user_id, auth_token="test token"
        )
        resp = self.client.request("DELETE", url, json=body, headers=self.header)
        self.assert_resp_ok(resp)
        return resp

    def put(self, requesting_user_id: str, url: str, body: dict = {}) -> Response:
        self.extract_user_patch.return_value = User(
            user_id=requesting_user_id, auth_token="test token"
        )
        resp = self.client.request("PUT", url, json=body, headers=self.header)
        self.assert_resp_ok(resp)
        return resp

    def assert_resp_ok(self, resp: Response) -> None:
        assert resp.status_code in (200, 201), f"Response status code is {resp.status_code}"

    ############################################################################
    # Tests
    ############################################################################
    def test_create_get_update_delete_agent(self) -> None:
        user = self.create_random_user()

        # Create an agent
        resp1 = self.post(user, "/agent/create-agent", {"first_prompt": "Hello!"})
        agent_id: str = resp1.json()["agent_id"]

        # Get all agents
        resp2 = self.get(user, "/agent/get-all-agents")
        agents = resp2.json()["agents"]
        self.assertEqual(len(agents), 1)
        self.assertEqual(agents[0]["agent_id"], agent_id)
        self.assertEqual(agents[0]["agent_name"], DEFAULT_AGENT_NAME)

        # Update agent name
        new_agent_name = "New Agent Name"
        resp3 = self.put(
            user, "/agent/update-agent", {"agent_id": agent_id, "agent_name": new_agent_name}
        )
        self.assertTrue(resp3.json()["success"])

        # Get again
        resp4 = self.get(user, "/agent/get-all-agents")
        agents = resp4.json()["agents"]
        self.assertEqual(len(agents), 1)
        self.assertEqual(agents[0]["agent_name"], new_agent_name)

        # Delete
        resp5 = self.delete(user, "/agent/delete-agent", {"agent_id": agent_id})
        self.assertTrue(resp5.json()["success"])

        # Get again
        resp6 = self.get(user, "/agent/get-all-agents")
        self.assertEqual(len(resp6.json()["agents"]), 0)

    def test_chat_with_agent(self) -> None:
        user = self.create_random_user()

        # Create an agent
        resp1 = self.post(user, "/agent/create-agent", {"first_prompt": "Hello!"})
        agent_id: str = resp1.json()["agent_id"]

        # Get all chats
        resp2 = self.get(user, "/agent/get-chat-history", {"agent_id": agent_id})
        messages = resp2.json()["messages"]
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["message"], "Hello!")
        self.assertEqual(messages[1]["message"], DUMMY_RESP_FROM_GPT)

        # Chat
        # NOTE: dates filter doesn't work when `skip_commit=True` because the time is the same
        resp3 = self.post(
            user,
            "/agent/chat-with-agent",
            {"agent_id": agent_id, "prompt": "Hi Boosted"},
        )
        new_messages = resp3.json()["success"]

        # Get all chats again
        start_time = parse(messages[1]["message_time"]) + datetime.timedelta(milliseconds=1)
        resp4 = self.get(
            user, "/agent/get-chat-history", {"agent_id": agent_id, "start": start_time.isoformat()}
        )
        new_messages = resp4.json()["messages"]
        self.assertEqual(len(new_messages), 2)
        self.assertEqual(new_messages[0]["message"], "Hi Boosted")
        self.assertEqual(new_messages[1]["message"], DUMMY_RESP_FROM_GPT)


if __name__ == "__main__":
    unittest.main()
