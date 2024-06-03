import uuid
from unittest.mock import MagicMock, patch

from agent_service.endpoints.authz_helper import User
from agent_service.endpoints.models import ChatWithAgentRequest, UpdateAgentRequest
from tests.test_agent_service_impl_base import TestAgentServiceImplBase


async def generate_initial_preplan_response(chat_context):
    return "Hi this is Warren AI, how can I help you today?"


async def send_chat_message(*args, **kwargs):
    return None


class TestAgentServiceImpl(TestAgentServiceImplBase):
    def test_agent_crud(self):
        test_user = str(uuid.uuid4())
        user = User(user_id=test_user, is_admin=False, is_super_admin=False, auth_token="")
        res = self.create_agent(user=user)
        agent_id = res.agent_id
        all_agents = self.get_all_agents(user=user)
        self.assertEqual(all_agents.agents[0].agent_id, agent_id)

        new_agent_name = "my new agent name :)"
        self.update_agent(agent_id=agent_id, req=UpdateAgentRequest(agent_name=new_agent_name))

        all_agents = self.get_all_agents(user=user)
        self.assertEqual(all_agents.agents[0].agent_id, agent_id)
        self.assertEqual(all_agents.agents[0].agent_name, new_agent_name)

        self.delete_agent(agent_id=agent_id)

        all_agents = self.get_all_agents(user=user)
        self.assertEqual(len(all_agents.agents), 0)

    @patch("agent_service.agent_service_impl.Chatbot")
    @patch("agent_service.agent_service_impl.send_chat_message")
    def test_chat_with_agent(self, mock_send_chat_message: MagicMock, MockChatBot: MagicMock):
        mock_chatbot = MockChatBot.return_value
        mock_chatbot.generate_initial_preplan_response.side_effect = (
            generate_initial_preplan_response
        )

        mock_send_chat_message.side_effect = send_chat_message

        test_user = str(uuid.uuid4())
        user = User(user_id=test_user, is_admin=False, is_super_admin=False, auth_token="")
        res = self.create_agent(user=user)
        agent_id = res.agent_id

        res = self.chat_with_agent(
            req=ChatWithAgentRequest(
                agent_id=agent_id, prompt="Please help me!", is_first_prompt=True
            ),
            user=user,
        )
        self.assertTrue(res.success)

        res = self.chat_with_agent(
            req=ChatWithAgentRequest(agent_id=agent_id, prompt="Let me try again"), user=user
        )
        self.assertTrue(res.success)

        # it will only have 3 messages because the second prompt has no response in the test
        res = self.get_chat_history(agent_id=agent_id)
        self.assertEqual(len(res.messages), 3)

    def test_agent_output(self):
        agent_id = "a440f037-a0d6-4fb1-9abc-1aec0a8db684"
        res = self.get_agent_output(agent_id=agent_id)
        self.assertGreater(len(res.outputs), 0)

    def test_agent_task_output(self):
        agent_id = "4b928481-4b8e-4d7f-9224-62196213e38c"
        plan_run_id = "f99eb6ee-b7ea-42a3-a889-734e2d20b24c"
        task_id = "a80de37f-60c1-4385-95ea-6e4fadbdf161"
        res = self.get_agent_task_output(
            agent_id=agent_id, plan_run_id=plan_run_id, task_id=task_id
        )
        self.assertIsNotNone(res)
