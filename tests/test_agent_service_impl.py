import uuid
from unittest.mock import MagicMock, patch

from agent_service.endpoints.authz_helper import User
from agent_service.endpoints.models import ChatWithAgentRequest, UpdateAgentRequest
from agent_service.types import Notification
from tests.test_agent_service_impl_base import TestAgentServiceImplBase


async def generate_initial_preplan_response(chat_context):
    return "Hi this is Warren AI, how can I help you today?"


async def generate_name_for_agent(agent_id, chat_context, existing_names, gpt_service_stub):
    return "Macroeconomic News"


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
    @patch("agent_service.agent_service_impl.generate_name_for_agent")
    def test_chat_with_agent(
        self,
        mock_generate_name_for_agent: MagicMock,
        mock_send_chat_message: MagicMock,
        MockChatBot: MagicMock,
    ):
        mock_chatbot = MockChatBot.return_value
        mock_chatbot.generate_initial_preplan_response.side_effect = (
            generate_initial_preplan_response
        )

        mock_generate_name_for_agent.side_effect = generate_name_for_agent

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
        self.assertTrue(res.name)
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

    def test_pg_notification_event_info(self):
        # create a new agent
        test_user = str(uuid.uuid4())
        user = User(user_id=test_user, is_admin=False, is_super_admin=False, auth_token="")
        res = self.create_agent(user=user)
        agent_id = res.agent_id

        # insert a notification
        self.loop.run_until_complete(
            self.pg.insert_notifications(
                notifications=[Notification(agent_id=agent_id, summary="Hello!", unread=True)]
            )
        )

        # get notification event info
        notif_res = self.loop.run_until_complete(
            self.pg.get_notification_event_info(agent_id=agent_id)
        )
        self.assertIsNotNone(notif_res)
        self.assertEqual(notif_res.get("unread_count"), 1)
        self.assertEqual(notif_res.get("latest_notification_string"), "Hello!")

        self.delete_agent(agent_id=agent_id)

    def test_agent_debug_no_info(self):
        # Agent with no info in logs
        agent_id = "a440f037-a0d6-4fb1-9abc-1aec0a8db684"
        debug_info = self.get_agent_debug_info(agent_id=agent_id)
        self.assertIsNotNone(debug_info)

    def test_get_agent_debug_info(self):

        agent_id = "7aa1a451-5811-4881-9d39-e596efb538cd"
        debug_info = self.get_agent_debug_info(agent_id=agent_id)
        self.assertIsNotNone(debug_info)
        self.assertIsNotNone(debug_info.debug.run_execution_plans)
        self.assertIsNotNone(debug_info.debug.create_execution_plans)

    def test_get_test_run_info(self):

        test_run_id = "8e1b5600-b6a5-44df-927f-ec3c90c32424"
        test_run_info = self.agent_service_impl.get_info_for_test_suite_run(
            test_run_id=test_run_id
        ).test_suite_run_info
        self.assertTrue(test_run_info)

    def test_get_test_suite_runs(self):

        test_suite_run_ids = self.agent_service_impl.get_test_suite_runs().test_suite_run_ids
        self.assertTrue(test_suite_run_ids)

    def test_get_test_case_info(self):

        test_name = "test_pe_nvda_feb_2024"
        test_case_info = self.agent_service_impl.get_info_for_test_case(
            test_name=test_name
        ).test_case_info
        self.assertTrue(test_case_info)
