import uuid
from typing import List
from unittest.mock import MagicMock, patch

from agent_service.endpoints.authz_helper import User
from agent_service.endpoints.models import (
    AgentNotificationEmail,
    ChatWithAgentRequest,
    MediaType,
    UpdateAgentRequest,
)
from agent_service.types import Notification
from agent_service.utils.constants import MEDIA_TO_MIMETYPE
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
        self.assertFalse(all_agents.agents[0].deleted)

        self.delete_agent(agent_id=agent_id)

        all_agents = self.get_all_agents(user=user)
        self.assertEqual(len(all_agents.agents), 0)

        self.restore_agent(agent_id=agent_id)
        all_agents = self.get_all_agents(user=user)
        self.assertEqual(all_agents.agents[0].agent_id, agent_id)
        self.assertEqual(all_agents.agents[0].agent_name, new_agent_name)
        self.assertFalse(all_agents.agents[0].deleted)

    def test_get_agent(self):
        agent_id = "4bc82c91-3946-4ae7-b05b-942a59701d49"
        user = User(
            user_id="ac7c96d7-3e57-40e7-a1a5-8e2ce5e23639",
            is_admin=False,
            is_super_admin=False,
            auth_token="",
        )
        agent_metadata = self.get_agent(user=user, agent_id=agent_id)
        self.assertIsNotNone(agent_metadata.cost_info)
        cost_info_val = agent_metadata.cost_info[0]
        self.assertTrue("label" in cost_info_val)
        self.assertTrue("val" in cost_info_val)
        self.assertGreater(cost_info_val["val"], 0)

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
        agent_id = "6a69d259-8d4c-4056-8f60-712f42f7546f"
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

        service_version = "0.0.544"
        test_run_info = self.get_info_for_test_suite_run(
            service_version=service_version
        ).test_suite_run_info
        self.assertTrue(test_run_info)

    def test_get_test_suite_runs(self):

        test_suite_runs = self.agent_service_impl.get_test_suite_runs().test_suite_runs
        self.assertTrue(test_suite_runs)

    def test_get_test_case_info(self):

        test_name = "test_pe_nvda_feb_2024"
        test_case_info = self.get_info_for_test_case(test_name=test_name).test_case_info
        self.assertTrue(test_case_info)

    def test_get_test_cases(self):
        test_cases = self.agent_service_impl.get_test_cases().test_cases
        self.assertTrue(test_cases)

    def test_convert_markdown(self):

        test_content = """
        # This is a test h1.
        ## This is a test h2.

        ### Test List
        - List Item 1
        - List Item 2
            - List Item 2.5
        - List Item 3

        `Inline code`

        ```
        Code block
        ```

        **Bold** __Bold__ *Italic* _Italic_ ***Bold and Italic***

        > This is a block quote.
        >
        >> This is a nested block quote.

        And finally, this is a [Link](https://example.com).
        """
        converted_data, mimetype = self.loop.run_until_complete(
            self.agent_service_impl.convert_markdown(test_content, MediaType.DOCX)
        )
        docx_magic = bytes([0x50, 0x4B, 0x03, 0x04])

        self.assertEqual(converted_data[:4], docx_magic)
        self.assertEqual(mimetype, MEDIA_TO_MIMETYPE["docx"])

        converted_data, mimetype = self.loop.run_until_complete(
            self.agent_service_impl.convert_markdown(test_content, MediaType.TXT)
        )

        self.assertIn("This is a test h1.", converted_data)
        self.assertEqual(mimetype, MEDIA_TO_MIMETYPE["plain"])

    def test_get_canned_prompts(self):
        canned_prompts = self.get_canned_prompts()
        self.assertEqual(len(canned_prompts.canned_prompts), 5)

    def test_agent_notification(self):
        # create the agent
        test_user = str(uuid.uuid4())
        user = User(user_id=test_user, is_admin=False, is_super_admin=False, auth_token="")
        res = self.create_agent(user=user)
        agent_id = res.agent_id

        # insert a email notification
        emails = ["test.email@test.com", "test.email.2@test.com"]
        self.loop.run_until_complete(
            self.pg.set_agent_subscriptions(agent_id=agent_id, emails=emails)
        )

        # get notification event info
        notif_res: List[AgentNotificationEmail] = self.loop.run_until_complete(
            self.pg.get_agent_subscriptions(agent_id=agent_id)
        )
        self.assertIsNotNone(notif_res)
        self.assertEqual(len(notif_res), 2)
        for email_notif in notif_res:
            # make sure that the emails are correct
            self.assertTrue(email_notif.email in emails)
        # add a new subscription
        new_email = "test.email.3@test.com"
        self.loop.run_until_complete(
            self.pg.set_agent_subscriptions(agent_id=agent_id, emails=[new_email])
        )
        notif_res: List[AgentNotificationEmail] = self.loop.run_until_complete(
            self.pg.get_agent_subscriptions(agent_id=agent_id)
        )
        self.assertEqual(len(notif_res), 3)
        res_emails = [agent_notif.email for agent_notif in notif_res]
        self.assertTrue(new_email in res_emails)
        # delete an email from the subscription
        email_to_delete = "test.email@test.com"
        self.loop.run_until_complete(
            self.pg.delete_agent_emails(agent_id=agent_id, email=email_to_delete)
        )
        notif_res: List[AgentNotificationEmail] = self.loop.run_until_complete(
            self.pg.get_agent_subscriptions(agent_id=agent_id)
        )
        self.assertEqual(len(notif_res), 2)
        res_emails = [agent_notif.email for agent_notif in notif_res]
        self.assertTrue(email_to_delete not in res_emails)

        self.delete_agent(agent_id=agent_id)
