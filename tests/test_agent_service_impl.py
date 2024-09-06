import uuid
from typing import List
from unittest.mock import MagicMock, patch

from agent_service.endpoints.authz_helper import User
from agent_service.endpoints.models import (
    AgentFeedback,
    AgentNotificationEmail,
    ChatWithAgentRequest,
    MediaType,
    NotificationUser,
    SetAgentFeedBackRequest,
    UpdateAgentRequest,
)
from agent_service.planner.constants import FirstAction
from agent_service.planner.planner_types import ExecutionPlan
from agent_service.types import Notification
from agent_service.utils.constants import MEDIA_TO_MIMETYPE
from tests.test_agent_service_impl_base import TestAgentServiceImplBase

AGENT_NAME = "Macroeconomic News"


def find_item(lst, field, value):
    for item in lst:
        if getattr(item, field) == value:
            return item
    return None


async def decide_action(chat_context):
    return FirstAction.NONE


async def generate_first_response_refer(chat_context):
    return "I can help you with that. Please contact the support team. "


async def generate_first_response_none(chat_context):
    return "I'm sorry, I don't understand. Can you please rephrase?"


async def generate_first_response_notification(chat_context):
    return "I can set up a notification for you. Please provide me with the details."


async def generate_initial_preplan_response(chat_context):
    return "Hi this is Warren AI, how can I help you today?"


async def generate_name_for_agent(
    agent_id, chat_context, existing_names, gpt_service_stub, user_id
):
    return AGENT_NAME


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

        plan_id = str(uuid.uuid4())
        plan_run_id = str(uuid.uuid4())
        self.terminate_agent(agent_id=agent_id, plan_id=plan_id, plan_run_id=plan_run_id)
        is_cancelled = self.loop.run_until_complete(
            self.pg.is_cancelled(ids_to_check=[plan_id, plan_run_id])
        )
        self.assertTrue(is_cancelled)

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
        agent_metadata = self.get_agent(agent_id=agent_id)
        self.assertIsNotNone(agent_metadata.cost_info)
        cost_info_val = agent_metadata.cost_info[0]
        self.assertTrue("label" in cost_info_val)
        self.assertTrue("val" in cost_info_val)
        self.assertGreater(cost_info_val["val"], 0)

    @patch("agent_service.agent_service_impl.Chatbot")
    @patch("agent_service.agent_service_impl.FirstActionDecider")
    @patch("agent_service.agent_service_impl.send_chat_message")
    @patch("agent_service.agent_service_impl.generate_name_for_agent")
    def test_chat_with_agent(
        self,
        mock_generate_name_for_agent: MagicMock,
        mock_send_chat_message: MagicMock,
        MockFirstActionDecider: MagicMock,
        MockChatBot: MagicMock,
    ):

        mock_generate_name_for_agent.side_effect = generate_name_for_agent
        mock_firstactiondecider = MockFirstActionDecider.return_value
        mock_firstactiondecider.decide_action = decide_action

        mock_chatbot = MockChatBot.return_value
        mock_chatbot.generate_initial_preplan_response.side_effect = (
            generate_initial_preplan_response
        )
        mock_chatbot.generate_first_response_refer.side_effect = generate_first_response_refer
        mock_chatbot.generate_first_response_none.side_effect = generate_first_response_none
        mock_chatbot.generate_first_response_notification.side_effect = (
            generate_first_response_notification
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

        # name generation is running in the background now
        res2 = self.get_agent(agent_id=agent_id)
        self.assertEqual(res2.agent_name, AGENT_NAME)

    def test_agent_plan_output(self):
        agent_id = "6a69d259-8d4c-4056-8f60-712f42f7546f"
        res = self.get_agent_plan_output(agent_id=agent_id)
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

    def test_get_agent_debug_args(self):
        replay_id = "aa35ccf6-594a-4b34-8255-ca0dfd821131"
        debug_args = self.get_debug_tool_args(replay_id=replay_id)
        self.assertIsNotNone(debug_args)

    def test_get_agent_debug_result(self):
        replay_id = "aa35ccf6-594a-4b34-8255-ca0dfd821131"
        debug_result = self.get_debug_tool_result(replay_id=replay_id)
        self.assertIsNotNone(debug_result)

    def test_get_tool_library(self):
        resp = self.get_tool_library()
        self.assertIsNotNone(resp)

    def test_get_test_run_info(self):
        service_version = "0.0.544"
        test_run_info = self.get_info_for_test_suite_run(
            service_version=service_version
        ).test_suite_run_info
        self.assertTrue(test_run_info)

    async def test_get_test_suite_runs(self):
        test_suite_runs = (await self.agent_service_impl.get_test_suite_runs()).test_suite_runs
        self.assertTrue(test_suite_runs)

    def test_get_test_case_info(self):
        test_name = "test_pe_nvda_feb_2024"
        test_case_info = self.get_info_for_test_case(test_name=test_name).test_case_info
        self.assertTrue(test_case_info)

    async def test_get_test_cases(self):
        test_cases = (await self.agent_service_impl.get_test_cases()).test_cases
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
        test_user_2 = str(uuid.uuid4())
        user = User(user_id=test_user, is_admin=False, is_super_admin=False, auth_token="")
        res = self.create_agent(user=user)
        agent_id = res.agent_id

        # insert a email notification
        emails = {
            "test.email@test.com": NotificationUser(
                user_id=test_user, username="testuser", name="TestUser", email="test.email@test.com"
            ),
            "test.email.2@test.com": NotificationUser(
                user_id=test_user_2,
                username="testuser2",
                name="TestUser2",
                email="test.email.2@test.com",
            ),
        }
        self.loop.run_until_complete(
            self.pg.set_agent_subscriptions(agent_id=agent_id, emails_to_user=emails)
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
        test_user_3 = str(uuid.uuid4())
        self.loop.run_until_complete(
            self.pg.set_agent_subscriptions(
                agent_id=agent_id,
                emails_to_user={
                    new_email: NotificationUser(
                        user_id=test_user_3,
                        username="testuser3",
                        name="TestUser3",
                        email="test.email.3@test.com",
                    )
                },
            )
        )
        notif_res: List[AgentNotificationEmail] = self.loop.run_until_complete(
            self.pg.get_agent_subscriptions(agent_id=agent_id)
        )
        self.assertEqual(len(notif_res), 1)
        res_emails = [agent_notif.email for agent_notif in notif_res]
        self.assertTrue(new_email in res_emails)
        # delete an email from the subscription
        email_to_delete = "test.email.3@test.com"
        self.loop.run_until_complete(
            self.pg.delete_agent_emails(agent_id=agent_id, email=email_to_delete)
        )
        notif_res: List[AgentNotificationEmail] = self.loop.run_until_complete(
            self.pg.get_agent_subscriptions(agent_id=agent_id)
        )
        self.assertEqual(len(notif_res), 0)
        res_emails = [agent_notif.email for agent_notif in notif_res]
        self.assertTrue(email_to_delete not in res_emails)

        self.delete_agent(agent_id=agent_id)

    def test_sidebar_organization(self):
        test_user_id = str(uuid.uuid4())
        test_user = User(user_id=test_user_id, is_admin=False, is_super_admin=False, auth_token="")
        agent_id_1 = self.create_agent(user=test_user).agent_id
        agent_id_2 = self.create_agent(user=test_user).agent_id
        agent_id_3 = self.create_agent(user=test_user).agent_id

        section_id_1 = self.loop.run_until_complete(
            self.agent_service_impl.create_sidebar_section(name="Section 1", user=test_user)
        )
        section_id_2 = self.loop.run_until_complete(
            self.agent_service_impl.create_sidebar_section(name="Section 2", user=test_user)
        )

        # Rename section
        new_name = "Section 4"
        self.loop.run_until_complete(
            self.agent_service_impl.rename_sidebar_section(
                new_name=new_name, section_id=section_id_1, user=test_user
            )
        )

        # Add agents to sections
        self.loop.run_until_complete(
            self.agent_service_impl.set_agent_sidebar_section(
                new_section_id=section_id_1, agent_id=agent_id_1, user=test_user
            )
        )

        self.loop.run_until_complete(
            self.agent_service_impl.set_agent_sidebar_section(
                new_section_id=section_id_1, agent_id=agent_id_2, user=test_user
            )
        )

        self.loop.run_until_complete(
            self.agent_service_impl.set_agent_sidebar_section(
                new_section_id=section_id_2, agent_id=agent_id_3, user=test_user
            )
        )

        # Reassign agent to another section
        self.loop.run_until_complete(
            self.agent_service_impl.set_agent_sidebar_section(
                new_section_id=section_id_2, agent_id=agent_id_1, user=test_user
            )
        )

        # Get all agents and sections
        response = self.loop.run_until_complete(
            self.agent_service_impl.get_all_agents(user=test_user)
        )

        self.assertEqual(len(response.sections), 2)
        self.assertEqual(response.sections[0].name, new_name)

        # Verifying that all agents are under expected sections
        self.assertEqual(
            find_item(response.agents, "agent_id", agent_id_1).section_id, section_id_2
        )
        self.assertEqual(
            find_item(response.agents, "agent_id", agent_id_2).section_id, section_id_1
        )
        self.assertEqual(
            find_item(response.agents, "agent_id", agent_id_3).section_id, section_id_2
        )

        # Deleting section
        self.loop.run_until_complete(
            self.agent_service_impl.delete_sidebar_section(section_id=section_id_2, user=test_user)
        )

        response = self.loop.run_until_complete(
            self.agent_service_impl.get_all_agents(user=test_user)
        )

        self.assertEqual(len(response.sections), 1)

        # Vberifying that agents in deleted section are in default section
        self.assertEqual(find_item(response.agents, "agent_id", agent_id_1).section_id, None)
        self.assertEqual(
            find_item(response.agents, "agent_id", agent_id_2).section_id, section_id_1
        )
        self.assertEqual(find_item(response.agents, "agent_id", agent_id_3).section_id, None)

        section_id_3 = self.loop.run_until_complete(
            self.agent_service_impl.create_sidebar_section(name="Section 3", user=test_user)
        )

        # Rearrange sections
        self.loop.run_until_complete(
            self.agent_service_impl.rearrange_sidebar_section(
                section_id=section_id_3, new_index=0, user=test_user
            )
        )

        response = self.loop.run_until_complete(
            self.agent_service_impl.get_all_agents(user=test_user)
        )
        if response.sections[0].index == 0:
            self.assertEqual(response.sections[0].id, section_id_3)
        else:
            self.assertEqual(response.sections[0].id, section_id_1)

        self.assertEqual(
            find_item(response.agents, "agent_id", agent_id_2).section_id, section_id_1
        )

    def test_agent_feedback(self):
        test_user_id = str(uuid.uuid4())
        test_user = User(user_id=test_user_id, is_admin=False, is_super_admin=False, auth_token="")
        agent = self.create_agent(user=test_user)
        plan_id = str(uuid.uuid4())
        plan_run_id = str(uuid.uuid4())
        output_id = str(uuid.uuid4())
        widget_title = str(uuid.uuid4())
        rating = -1
        feedback_comment = ""
        # Create a plan
        self.loop.run_until_complete(
            self.pg.write_execution_plan(
                plan_id=plan_id, agent_id=agent.agent_id, plan=ExecutionPlan(nodes=[])
            )
        )
        self.loop.run_until_complete(
            self.pg.update_plan_run(
                plan_run_id=plan_run_id, plan_id=plan_id, agent_id=agent.agent_id
            )
        )
        req = SetAgentFeedBackRequest(
            agent_id=agent.agent_id,
            plan_id=plan_id,
            plan_run_id=plan_run_id,
            output_id=output_id,
            widget_title=widget_title,
            rating=rating,
            feedback_comment=feedback_comment,
        )
        self.loop.run_until_complete(
            self.pg.set_agent_feedback(feedback_data=req, user_id=test_user_id)
        )

        # get the feedback
        feedback: List[AgentFeedback] = self.loop.run_until_complete(
            self.pg.get_agent_feedback(
                agent_id=agent.agent_id,
                plan_id=plan_id,
                plan_run_id=plan_run_id,
                output_id=output_id,
                feedback_user_id=test_user_id,
            )
        )

        self.assertEqual(feedback[0].feedback_comment, feedback_comment)
        self.assertEqual(feedback[0].agent_id, agent.agent_id)
        self.assertEqual(feedback[0].plan_id, plan_id)
        self.assertEqual(feedback[0].widget_title, widget_title)

        feedback_comment = "new feedback"
        req = SetAgentFeedBackRequest(
            agent_id=agent.agent_id,
            plan_id=plan_id,
            plan_run_id=plan_run_id,
            output_id=output_id,
            widget_title=widget_title,
            rating=rating,
            feedback_comment=feedback_comment,
        )
        self.loop.run_until_complete(
            self.pg.set_agent_feedback(feedback_data=req, user_id=test_user_id)
        )

        # get the feedback
        feedback: List[AgentFeedback] = self.loop.run_until_complete(
            self.pg.get_agent_feedback(
                agent_id=agent.agent_id,
                plan_id=plan_id,
                plan_run_id=plan_run_id,
                output_id=output_id,
                feedback_user_id=test_user_id,
            )
        )
        self.assertEqual(feedback[0].feedback_comment, feedback_comment)
        self.assertEqual(feedback[0].agent_id, agent.agent_id)
        self.assertEqual(feedback[0].plan_id, plan_id)
        self.assertEqual(feedback[0].widget_title, widget_title)

    def test_duplicate_agent(self):
        agent_to_duplicate = "ebc63574-4cb0-409c-9965-2a0b5a9a5037"
        user_id_to_send_to = "ac7c96d7-3e57-40e7-a1a5-8e2ce5e23639"  # tommy
        res = self.duplicate_agent(
            src_agent_id=agent_to_duplicate, dest_user_ids=[user_id_to_send_to]
        )
        print(res)
