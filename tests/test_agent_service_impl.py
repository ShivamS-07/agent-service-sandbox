import unittest
import uuid
from datetime import datetime
from typing import List
from unittest.mock import MagicMock, patch

from fastapi import HTTPException

from agent_service.endpoints.authz_helper import User
from agent_service.endpoints.models import (
    AgentFeedback,
    AgentHelpRequest,
    AgentInfo,
    AgentNotificationEmail,
    AgentQC,
    ChatWithAgentRequest,
    DeleteAgentOutputRequest,
    HorizonCriteria,
    HorizonCriteriaOperator,
    LockAgentOutputRequest,
    MediaType,
    Pagination,
    SetAgentFeedBackRequest,
    Status,
    TransformTableOutputRequest,
    UnlockAgentOutputRequest,
    UpdateAgentRequest,
    UpdateAgentWidgetNameRequest,
    UpdateTransformationSettingsRequest,
)
from agent_service.io_types.table import TableTransformation
from agent_service.planner.constants import FirstAction
from agent_service.planner.planner_types import ExecutionPlan
from agent_service.types import Notification
from agent_service.utils.async_db import AsyncDB
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
    return "Hi this is Warren AI, how can I help you today ?"


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

    @unittest.skip("slow = 86 seconds")
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
            "test.email@test.com": test_user,
            "test.email.2@test.com": test_user_2,
        }
        self.loop.run_until_complete(
            self.pg.set_agent_subscriptions(agent_id=agent_id, email_to_user_id=emails)
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
                email_to_user_id={
                    new_email: test_user_3,
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
                new_section_id=section_id_1, agent_ids=[agent_id_1], user=test_user
            )
        )

        self.loop.run_until_complete(
            self.agent_service_impl.set_agent_sidebar_section(
                new_section_id=section_id_1, agent_ids=[agent_id_2], user=test_user
            )
        )

        self.loop.run_until_complete(
            self.agent_service_impl.set_agent_sidebar_section(
                new_section_id=section_id_2, agent_ids=[agent_id_3], user=test_user
            )
        )

        # Reassign agent to another section
        self.loop.run_until_complete(
            self.agent_service_impl.set_agent_sidebar_section(
                new_section_id=section_id_2, agent_ids=[agent_id_1], user=test_user
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
        self.assertIsNotNone(res)

    def test_duplicate_agent_null_task_id(self):
        agent_to_duplicate = "512f3ce0-8796-4bdd-a1fa-c2115f4ff579"
        user_id_to_send_to = "ac7c96d7-3e57-40e7-a1a5-8e2ce5e23639"  # tommy
        res = self.duplicate_agent(
            src_agent_id=agent_to_duplicate, dest_user_ids=[user_id_to_send_to]
        )
        self.assertIsNotNone(res)

    @unittest.skip("Causes hanging")
    def test_delete_agent_output(self):
        TEST_AGENT_ID = "fd7b8b1a-3e3a-4195-8a6c-99f500149fde"
        TEST_PLAN_ID = "5633b272-1a29-45c0-b2d0-de12160bd23c"
        EXPECTED_TASK_IDs = [
            "b7cd537c-eb71-4be9-8369-727e592f284c",
            "3be8d66e-70b8-4034-b036-073cad924dfc",
            "ecd58097-16bc-4fdf-adc8-c82491a9e382",
            "e2e1bfd1-9f7f-420d-97e8-278b379ee704",
            "4db70330-e715-4585-9c0e-0058dba7bbbc",
        ]
        res = self.delete_agent_output(
            agent_id=TEST_AGENT_ID,
            req=DeleteAgentOutputRequest(
                plan_id=TEST_PLAN_ID,
                output_ids=["e0fafdcd-19ae-4740-bdf7-8bd09e23ac7a"],
                task_ids=["1982aa37-48d1-4451-be70-5d42dc66fab6"],
            ),
        )
        self.assertTrue(res.success)
        plan_id, plan, _, _, _ = self.loop.run_until_complete(
            self.pg.get_latest_execution_plan(agent_id=TEST_AGENT_ID)
        )
        self.assertEqual(plan_id, TEST_PLAN_ID)
        assert plan is not None
        self.assertEqual([node.tool_task_id for node in plan.nodes], EXPECTED_TASK_IDs)
        outputs = self.loop.run_until_complete(self.pg.get_agent_outputs(agent_id=TEST_AGENT_ID))
        self.assertEqual(len(outputs), 1)

        # Try deleting again, make sure both deletions are applied
        res = self.delete_agent_output(
            agent_id=TEST_AGENT_ID,
            req=DeleteAgentOutputRequest(
                plan_id=TEST_PLAN_ID,
                output_ids=["c0aaed83-c0cf-47f5-910d-ec7666e2725d"],
                task_ids=["4db70330-e715-4585-9c0e-0058dba7bbbc"],
            ),
        )
        self.assertTrue(res.success)
        plan_id, plan, _, _, _ = self.loop.run_until_complete(
            self.pg.get_latest_execution_plan(agent_id=TEST_AGENT_ID)
        )
        self.assertEqual(plan_id, TEST_PLAN_ID)
        assert plan is not None
        self.assertEqual([node.tool_task_id for node in plan.nodes], [])
        outputs = self.loop.run_until_complete(self.pg.get_agent_outputs(agent_id=TEST_AGENT_ID))
        self.assertEqual(len(outputs), 0)

    def test_lock_unlock_output(self):
        TEST_AGENT_ID = "fd7b8b1a-3e3a-4195-8a6c-99f500149fde"
        TEST_PLAN_ID = "5633b272-1a29-45c0-b2d0-de12160bd23c"

        res = self.lock_agent_output(
            agent_id=TEST_AGENT_ID,
            req=LockAgentOutputRequest(
                plan_id=TEST_PLAN_ID,
                output_ids=["e0fafdcd-19ae-4740-bdf7-8bd09e23ac7a"],
                task_ids=["1982aa37-48d1-4451-be70-5d42dc66fab6"],
            ),
        )
        self.assertTrue(res.success)
        _, plan, _, _, _ = self.loop.run_until_complete(
            self.pg.get_latest_execution_plan(agent_id=TEST_AGENT_ID)
        )
        assert plan is not None
        self.assertEqual(plan.locked_task_ids, ["1982aa37-48d1-4451-be70-5d42dc66fab6"])
        res = self.lock_agent_output(
            agent_id=TEST_AGENT_ID,
            req=LockAgentOutputRequest(
                plan_id=TEST_PLAN_ID,
                output_ids=["c0aaed83-c0cf-47f5-910d-ec7666e2725d"],
                task_ids=["4db70330-e715-4585-9c0e-0058dba7bbbc"],
            ),
        )
        self.assertTrue(res.success)
        _, plan, _, _, _ = self.loop.run_until_complete(
            self.pg.get_latest_execution_plan(agent_id=TEST_AGENT_ID)
        )
        assert plan is not None
        self.assertEqual(
            sorted(plan.locked_task_ids),
            ["1982aa37-48d1-4451-be70-5d42dc66fab6", "4db70330-e715-4585-9c0e-0058dba7bbbc"],
        )

        res = self.unlock_agent_output(
            agent_id=TEST_AGENT_ID,
            req=UnlockAgentOutputRequest(
                plan_id=TEST_PLAN_ID,
                output_ids=["e0fafdcd-19ae-4740-bdf7-8bd09e23ac7a"],
                task_ids=["1982aa37-48d1-4451-be70-5d42dc66fab6"],
            ),
        )
        self.assertTrue(res.success)
        _, plan, _, _, _ = self.loop.run_until_complete(
            self.pg.get_latest_execution_plan(agent_id=TEST_AGENT_ID)
        )
        assert plan is not None
        self.assertEqual(
            plan.locked_task_ids,
            ["4db70330-e715-4585-9c0e-0058dba7bbbc"],
        )

    @unittest.skip("Causes hanging")
    def test_delete_with_lock_unlock_agent_output(self):
        TEST_AGENT_ID = "fd7b8b1a-3e3a-4195-8a6c-99f500149fde"
        TEST_PLAN_ID = "5633b272-1a29-45c0-b2d0-de12160bd23c"

        # First, lock an output
        res = self.lock_agent_output(
            agent_id=TEST_AGENT_ID,
            req=LockAgentOutputRequest(
                plan_id=TEST_PLAN_ID,
                output_ids=["c0aaed83-c0cf-47f5-910d-ec7666e2725d"],
                task_ids=["4db70330-e715-4585-9c0e-0058dba7bbbc"],
            ),
        )
        self.assertTrue(res.success)
        _, plan, _, _, _ = self.loop.run_until_complete(
            self.pg.get_latest_execution_plan(agent_id=TEST_AGENT_ID)
        )
        assert plan is not None
        self.assertEqual(
            plan.locked_task_ids,
            ["4db70330-e715-4585-9c0e-0058dba7bbbc"],
        )

        # Then, delete a different output
        res = self.delete_agent_output(
            agent_id=TEST_AGENT_ID,
            req=DeleteAgentOutputRequest(
                plan_id=TEST_PLAN_ID,
                output_ids=["e0fafdcd-19ae-4740-bdf7-8bd09e23ac7a"],
                task_ids=["1982aa37-48d1-4451-be70-5d42dc66fab6"],
            ),
        )
        self.assertTrue(res.success)
        _, plan, _, _, _ = self.loop.run_until_complete(
            self.pg.get_latest_execution_plan(agent_id=TEST_AGENT_ID)
        )
        assert plan is not None
        # Make sure locked output is not changed
        self.assertEqual(
            plan.locked_task_ids,
            ["4db70330-e715-4585-9c0e-0058dba7bbbc"],
        )

        # Then unlock and make sure it works

        res = self.unlock_agent_output(
            agent_id=TEST_AGENT_ID,
            req=UnlockAgentOutputRequest(
                plan_id=TEST_PLAN_ID,
                output_ids=["c0aaed83-c0cf-47f5-910d-ec7666e2725d"],
                task_ids=["4db70330-e715-4585-9c0e-0058dba7bbbc"],
            ),
        )
        self.assertTrue(res.success)
        _, plan, _, _, _ = self.loop.run_until_complete(
            self.pg.get_latest_execution_plan(agent_id=TEST_AGENT_ID)
        )
        assert plan is not None
        self.assertEqual(plan.locked_task_ids, [])

    def test_agent_quality_endpoints(self):
        # Step 1: Create and Insert mock AgentQC data
        reviewer1 = str(uuid.uuid4())
        reviewer2 = str(uuid.uuid4())
        reviewer3 = str(uuid.uuid4())
        agent_id = str(uuid.uuid4())
        user_id = str(uuid.uuid4())
        agent = AgentInfo(
            agent_id=agent_id,
            user_id=user_id,
            agent_name="agent name",
            created_at=datetime.now(),
            last_updated=datetime.now(),
            deleted=False,
        )
        agent_qc = AgentQC(
            agent_qc_id=str(uuid.uuid4()),
            agent_id=agent_id,
            plan_id=str(uuid.uuid4()),
            user_id=user_id,
            query="initial query",
            agent_status=Status.COMPLETE,
            cs_reviewer=reviewer1,
            eng_reviewer=reviewer2,
            prod_reviewer=reviewer3,
            use_case="COMMENTARY",
            problem_area="test problem",
            created_at=datetime.now(),
            last_updated=datetime.now(),
            is_spoofed=False,
        )

        async def insert_agent(pg: AsyncDB, agent_metadata: AgentInfo) -> None:
            await pg.multi_row_insert(
                table_name="agent.agents", rows=[agent_metadata.to_agent_row()]
            )

        # Insert Agent record
        self.loop.run_until_complete(insert_agent(self.pg, agent))

        # Insert the AgentQC record
        self.loop.run_until_complete(self.pg.insert_agent_qc(agent_qc))

        # Retrieve and verify the inserted record
        inserted_qc = self.loop.run_until_complete(
            self.pg.get_agent_qc_by_ids([agent_qc.agent_qc_id])
        )
        self.assertEqual(inserted_qc[0].agent_qc_id, agent_qc.agent_qc_id)
        self.assertEqual(inserted_qc[0].use_case, agent_qc.use_case)

        # Step 2: Update the inserted AgentQC record
        agent_qc.follow_up = "follow up 2"
        agent_qc.score_rating = 5
        agent_qc.priority = "high"
        agent_qc.use_case = "updated use case"

        # Update the record
        self.loop.run_until_complete(self.pg.update_agent_qc(agent_qc=agent_qc))

        # Retrieve and verify the updated record
        updated_qc = self.loop.run_until_complete(
            self.pg.get_agent_qc_by_ids([agent_qc.agent_qc_id])
        )
        self.assertEqual(updated_qc[0].follow_up, "follow up 2")
        self.assertEqual(updated_qc[0].score_rating, 5)
        self.assertEqual(updated_qc[0].priority, "high")
        self.assertEqual(updated_qc[0].use_case, "updated use case")

        # Step 3: Insert additional records for the same user
        user_id = agent_qc.user_id
        for _ in range(2):
            agent_id = str(uuid.uuid4())
            agent = AgentInfo(
                agent_id=agent_id,
                user_id=user_id,
                agent_name="agent name",
                created_at=datetime.now(),
                last_updated=datetime.now(),
                deleted=False,
            )
            self.loop.run_until_complete(insert_agent(self.pg, agent))
            new_agent_qc = AgentQC(
                agent_qc_id=str(uuid.uuid4()),
                agent_id=agent_id,
                plan_id=str(uuid.uuid4()),
                user_id=user_id,
                query="user-specific query",
                agent_status="CS",
                score_rating=2,
                created_at=datetime.now(),
                last_updated=datetime.now(),
                is_spoofed=False,
            )
            self.loop.run_until_complete(self.pg.insert_agent_qc(new_agent_qc))

        # Retrieve all records by user_id and verify count
        retrieved_qcs_by_user = self.loop.run_until_complete(
            self.pg.get_agent_qc_by_user_ids([user_id])
        )
        self.assertEqual(len(retrieved_qcs_by_user), 3)
        self.assertEqual(retrieved_qcs_by_user[0].user_id, user_id)

        criteria = [
            HorizonCriteria(
                column="use_case",
                operator=HorizonCriteriaOperator.ilike.value,
                arg1="updated use case",
                arg2=None,
            )
        ]
        search_results = self.loop.run_until_complete(
            self.pg.search_agent_qc(criteria, [], pagination=Pagination(page_index=0, page_size=10))
        )

        # Verify the search results
        self.assertEqual(len(search_results), 2)
        self.assertEqual(len(search_results[0]), 1)

        criteria_1 = [
            HorizonCriteria(
                column="query",
                operator=HorizonCriteriaOperator.ilike.value,
                arg1="initial query",
                arg2=None,
            ),
        ]

        criteria_2 = [
            HorizonCriteria(
                column="use_case",
                operator=HorizonCriteriaOperator.ilike.value,
                arg1="updated use case",
                arg2=None,
            ),
        ]
        search_results = self.loop.run_until_complete(
            self.pg.search_agent_qc(
                criteria_1, criteria_2, pagination=Pagination(page_index=0, page_size=10)
            )
        )
        # Verify the search results
        self.assertEqual(len(search_results), 2)
        self.assertEqual(len(search_results[0]), 1)

        criteria_2 = [
            HorizonCriteria(
                column="query",
                operator=HorizonCriteriaOperator.ilike.value,
                arg1="user-specific query",
                arg2=None,
            ),
            HorizonCriteria(
                column="use_case",
                operator=HorizonCriteriaOperator.ilike.value,
                arg1="updated use case",
                arg2=None,
            ),
        ]

        search_results = self.loop.run_until_complete(
            self.pg.search_agent_qc(
                [], criteria_2, pagination=Pagination(page_index=0, page_size=10)
            )
        )
        # Verify the search results
        self.assertEqual(len(search_results), 2)
        self.assertEqual(len(search_results[0]), 3)

    def test_update_agent_widget_title(self):
        TEST_AGENT_ID = "565de7d2-d050-4813-8610-3b474b06823e"
        TEST_OUTPUT_ID = "4c25a13e-5977-4550-af33-62e1abb7277b"
        NEW_WIDGET_TITLE = "New Widget Title"

        # Change to new widget title
        req = UpdateAgentWidgetNameRequest(
            output_id=TEST_OUTPUT_ID, new_widget_title=NEW_WIDGET_TITLE
        )
        res = self.update_agent_widget_name(agent_id=TEST_AGENT_ID, req=req)
        self.assertTrue(res.success)

        # Retrieve the updated widget title
        agent_output = self.loop.run_until_complete(
            self.pg.get_agent_outputs_data_from_db(
                agent_id=TEST_AGENT_ID, include_output=True, output_id=TEST_OUTPUT_ID
            )
        )

        # Assert that output contains the new title
        self.assertEqual(NEW_WIDGET_TITLE in agent_output[0]["output"], True)

        # Assert that what's new contains the new title
        self.assertEqual(
            NEW_WIDGET_TITLE in agent_output[0]["run_metadata"]["run_summary_long"]["val"], True
        )

        # Assert that plan node contain the new title
        new_title_in_nodes = False
        for node in agent_output[0]["plan"]["nodes"]:
            if "title" in node["args"] and NEW_WIDGET_TITLE == node["args"]["title"]:
                new_title_in_nodes = True
        self.assertEqual(new_title_in_nodes, True)

    def test_help_requests(self):
        test_user = str(uuid.uuid4())
        user = User(user_id=test_user, is_admin=False, is_super_admin=False, auth_token="")
        TEST_AGENT_ID = "565de7d2-d050-4813-8610-3b474b06823e"
        self.set_agent_help_requested(
            agent_id=TEST_AGENT_ID,
            req=AgentHelpRequest(is_help_requested=True),
            requesting_user=user,
        )
        self.assertTrue(
            self.loop.run_until_complete(self.pg.is_agent_help_requested(agent_id=TEST_AGENT_ID))
        )
        self.set_agent_help_requested(
            agent_id=TEST_AGENT_ID,
            req=AgentHelpRequest(is_help_requested=False),
            requesting_user=user,
        )
        self.assertFalse(
            self.loop.run_until_complete(self.pg.is_agent_help_requested(agent_id=TEST_AGENT_ID))
        )

    def test_transform_table_output(self):
        # initialize entries
        user_id = str(uuid.uuid4())
        agent_id = self.loop.run_until_complete(self._create_fake_agent(user_id))
        plan_id = self.loop.run_until_complete(self._create_fake_plan_for_agent(agent_id))
        plan_run_ids = self.loop.run_until_complete(
            self._create_fake_plan_runs(agent_id, plan_id, num=2)
        )

        task_id = str(uuid.uuid4())

        # Bad request for when `is_transformation_local` is False but `effective_from` is null
        with self.assertRaises(HTTPException):
            self.loop.run_until_complete(
                self.agent_service_impl.transform_table_output(
                    req=TransformTableOutputRequest(
                        agent_id=agent_id,
                        plan_id=plan_id,
                        plan_run_id=plan_run_ids[0],
                        task_id=task_id,
                        is_transformation_local=False,
                        transformation=TableTransformation(
                            columns=[
                                TableTransformation.Column(
                                    original_name="column1",
                                    display_name="display_column1",
                                    is_visible=True,
                                )
                            ]
                        ),
                    )
                )
            )

        # Create 3 global transformations for 2 runs
        global_transformation_id1 = self.loop.run_until_complete(
            self.agent_service_impl.transform_table_output(
                req=TransformTableOutputRequest(
                    agent_id=agent_id,
                    plan_id=plan_id,
                    plan_run_id=plan_run_ids[0],
                    task_id=task_id,
                    is_transformation_local=False,
                    transformation=TableTransformation(
                        columns=[
                            TableTransformation.Column(
                                original_name="column",
                                display_name="global_transformation_id1",
                                is_visible=True,
                            )
                        ]
                    ),
                    effective_from=datetime(1900, 1, 1),
                )
            )
        )
        global_transformation_id2 = self.loop.run_until_complete(
            self.agent_service_impl.transform_table_output(
                req=TransformTableOutputRequest(
                    agent_id=agent_id,
                    plan_id=plan_id,
                    plan_run_id=plan_run_ids[1],
                    task_id=task_id,
                    is_transformation_local=False,
                    transformation=TableTransformation(
                        columns=[
                            TableTransformation.Column(
                                original_name="column",
                                display_name="global_transformation_id2",
                                is_visible=True,
                            )
                        ]
                    ),
                    effective_from=datetime(1900, 1, 1),
                )
            )
        )
        _ = self.loop.run_until_complete(
            self.agent_service_impl.transform_table_output(
                req=TransformTableOutputRequest(
                    agent_id=agent_id,
                    plan_id=plan_id,
                    plan_run_id=plan_run_ids[1],
                    task_id=task_id,
                    is_transformation_local=False,
                    transformation=TableTransformation(
                        columns=[
                            TableTransformation.Column(
                                original_name="column",
                                display_name="global_transformation_id2",
                                is_visible=True,
                            )
                        ]
                    ),
                    effective_from=datetime(4000, 1, 1),
                )
            )
        )
        # should return the 2nd one since it's the latest one which also matches `effective_from`
        output_transformations = self.loop.run_until_complete(
            self.pg.get_output_transformations(agent_id=agent_id, plan_run_id=plan_run_ids[0])
        )
        transformation = output_transformations[task_id]
        self.assertEqual(
            transformation["transformation_id"], global_transformation_id2.transformation_id
        )

        # Create 2 more local transformations for 2 runs
        local_transformation_id1 = self.loop.run_until_complete(
            self.agent_service_impl.transform_table_output(
                req=TransformTableOutputRequest(
                    agent_id=agent_id,
                    plan_id=plan_id,
                    plan_run_id=plan_run_ids[0],
                    task_id=task_id,
                    is_transformation_local=True,
                    transformation=TableTransformation(
                        columns=[
                            TableTransformation.Column(
                                original_name="column",
                                display_name="local_transformation_id1",
                                is_visible=True,
                            )
                        ]
                    ),
                )
            )
        )
        local_transformation_id2 = self.loop.run_until_complete(
            self.agent_service_impl.transform_table_output(
                req=TransformTableOutputRequest(
                    agent_id=agent_id,
                    plan_id=plan_id,
                    plan_run_id=plan_run_ids[1],
                    task_id=task_id,
                    is_transformation_local=True,
                    transformation=TableTransformation(
                        columns=[
                            TableTransformation.Column(
                                original_name="column",
                                display_name="local_transformation_id2",
                                is_visible=True,
                            )
                        ]
                    ),
                )
            )
        )

        output_transformations = self.loop.run_until_complete(
            self.pg.get_output_transformations(agent_id=agent_id, plan_run_id=plan_run_ids[0])
        )
        transformation = output_transformations[task_id]
        self.assertEqual(
            transformation["transformation_id"], local_transformation_id1.transformation_id
        )

        output_transformations = self.loop.run_until_complete(
            self.pg.get_output_transformations(agent_id=agent_id, plan_run_id=plan_run_ids[1])
        )
        transformation = output_transformations[task_id]
        self.assertEqual(
            transformation["transformation_id"], local_transformation_id2.transformation_id
        )

        # delete the local transformation - should get global then
        self.loop.run_until_complete(
            self.agent_service_impl.delete_transformation(
                transformation_id=local_transformation_id1.transformation_id
            )
        )
        output_transformations = self.loop.run_until_complete(
            self.pg.get_output_transformations(agent_id=agent_id, plan_run_id=plan_run_ids[0])
        )
        transformation = output_transformations[task_id]
        self.assertEqual(
            transformation["transformation_id"], global_transformation_id2.transformation_id
        )

        # update global transformation1 - should get global1 then (latest)
        self.loop.run_until_complete(
            self.agent_service_impl.update_transformation_settings(
                req=UpdateTransformationSettingsRequest(
                    agent_id=agent_id,
                    transformation_id=global_transformation_id1.transformation_id,
                    is_transformation_local=False,
                    effective_from=datetime(1901, 1, 1),
                )
            )
        )
        output_transformations = self.loop.run_until_complete(
            self.pg.get_output_transformations(agent_id=agent_id, plan_run_id=plan_run_ids[0])
        )
        transformation = output_transformations[task_id]
        self.assertEqual(
            transformation["transformation_id"], global_transformation_id1.transformation_id
        )
