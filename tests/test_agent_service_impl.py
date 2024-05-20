import uuid

from agent_service.endpoints.authz_helper import User
from agent_service.endpoints.models import (
    ChatWithAgentRequest,
    CreateAgentRequest,
    UpdateAgentRequest,
)
from tests.test_agent_service_impl_base import TestAgentServiceImplBase


class TestAgentServiceImpl(TestAgentServiceImplBase):
    def test_agent_crud(self):
        test_user = str(uuid.uuid4())
        user = User(user_id=test_user, is_admin=False, is_super_admin=False, auth_token="")
        res = self.create_agent(req=CreateAgentRequest(first_prompt="Please help me!"), user=user)
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

    def test_chat_with_agent(self):
        test_user = str(uuid.uuid4())
        user = User(user_id=test_user, is_admin=False, is_super_admin=False, auth_token="")
        res = self.create_agent(req=CreateAgentRequest(first_prompt="Please help me!"), user=user)
        agent_id = res.agent_id

        res = self.chat_with_agent(
            req=ChatWithAgentRequest(agent_id=agent_id, prompt="Let me try again"), user=user
        )
        self.assertIsNotNone(res)

        res = self.get_chat_history(agent_id=agent_id)
        self.assertIsNotNone(res)

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
