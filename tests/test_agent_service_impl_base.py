# flake8: noqa
import datetime
import logging
from typing import Optional

from fastapi import UploadFile

from agent_service.slack.slack_sender import SlackSender
from agent_service.utils.async_db import AsyncDB
from agent_service.utils.clickhouse import Clickhouse
from agent_service.utils.logs import init_stdout_logging
from tests.skip_commit_boosted_db import SkipCommitBoostedPG

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s.%(funcName)s:%(lineno)d - %(levelname)s - %(message)s",
)
import asyncio
import unittest

from gpt_service_proto_v1.service_grpc import GPTServiceStub
from grpclib.client import Channel

from agent_service.agent_service_impl import AgentServiceImpl
from agent_service.endpoints.authz_helper import User
from agent_service.endpoints.models import (
    AgentMetadata,
    ChatWithAgentRequest,
    ChatWithAgentResponse,
    CreateAgentResponse,
    DeleteAgentResponse,
    GetAgentDebugInfoResponse,
    GetAgentOutputResponse,
    GetAgentTaskOutputResponse,
    GetAllAgentsResponse,
    GetCannedPromptsResponse,
    GetChatHistoryResponse,
    GetDebugToolArgsResponse,
    GetDebugToolResultResponse,
    GetTestCaseInfoResponse,
    GetTestSuiteRunInfoResponse,
    GetToolLibraryResponse,
    RestoreAgentResponse,
    TerminateAgentResponse,
    UpdateAgentRequest,
    UpdateAgentResponse,
)
from agent_service.utils.do_nothing_task_executor import DoNothingTaskExecutor


class TestAgentServiceImplBase(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        init_stdout_logging()

        cls.loop = asyncio.get_event_loop()  # type: ignore[assignment]
        cls.pg = AsyncDB(pg=SkipCommitBoostedPG())
        cls.channel = Channel(host="gpt-service-2.boosted.ai", port=50051)
        cls.gpt_service_stub = GPTServiceStub(cls.channel)
        cls.agent_service_impl = AgentServiceImpl(
            task_executor=DoNothingTaskExecutor(),
            gpt_service_stub=cls.gpt_service_stub,
            async_db=cls.pg,
            clickhouse_db=Clickhouse(),
            slack_sender=SlackSender(channel="tommy-test"),
            base_url="https://agent-dev.boosted.ai",
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.channel.close()

    def create_agent(self, user: User) -> CreateAgentResponse:
        return self.loop.run_until_complete(self.agent_service_impl.create_agent(user=user))

    def get_all_agents(self, user: User) -> GetAllAgentsResponse:
        return self.loop.run_until_complete(self.agent_service_impl.get_all_agents(user=user))

    def get_agent(self, agent_id: str) -> AgentMetadata:
        return self.loop.run_until_complete(self.agent_service_impl.get_agent(agent_id=agent_id))

    def terminate_agent(
        self, agent_id: str, plan_id: Optional[str] = None, plan_run_id: Optional[str] = None
    ) -> TerminateAgentResponse:
        return self.loop.run_until_complete(
            self.agent_service_impl.terminate_agent(
                agent_id=agent_id, plan_id=plan_id, plan_run_id=plan_run_id
            )
        )

    def delete_agent(self, agent_id: str) -> DeleteAgentResponse:
        return self.loop.run_until_complete(self.agent_service_impl.delete_agent(agent_id=agent_id))

    def restore_agent(self, agent_id: str) -> RestoreAgentResponse:
        return self.loop.run_until_complete(
            self.agent_service_impl.restore_agent(agent_id=agent_id)
        )

    def update_agent(self, agent_id: str, req: UpdateAgentRequest) -> UpdateAgentResponse:
        return self.loop.run_until_complete(
            self.agent_service_impl.update_agent(agent_id=agent_id, req=req)
        )

    def chat_with_agent(self, req: ChatWithAgentRequest, user: User) -> ChatWithAgentResponse:
        return self.loop.run_until_complete(
            self.agent_service_impl.chat_with_agent(req=req, user=user)
        )

    def get_chat_history(
        self,
        agent_id: str,
        start: Optional[datetime.datetime] = None,
        end: Optional[datetime.datetime] = None,
    ) -> GetChatHistoryResponse:
        return self.loop.run_until_complete(
            self.agent_service_impl.get_chat_history(agent_id=agent_id, start=start, end=end)
        )

    def get_agent_task_output(
        self, agent_id: str, plan_run_id: str, task_id: str
    ) -> GetAgentTaskOutputResponse:
        return self.loop.run_until_complete(
            self.agent_service_impl.get_agent_task_output(
                agent_id=agent_id, plan_run_id=plan_run_id, task_id=task_id
            )
        )

    def get_agent_plan_output(self, agent_id: str) -> GetAgentOutputResponse:
        return self.loop.run_until_complete(
            self.agent_service_impl.get_agent_plan_output(agent_id=agent_id)
        )

    def get_agent_debug_info(self, agent_id: str) -> GetAgentDebugInfoResponse:
        return self.loop.run_until_complete(
            self.agent_service_impl.get_agent_debug_info(agent_id=agent_id)
        )

    def get_debug_tool_args(self, replay_id: str) -> GetDebugToolArgsResponse:
        return self.loop.run_until_complete(
            self.agent_service_impl.get_debug_tool_args(replay_id=replay_id)
        )

    def get_debug_tool_result(self, replay_id: str) -> GetDebugToolResultResponse:
        return self.loop.run_until_complete(
            self.agent_service_impl.get_debug_tool_result(replay_id=replay_id)
        )

    def get_tool_library(self) -> GetToolLibraryResponse:
        return self.loop.run_until_complete(self.agent_service_impl.get_tool_library())

    def get_info_for_test_suite_run(self, service_version: str) -> GetTestSuiteRunInfoResponse:
        return self.loop.run_until_complete(
            self.agent_service_impl.get_info_for_test_suite_run(service_version=service_version)
        )

    def get_info_for_test_case(self, test_name: str) -> GetTestCaseInfoResponse:
        return self.loop.run_until_complete(
            self.agent_service_impl.get_info_for_test_case(test_name=test_name)
        )

    def get_canned_prompts(self) -> GetCannedPromptsResponse:
        return self.agent_service_impl.get_canned_prompts()
