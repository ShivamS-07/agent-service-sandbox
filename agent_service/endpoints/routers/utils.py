import logging
import os
from typing import Optional

from gbi_common_py_utils.utils.environment import get_environment_tag

from agent_service.agent_quality_worker.jira_integration import JiraIntegration
from agent_service.agent_service_impl import AgentServiceImpl
from agent_service.GPT.requests import _get_gpt_service_stub
from agent_service.slack.slack_sender import SlackSender
from agent_service.utils.async_db import get_async_db
from agent_service.utils.cache_utils import get_redis_cache_backend_for_output
from agent_service.utils.clickhouse import Clickhouse
from agent_service.utils.default_task_executor import DefaultTaskExecutor
from agent_service.utils.feature_flags import agent_output_cache_enabled

logger = logging.getLogger(__name__)


AGENT_SERVICE_IMPL: Optional[AgentServiceImpl] = None


def get_agent_svc_impl() -> AgentServiceImpl:
    global AGENT_SERVICE_IMPL
    if AGENT_SERVICE_IMPL:
        return AGENT_SERVICE_IMPL

    logger.warning("### Creating new AgentServiceImpl instance ###")

    env = get_environment_tag()
    channel = "alfa-client-queries" if env == "ALPHA" else "alfa-client-queries-dev"

    base_url = "alfa.boosted.ai" if env == "ALPHA" else "agent-dev.boosted.ai"

    cache = None
    if agent_output_cache_enabled() and os.getenv("REDIS_HOST"):
        logger.info(f"Using redis output cache. Connecting to {os.getenv('REDIS_HOST')}")
        cache = get_redis_cache_backend_for_output()

    async_db = get_async_db(min_pool_size=1, max_pool_size=4)

    AGENT_SERVICE_IMPL = AgentServiceImpl(
        task_executor=DefaultTaskExecutor(),
        gpt_service_stub=_get_gpt_service_stub()[0],
        async_db=async_db,
        clickhouse_db=Clickhouse(),
        slack_sender=SlackSender(channel=channel),
        base_url=base_url,
        cache=cache,
        jira_integration=JiraIntegration(),
        env=env,
    )
    return AGENT_SERVICE_IMPL
