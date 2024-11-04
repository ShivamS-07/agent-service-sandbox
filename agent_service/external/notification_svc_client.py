import logging
import os
from contextlib import contextmanager
from functools import lru_cache
from typing import Generator, List, Tuple

from gbi_common_py_utils.utils.environment import (
    DEV_TAG,
    LOCAL_TAG,
    PROD_TAG,
    get_environment_tag,
)
from grpclib.client import Channel
from notification_service_proto_v1.notif_service_grpc import NotificationServiceStub
from notification_service_proto_v1.notification_messages_pb2 import (
    GetOnboardingEmailSequenceLogRequest,
    OnboardingEmailSequenceLog,
)

from agent_service.external.grpc_utils import get_default_grpc_metadata, grpc_retry
from agent_service.utils.logs import async_perf_logger

logger = logging.getLogger(__name__)

DEFAULT_URLS = {
    LOCAL_TAG: ("notification-service-dev.boosted.ai", 50051),
    DEV_TAG: ("notification-service-dev.boosted.ai", 50051),
    PROD_TAG: ("notification-service-prod.boosted.ai", 50051),
}


@lru_cache(maxsize=1)
def get_url_and_port() -> Tuple[str, int]:
    url = os.environ.get("NOTIFICATION_SERVICE_URL")
    if url is not None:
        logger.warning(f"Found NOTIFICATION_SERVICE_URL override: {url}")
        split_url, port = url.split(":")
        return split_url, int(port)

    env = get_environment_tag()
    if env not in DEFAULT_URLS:
        raise ValueError(f"No URL is set for environment {env}")
    return DEFAULT_URLS[env]


@contextmanager
def _get_service_stub() -> Generator[NotificationServiceStub, None, None]:
    try:
        url, port = get_url_and_port()
        channel = Channel(url, port)
        yield NotificationServiceStub(channel)
    finally:
        channel.close()


@grpc_retry
@async_perf_logger
async def get_onboarding_email_sequence_log(user_id: str) -> List[OnboardingEmailSequenceLog]:
    with _get_service_stub() as stub:
        res = await stub.GetOnboardingEmailSequenceLog(
            GetOnboardingEmailSequenceLogRequest(),
            metadata=get_default_grpc_metadata(user_id=user_id),
        )
        return [log for log in res.onboarding_email_sequence_log]
