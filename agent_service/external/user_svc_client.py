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
from user_service_proto_v1.user_service_grpc import UserServiceStub
from user_service_proto_v1.user_service_pb2 import (
    GetUsersRequest,
    ListTeamMembersRequest,
    ListTeamMembersResponse,
    UpdateUserRequest,
    User,
)
from user_service_proto_v1.well_known_types_pb2 import UUID

from agent_service.external.grpc_utils import get_default_grpc_metadata, grpc_retry
from agent_service.utils.logs import async_perf_logger

logger = logging.getLogger(__name__)

DEFAULT_URLS = {
    LOCAL_TAG: ("user-service-dev.boosted.ai", 50051),
    DEV_TAG: ("user-service-dev.boosted.ai", 50051),
    PROD_TAG: ("user-service-grpc-server-prod.us-west-2.elasticbeanstalk.com", 50051),
}


@lru_cache(maxsize=1)
def get_url_and_port() -> Tuple[str, int]:
    url = os.environ.get("USER_SERVICE_URL")
    if url is not None:
        logger.warning(f"Found USER_SERVICE_URL override: {url}")
        split_url, port = url.split(":")
        return split_url, int(port)

    env = get_environment_tag()
    if env not in DEFAULT_URLS:
        raise ValueError(f"No URL is set for environment {env}")
    return DEFAULT_URLS[env]


@contextmanager
def _get_service_stub() -> Generator[UserServiceStub, None, None]:
    try:
        url, port = get_url_and_port()
        print(f"url and port: {url} {port}")
        channel = Channel(url, port)
        yield UserServiceStub(channel)
    finally:
        channel.close()


@grpc_retry
@async_perf_logger
async def get_users(user_id: str, user_ids: List[str], include_user_enabled: bool) -> List[User]:
    with _get_service_stub() as stub:
        res = await stub.GetUsers(
            GetUsersRequest(
                user_id=[UUID(id=user_id) for user_id in user_ids],
                include_user_enabled=include_user_enabled,
            ),
            metadata=get_default_grpc_metadata(user_id=user_id),
        )
        return [u for u in res.user]


@grpc_retry
@async_perf_logger
async def update_user(user_id: str, name: str, username: str, email: str) -> bool:
    with _get_service_stub() as stub:
        await stub.UpdateUser(
            UpdateUserRequest(
                user_id=UUID(id=user_id), new_email=email, new_username=username, new_name=name
            ),
            metadata=get_default_grpc_metadata(user_id=user_id),
        )
        return True


@grpc_retry
@async_perf_logger
async def list_team_members(
    team_id: str, user_id: str, include_cognito_enabled: bool = False
) -> List[User]:
    with _get_service_stub() as stub:
        team_members: ListTeamMembersResponse = await stub.ListTeamMembers(
            ListTeamMembersRequest(
                team_id=UUID(id=team_id), include_cognito_enabled=include_cognito_enabled
            ),
            metadata=get_default_grpc_metadata(user_id=user_id),
        )
        return [user for user in team_members.users if user.cognito_enabled]
