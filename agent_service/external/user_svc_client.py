import logging
import os
from contextlib import contextmanager
from functools import lru_cache
from typing import Generator, List, Optional, Tuple

from gbi_common_py_utils.utils.environment import (
    DEV_TAG,
    LOCAL_TAG,
    PROD_TAG,
    get_environment_tag,
)
from gbi_common_py_utils.utils.redis import is_redis_available
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
from agent_service.utils.async_utils import run_async_background
from agent_service.utils.cache_utils import RedisCacheBackend
from agent_service.utils.feature_flags import is_user_agent_admin
from agent_service.utils.logs import async_perf_logger

logger = logging.getLogger(__name__)

DEFAULT_URLS = {
    LOCAL_TAG: ("user-service-dev.boosted.ai", 50051),
    DEV_TAG: ("user-service-dev.boosted.ai", 50051),
    PROD_TAG: ("user-service-grpc-server-prod.us-west-2.elasticbeanstalk.com", 50051),
}


REDIS_AGENT_USER_NAMESPACE = "redis-agent-user-namespace"


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


####################################################################################################
# GetUsers
####################################################################################################
@grpc_retry
@async_perf_logger
async def get_users(user_id: str, user_ids: List[str], include_user_enabled: bool) -> List[User]:
    with _get_service_stub() as stub:
        metadata = [
            *get_default_grpc_metadata(user_id=user_id),
            ("is-warren-agent-admin", "True" if await is_user_agent_admin(user_id) else "False"),
        ]
        res = await stub.GetUsers(
            GetUsersRequest(
                user_id=[UUID(id=user_id) for user_id in user_ids],
                include_user_enabled=include_user_enabled,
            ),
            metadata=metadata,
        )
        return [u for u in res.user]


def get_user_key_func(user_id: str) -> str:
    return f"{user_id}||True"


def get_user_serializer(user: User) -> bytes:
    return user.SerializeToString()


def get_user_deserializer(x: bytes) -> User:
    user = User()
    user.ParseFromString(x)
    return user


@lru_cache(maxsize=1)
def get_redis_cache_for_user() -> Optional[RedisCacheBackend]:
    if not is_redis_available():
        return None
    return RedisCacheBackend(
        namespace=REDIS_AGENT_USER_NAMESPACE,
        serialize_func=get_user_serializer,
        deserialize_func=get_user_deserializer,
    )


@async_perf_logger
async def get_user_cached(user_id: str) -> Optional[User]:
    """
    Get user's information from cache if available, otherwise fetch from user service and cache it.
    NOTE that we only cache the user when they have access to Alfa.
    TTL sets to 1 hour in case someone is removed from Alfa.
    """

    redis_cache = get_redis_cache_for_user()
    if not redis_cache:
        users = await get_users(user_id, user_ids=[user_id], include_user_enabled=True)
        return users[0] if users else None

    key = get_user_key_func(user_id)
    user = await redis_cache.get(key=key)
    if user is not None:
        return user  # type: ignore

    users = await get_users(user_id, user_ids=[user_id], include_user_enabled=True)
    if not users:
        return None

    user = users[0]
    if user.cognito_enabled.value and user.has_alfa_access:
        run_async_background(redis_cache.set(key, user, ttl=3600))

    return user


####################################################################################################
# UpdateUser
####################################################################################################
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

        redis_cache = get_redis_cache_for_user()
        if redis_cache:
            run_async_background(redis_cache.invalidate(get_user_key_func(user_id)))

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
