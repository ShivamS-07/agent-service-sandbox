import datetime
import json
from typing import cast

from user_service_proto_v1.user_service_pb2 import User

from agent_service.external.notification_svc_client import (
    get_onboarding_email_sequence_log,
)
from agent_service.external.user_svc_client import (
    REDIS_AGENT_USER_NAMESPACE,
    get_user_cached,
)
from agent_service.io_type_utils import IOType
from agent_service.utils.cache_utils import RedisCacheBackend
from agent_service.utils.feature_flags import is_user_agent_admin

# if users are created before this date, they are ignored
# so we only consider new users from this date onwards
IGNORE_USERS_BEFORE_DATE = datetime.datetime(2024, 11, 1)

cache = RedisCacheBackend(
    namespace=REDIS_AGENT_USER_NAMESPACE,
    serialize_func=json.dumps,
    deserialize_func=json.loads,
)


async def is_user_first_login(user_id: str) -> IOType:
    cache_key = f"first_login_{user_id}"

    # Try to get from cache first
    cached_result = await cache.get(cache_key)
    if cached_result is not None:
        return cached_result

    user_metadata = cast(User, await get_user_cached(user_id))
    user_created_date = datetime.datetime.fromtimestamp(user_metadata.user_created.seconds)

    # ignore external users who are created before this date
    if not is_user_agent_admin(user_id) and user_created_date < IGNORE_USERS_BEFORE_DATE:
        await cache.set(cache_key, False)
        return False

    log = await get_onboarding_email_sequence_log(user_id=user_id)
    if log:
        await cache.set(cache_key, False)
        return False

    # We don't want to cache that it's the user's first login
    return True
