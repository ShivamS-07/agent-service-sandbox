import datetime
import json
import logging
from functools import lru_cache
from typing import Optional, cast

import sentry_sdk
from gbi_common_py_utils.utils.redis import is_redis_available
from user_service_proto_v1.user_service_pb2 import User

from agent_service.external.notification_svc_client import (
    get_onboarding_email_sequence_log,
)
from agent_service.external.user_svc_client import (
    REDIS_AGENT_USER_NAMESPACE,
    get_user_cached,
)
from agent_service.io_type_utils import IOType
from agent_service.utils.async_utils import run_async_background
from agent_service.utils.cache_utils import InMemoryCacheBackend, RedisCacheBackend
from agent_service.utils.logs import async_perf_logger

logger = logging.getLogger(__name__)

# if users are created before this date, they are ignored
# so we only consider new users from this date onwards
IGNORE_USERS_BEFORE_DATE = datetime.datetime(2024, 11, 19)


@lru_cache(maxsize=1)
def get_user_first_login_cache_client() -> Optional[RedisCacheBackend]:
    if not is_redis_available():
        return None
    return RedisCacheBackend(
        namespace=REDIS_AGENT_USER_NAMESPACE,
        serialize_func=json.dumps,
        deserialize_func=json.loads,
    )


FIRST_LOGIN_CACHE = InMemoryCacheBackend(maxsize=256, ttl=3600 * 24 * 7)  # 7 days


@async_perf_logger
async def is_user_first_login(user_id: str) -> IOType:
    val = await FIRST_LOGIN_CACHE.get(user_id)
    if val is False:  # we only cache False values
        logger.info(f"Get user {user_id} first login 'False' from in-memory TTL Cache!")
        return val

    cache_key = f"first_login_{user_id}"

    # Try to get from cache first
    redis_cache = get_user_first_login_cache_client()
    if redis_cache:
        with sentry_sdk.start_span(
            op="redis.get", description="Get user first login from Redis cache"
        ):
            cached_result = await redis_cache.get(cache_key)
            if cached_result is False:
                await FIRST_LOGIN_CACHE.set(user_id, False)
                return cached_result

    user_metadata = cast(User, await get_user_cached(user_id))
    user_created_date = user_metadata.user_created.ToDatetime()

    # ignore users who are created before this date
    if user_created_date < IGNORE_USERS_BEFORE_DATE:
        await FIRST_LOGIN_CACHE.set(user_id, False)
        if redis_cache:
            run_async_background(redis_cache.set(cache_key, False))

        return False

    log = await get_onboarding_email_sequence_log(user_id=user_id)
    if log:
        await FIRST_LOGIN_CACHE.set(user_id, False)
        if redis_cache:
            run_async_background(redis_cache.set(cache_key, False))

        return False

    # We don't want to cache that it's the user's first login
    return True
