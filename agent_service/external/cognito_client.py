import logging
import os
from functools import lru_cache

import aioboto3
import backoff
import sentry_sdk
from async_lru import alru_cache
from gbi_common_py_utils.utils.redis import is_redis_available

from agent_service.utils.async_utils import run_async_background
from agent_service.utils.cache_utils import RedisCacheBackend
from agent_service.utils.string_utils import is_valid_uuid

COGNITO_SESSION: aioboto3.Session | None = None
logger = logging.getLogger(__name__)

REGION_NAME = os.environ.get("AWS_DEFAULT_REGION", "us-west-2")
USER_POOL_ID = "us-west-2_csB4xjtUm"
USER_POOL_JWKS_URL = (
    "https://cognito-idp.us-west-2.amazonaws.com///us-west-2_csB4xjtUm/.well-known/jwks.json"
)
COGNITO_URLS = [USER_POOL_JWKS_URL]
COGNITO_USER_ID = "custom:user_id"


class CognitoRetriableException(Exception):
    pass


@lru_cache(maxsize=1)
def get_cognito_redis_cache() -> RedisCacheBackend | None:
    if not is_redis_available():
        return None
    return RedisCacheBackend(
        namespace="cognito_user_lookup",
        serialize_func=lambda b: str(b).encode("utf-8"),
        deserialize_func=lambda s: s.decode("utf-8"),
    )


@alru_cache(maxsize=256)
@backoff.on_exception(
    backoff.expo,  # 1s, 2s, 4s
    exception=CognitoRetriableException,
    max_tries=3,
    logger=logger,
    jitter=backoff.random_jitter,
)
async def get_cognito_user_id_from_access_token(access_token: str) -> str:
    redis_cache = get_cognito_redis_cache()
    if redis_cache:
        with sentry_sdk.start_span(op="redis.get", description="aws.cognito.get_user"):
            cached_user_id: str | None = await redis_cache.get(access_token)  # type: ignore
            if is_valid_uuid(cached_user_id):
                return cached_user_id  # type: ignore

    global COGNITO_SESSION
    if COGNITO_SESSION is None:
        COGNITO_SESSION = aioboto3.Session(region_name=REGION_NAME)

    with sentry_sdk.start_span(op="aws.cognito.get_user", description="get_user"):
        async with COGNITO_SESSION.client("cognito-idp") as cognito_client:
            try:
                cognito_user = await cognito_client.get_user(AccessToken=access_token)
            except (
                cognito_client.exceptions.TooManyRequestsException,
                cognito_client.exceptions.InternalErrorException,
            ) as e:
                logger.warning(f"Error getting user from cognito: {e}")
                raise CognitoRetriableException from e

            user_attributes = cognito_user.get("UserAttributes", [])
            user_id = next(
                (
                    attribute.get("Value")
                    for attribute in user_attributes
                    if attribute.get("Name") == "custom:user_id"
                ),
                None,
            )

            if not user_id:
                raise CognitoRetriableException(
                    f'Could not get field "custom:user_id" from {access_token}'
                )

            if redis_cache:
                _ = run_async_background(redis_cache.set(access_token, user_id, ttl=3600))

            return user_id
