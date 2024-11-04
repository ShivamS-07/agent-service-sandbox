import logging
import time
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Dict, List, Optional

import jwt
from fastapi import HTTPException, Security, status
from fastapi.security.api_key import APIKeyHeader
from gbi_common_py_utils.utils.ssm import get_param
from starlette.requests import Request

from agent_service.external.cognito_client import (
    COGNITO_URLS,
    get_cognito_user_id_from_access_token,
)
from agent_service.utils.async_db import AsyncDB
from agent_service.utils.cache_utils import RedisCacheBackend
from agent_service.utils.environment import EnvironmentUtils

KEY_ID = "kid"
RS_256 = "RS256"
SUBJECT = "sub"

SSM_KEYS = ["token/services/jwk"]

COGNITO_PREFIX = "Cognito "

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class User:
    auth_token: str  # it can be useful to pass to other services
    user_id: str  # user id from user service

    # Flag to indicate if the current user is a super admin
    is_super_admin: bool = False
    is_admin: bool = False
    groups: List[str] = field(default_factory=list)
    real_user_id: str = ""
    fullstory_link: str = ""

    def __str__(self) -> str:
        admin_str = " (is super admin)" if self.is_super_admin else ""
        return f"{self.user_id}{admin_str}"


@lru_cache(maxsize=1)
def get_keyid_to_key_map() -> Dict[Any, jwt.PyJWK]:
    """
    Creates a mapping of keyIDs defined in Cognito to PyJWK (JSON Web Token) objects.
    Returns: A mapping of Cognito Key IDs to PyJWK tokens.

    """
    logger.info("Initializing JWT Keys from Cognito...")
    key_id_map = {}
    for url in COGNITO_URLS:
        client = jwt.PyJWKClient(url)
        keys = client.get_jwk_set()
        for key in keys.keys:
            key_id_map[key.key_id] = key

    logger.info("Initializing JWT Keys from SSM...")
    environment = EnvironmentUtils.aws_ssm_prefix
    for ssm_key in SSM_KEYS:
        param = get_param(f"/{environment}/{ssm_key}")
        key_set = jwt.PyJWKSet.from_json(param)
        for key in key_set.keys:
            key_id_map[key.key_id] = key

    return key_id_map


def extract_user_from_jwt(
    auth_token: str, real_user_id: str, fullstory_link: str
) -> Optional[User]:
    try:
        header = jwt.get_unverified_header(auth_token)
        key_id = header.get(KEY_ID, None)
        if key_id is None:
            raise ValueError(f"Missing key id in token {auth_token}")

        keyid_to_key_map = get_keyid_to_key_map()
        signing_key = keyid_to_key_map.get(key_id, None)
        if signing_key is None:
            raise ValueError(
                f"Missing signing key for {key_id} (known keys are:\n"
                + "\n".join(keyid_to_key_map.keys())
                + f"\n)\nin token {auth_token}"
            )
        data = jwt.decode(
            auth_token,
            signing_key.key,
            algorithms=[RS_256],
        )

        # Check if the token is expired
        if "exp" in data and data["exp"] < time.time():
            raise ValueError("Token has expired")

        sub = data.get(SUBJECT, None)
        user_id = get_cognito_user_id_from_access_token(auth_token)
        groups = data.get("cognito:groups", [])
        is_super_admin = "super-admin" in groups
        is_admin = "admin" in groups
        return User(
            auth_token=auth_token,
            user_id=user_id if user_id else sub,
            is_super_admin=is_super_admin,
            is_admin=is_admin,
            groups=groups,
            real_user_id=real_user_id,
            fullstory_link=fullstory_link,
        )
    except Exception as e:
        logger.exception(f"Failed to parse auth_token with exception: {e}")
        return None


def parse_header(
    request: Request, auth_token: Optional[str] = Security(APIKeyHeader(name="Authorization"))
) -> User:
    user_info = getattr(request.state, "user_info", None)
    if user_info:
        return user_info
    if not auth_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing JWT token in header"
        )

    if auth_token.startswith(COGNITO_PREFIX):
        auth_token = auth_token[len(COGNITO_PREFIX) :]
    real_user_id = request.headers.get("realuserid", "")
    fullstory_link = request.headers.get("fullstorylink", "")
    user = extract_user_from_jwt(
        auth_token, real_user_id=real_user_id, fullstory_link=fullstory_link
    )
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized user")
    return user


####################################################################################################
# auth methods - if we find that we need to add more auth methods, we can make a new file
####################################################################################################
def get_agent_owner_redis() -> RedisCacheBackend:
    return RedisCacheBackend(
        namespace="agent_owner",
        serialize_func=lambda user_id: user_id.encode("utf-8"),
        deserialize_func=lambda user_id: user_id.decode("utf-8"),
    )


async def get_agent_owner(
    agent_id: str, async_db: AsyncDB, no_cache: bool = False
) -> Optional[str]:
    if no_cache:
        return await async_db.get_agent_owner(agent_id, include_deleted=False)

    redis_cache = get_agent_owner_redis()
    owner_id = await redis_cache.get(agent_id)
    if owner_id:
        return owner_id  # type: ignore

    owner_id = await async_db.get_agent_owner(agent_id, include_deleted=False)
    if owner_id:
        await redis_cache.set(agent_id, owner_id, ttl=3600 * 24)

    return owner_id


async def invalidate_agent_owner_cache(agent_id: str) -> None:
    redis_cache = get_agent_owner_redis()
    await redis_cache.invalidate(agent_id)


async def validate_user_agent_access(
    request_user_id: Optional[str], agent_id: str, async_db: AsyncDB, invalidate_cache: bool = False
) -> None:
    """
    Validates whether the request user is the owner of the agent.
    - The agent must not be deleted, otherwise `owner_id` will be None, thus raising Exception
    - We use Redis to cache the owner_id as the owner cannot be changed
    - If `invalidate_cache` is True, we will use DB to get the owner, and remove cache from Redis.
    This flag is currently only used by `/delete-agent` endpoint.
    """

    if not request_user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="User is not authorized"
        )

    owner_id = await get_agent_owner(agent_id, async_db, no_cache=invalidate_cache)
    if owner_id is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Agent {agent_id} not found"
        )
    elif owner_id != request_user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"User {request_user_id} is not authorized to access agent {agent_id}",
        )

    if invalidate_cache:
        await invalidate_agent_owner_cache(agent_id)


async def validate_user_plan_run_access(
    request_user_id: Optional[str],
    plan_run_id: str,
    async_db: AsyncDB,
    require_owner: bool = True,
) -> None:
    sql = """
        SELECT agent_id::VARCHAR, plan_id::VARCHAR, created_at, shared
        FROM agent.plan_runs
        WHERE plan_run_id = %(plan_run_id)s
        LIMIT 1
    """
    rows = await async_db.generic_read(sql, params={"plan_run_id": plan_run_id})
    if not rows:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Plan run {plan_run_id} not found",
        )

    # do not require auth for shared plan runs
    plan_run = rows[0]
    if plan_run.get("shared") and not require_owner:
        return

    agent_id = plan_run.get("agent_id")
    if agent_id is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent for plan run {plan_run_id} not found",
        )
    await validate_user_agent_access(request_user_id, agent_id, async_db)
