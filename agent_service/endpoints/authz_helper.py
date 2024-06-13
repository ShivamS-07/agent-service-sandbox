import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, Optional

import jwt
from fastapi import HTTPException, Security, status
from fastapi.security.api_key import APIKeyHeader
from gbi_common_py_utils.utils.ssm import get_param

from agent_service.utils.environment import EnvironmentUtils
from agent_service.utils.postgres import get_psql

KEY_ID = "kid"
RS_256 = "RS256"
SUBJECT = "sub"
AUD = "o601d0dtctfaidudcanl2f3tt"

COGNITO_URLS = [
    "https://cognito-idp.us-west-2.amazonaws.com///us-west-2_csB4xjtUm/.well-known/jwks.json"
]
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


def extract_user_from_jwt(auth_token: str) -> Optional[User]:
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
            audience=AUD,
            algorithms=[RS_256],
        )
        sub = data.get(SUBJECT, None)
        user_id = data.get("custom:user_id", sub)
        groups = data.get("cognito:groups", [])
        is_super_admin = "super-admin" in groups
        is_admin = "admin" in groups
        return User(
            auth_token=auth_token,
            user_id=user_id,
            is_super_admin=is_super_admin,
            is_admin=is_admin,
        )
    except Exception as e:
        logger.exception(f"Failed to parse auth_token with exception: {e}")
        return None


def parse_header(auth_token: Optional[str] = Security(APIKeyHeader(name="Authorization"))) -> User:
    if not auth_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing JWT token in header"
        )

    if auth_token.startswith(COGNITO_PREFIX):
        auth_token = auth_token[len(COGNITO_PREFIX) :]

    user = extract_user_from_jwt(auth_token)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized user")
    return user


####################################################################################################
# auth methods - if we find that we need to add more auth methods, we can make a new file
####################################################################################################
def validate_user_agent_access(request_user_id: Optional[str], agent_id: str) -> None:
    if not request_user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="User is not authorized"
        )

    owner_id = get_psql().get_agent_owner(agent_id)
    if owner_id is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Agent {agent_id} not found"
        )
    elif owner_id != request_user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"User {request_user_id} is not authorized to access agent {agent_id}",
        )


def validate_user_plan_run_access(
    request_user_id: Optional[str], plan_run_id: str, require_owner: bool = True
) -> None:
    plan_run = get_psql().get_plan_run(plan_run_id)
    if plan_run is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Plan run {plan_run_id} not found",
        )

    # do not require auth for shared plan runs
    if plan_run.get("shared") and not require_owner:
        return

    agent_id = plan_run.get("agent_id")
    if agent_id is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent for plan run {plan_run_id} not found",
        )
    validate_user_agent_access(request_user_id, agent_id)
