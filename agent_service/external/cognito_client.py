import logging
import os
from functools import lru_cache

import backoff
import boto3
from mypy_boto3_cognito_idp import CognitoIdentityProviderClient

COGNITO_CLIENT = None
logger = logging.getLogger(__name__)

REGION_NAME = os.environ.get("AWS_DEFAULT_REGION", "us-west-2")
USER_POOL_ID = "us-west-2_csB4xjtUm"
USER_POOL_JWKS_URL = (
    "https://cognito-idp.us-west-2.amazonaws.com///us-west-2_csB4xjtUm/.well-known/jwks.json"
)
COGNITO_URLS = [USER_POOL_JWKS_URL]
COGNITO_USER_ID = "custom:user_id"


def get_cognito_client() -> CognitoIdentityProviderClient:
    global COGNITO_CLIENT
    if COGNITO_CLIENT is None:
        COGNITO_CLIENT = boto3.client("cognito-idp", region_name=REGION_NAME)
    return COGNITO_CLIENT


@lru_cache(maxsize=256)
@backoff.on_exception(backoff.constant, exception=Exception, max_tries=3, logger=logger)
def get_cognito_user_id_from_access_token(access_token: str) -> str:
    cognito_client = get_cognito_client()
    cognito_user = cognito_client.get_user(
        AccessToken=access_token,
    )
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
        raise ValueError(f'Could not get field "custom:user_id" from {access_token}')

    return user_id
