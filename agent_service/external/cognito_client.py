import logging
import os

import aioboto3
import backoff
import sentry_sdk
from async_lru import alru_cache

COGNITO_SESSION: aioboto3.Session | None = None
logger = logging.getLogger(__name__)

REGION_NAME = os.environ.get("AWS_DEFAULT_REGION", "us-west-2")
USER_POOL_ID = "us-west-2_csB4xjtUm"
USER_POOL_JWKS_URL = (
    "https://cognito-idp.us-west-2.amazonaws.com///us-west-2_csB4xjtUm/.well-known/jwks.json"
)
COGNITO_URLS = [USER_POOL_JWKS_URL]
COGNITO_USER_ID = "custom:user_id"


@alru_cache(maxsize=256)
@backoff.on_exception(
    backoff.expo,  # 1s, 2s, 4s
    exception=Exception,
    max_tries=3,
    logger=logger,
    jitter=backoff.random_jitter,
)
async def get_cognito_user_id_from_access_token(access_token: str) -> str:
    global COGNITO_SESSION
    if COGNITO_SESSION is None:
        COGNITO_SESSION = aioboto3.Session(region_name=REGION_NAME)

    with sentry_sdk.start_span(op="aws.cognito.get_user", description="get_user"):
        async with COGNITO_SESSION.client("cognito-idp") as cognito_client:
            cognito_user = await cognito_client.get_user(AccessToken=access_token)

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
