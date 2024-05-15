import datetime
import functools
import json
from typing import Callable, List, Optional, Tuple, TypeVar

import backoff
from gbi_common_py_utils.utils.environment import (
    PROD_TAG,
    STAGING_TAG,
    get_environment_tag,
)
from gbi_common_py_utils.utils.ssm import get_param
from google.protobuf.timestamp_pb2 import Timestamp
from grpclib import GRPCError
from jwt import PyJWT
from jwt.algorithms import RSAAlgorithm


def get_default_grpc_metadata(
    user_id: str = "", cognito_groups: Optional[List[str]] = None
) -> List[Tuple[str, str]]:
    jwt_auth_token = create_jwt(user_id=user_id, cognito_groups=cognito_groups)
    return build_auth_metadata(jwt_auth_token)


def create_jwt(user_id: str, cognito_groups: Optional[List[str]] = None) -> str:
    tag = get_environment_tag()
    param_name = "/dev/token/webserver/jwk"
    if tag == PROD_TAG:
        param_name = "/alpha/token/webserver/jwk"
    elif tag == STAGING_TAG:
        param_name = "/staging/token/webserver/jwk"
    jwk_dict = json.loads(get_param(param_name))
    rsa_algo = RSAAlgorithm.from_jwk(jwk_dict["keys"][0])
    py_jwt = PyJWT()

    return py_jwt.encode(
        payload={
            "sub": user_id,
            "cognito:groups": [] if cognito_groups is None else cognito_groups,
            "iss": "https://boosted.ai",
        },
        headers={"kid": jwk_dict["keys"][0]["kid"], "alg": "RS256"},
        key=rsa_algo,
    )


def build_auth_metadata(auth_token: str) -> List[Tuple[str, str]]:
    # TODO handle audit log stuff
    metadata = [("authorization", auth_token)]
    return metadata


def date_to_timestamp(dt: datetime.date) -> Timestamp:
    timestamp = int(dt.strftime("%s"))
    return Timestamp(seconds=int(timestamp), nanos=int(timestamp % 1 * 1e9))


def timestamp_to_datetime(timestamp: Timestamp) -> datetime.datetime:
    return datetime.datetime.fromtimestamp(timestamp.seconds + timestamp.nanos / 1e9)


T = TypeVar("T")


def grpc_retry(func: Callable[..., T]) -> Callable[..., T]:
    """
    A generic retry decorator for gRPC calls - 3 retries with a `1 + uniform(0,1)` seconds interval
    Currently only retry on `GRPCError` in case of a network error, we can further customize to
    certain status code if needed.
    """

    @functools.wraps(func)
    async def run(*args, **kwargs) -> T:  # type: ignore
        return await backoff.on_exception(  # type: ignore
            backoff.constant,
            exception=(GRPCError,),
            interval=1,
            max_tries=3,
            jitter=backoff.random_jitter,
        )(func)(*args, **kwargs)

    return run  # type: ignore