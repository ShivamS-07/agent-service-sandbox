import logging
from functools import lru_cache
from typing import Any, Dict, Optional, Union

import ldclient
from gbi_common_py_utils import get_config
from gbi_common_py_utils.utils import ssm
from gbi_common_py_utils.utils.feature_flags import (
    LDUser,
    create_anonymous_user,
    create_user_from_userid,
)
from gbi_common_py_utils.utils.redis import redis_cached

from agent_service.utils.postgres import get_psql
from definitions import CONFIG_PATH

logger = logging.getLogger(__name__)


def __init_ld_client() -> ldclient.LDClient:
    """
    Initialize an LD client using the SDK key from our local config.
    """
    config = get_config(local_dir_path=CONFIG_PATH)
    sdk_key = ssm.get_param(config.ld_sdk_key)
    ldclient.set_config(ldclient.Config(sdk_key, flush_interval=5 * 60))
    return ldclient.get()


@lru_cache(maxsize=128)
def get_user_context(user_id: str) -> LDUser:
    """
    Retrieve a user context from a user id.
    """
    psql = get_psql()
    return create_user_from_userid(user_id=user_id, db=psql)


def get_ld_flag(flag_name: str, default: Any, user_context: Union[None, str, LDUser]) -> Any:
    """
    Evaluate a flag for a user, defaulting to some value if the flag is not available.
    """
    client = None
    try:
        client = ldclient.get()
    except Exception:
        __init_ld_client()
        client = ldclient.get()

    if isinstance(user_context, str):
        try:
            result_context = get_user_context(user_context)
        except Exception:
            logger.warning(f"Bad userID string input: {user_context}")
            result_context = create_anonymous_user()
    elif user_context is None:
        result_context = create_anonymous_user()
    else:
        result_context = user_context

    result = client.variation(flag_name, result_context.to_dict(), default)
    return result


# modified dictionary/object for frontend to use
def get_custom_user_dict(user_context: LDUser) -> Dict[str, Optional[Union[str, bool]]]:
    return {
        "kind": "user",
        "key": user_context.key,
        "name": user_context.name,
        "email": user_context.email,
        "anonymous": user_context.anonymous,
        "company_id": user_context.company_id,
    }


def get_secure_mode_hash(user_context: LDUser) -> str:
    """
    Creates a hash string that is used by the frontend.
    """
    client = None
    try:
        client = ldclient.get()
    except Exception:
        __init_ld_client()
        client = ldclient.get()
    if user_context is None:
        user_context = create_anonymous_user()

    return client.secure_mode_hash(get_custom_user_dict(user_context=user_context))


@redis_cached(
    namespace="is_user_agent_admin",
    key_func=lambda user_id: user_id,
    serialize_func=lambda b: str(b).encode("utf-8"),
    deserialize_func=lambda s: s == b"True",
    ttl=3600,
)
def is_user_agent_admin(user_id: str) -> bool:
    """
    Users with flag on can access some agent windows owned by other users. Currently the endpoints
    are:
    - `get_chat_history`
    - `get_agent_worklog_board`
    - `get_agent_worklog_output`
    - `get_agent_task_output`
    - `get_agent_output`
    - `get_agent_plan_output`
    - `stream_agent_events`
    """

    return get_ld_flag(
        flag_name="warren-agent-admin", user_context=get_user_context(user_id), default=False
    )


def user_has_qc_tool_access(user_id: str, default: bool = False) -> bool:
    """
    Users with flag on can access some agent windows owned by other users. Currently the endpoints
    are:
    - `get_qc_agent_by_id`
    - `get_qc_agent_by_user`
    - `search_agent_qc`
    - `update_agent_qc`
    """

    return get_ld_flag(
        flag_name="horizon-qc-tool", user_context=get_user_context(user_id), default=default
    )


def use_boosted_dag_for_run_execution_plan() -> bool:
    return get_ld_flag(flag_name="boosted-dag-run-execution-plan", user_context=None, default=False)


def agent_output_cache_enabled() -> bool:
    return get_ld_flag(flag_name="agent-output-cache", user_context=None, default=False)


def is_database_access_check_enabled_for_user(user_id: str) -> bool:
    return get_ld_flag(
        flag_name="database-access-check", user_context=get_user_context(user_id), default=False
    )
