from gbi_common_py_utils.utils.feature_flags import create_user_from_userid, get_ld_flag

from agent_service.utils.postgres import get_psql


def is_user_agent_admin(user_id: str, default: bool = False) -> bool:
    """
    Users with flag on can access some agent windows owned by other users. Currently the endpoints
    are:
    - `get_chat_history`
    - `get_agent_worklog_board`
    - `get_agent_worklog_output`
    - `get_agent_task_output`
    - `get_agent_output`
    - `get_agent_plan_output`
    - `steam_agent_events`
    """

    psql = get_psql()
    user_context = create_user_from_userid(user_id=user_id, db=psql)
    return get_ld_flag(flag_name="warren-agent-admin", user_context=user_context, default=default)
