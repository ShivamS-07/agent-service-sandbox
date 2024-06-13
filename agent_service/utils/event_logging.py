from typing import Any, Dict, Optional

from gbi_common_py_utils.utils.event_logging import log_event as gbi_common_log_event

from agent_service.GPT.constants import CLIENT_NAMESPACE


def log_event(
    event_name: str,
    event_data: Optional[Dict[str, Any]] = None,
    event_namespace: Optional[str] = None,
) -> None:
    if CLIENT_NAMESPACE != "LOCAL":
        gbi_common_log_event(
            event_name=event_name, event_data=event_data, event_namespace=event_namespace
        )
