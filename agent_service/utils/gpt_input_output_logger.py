import json
import logging
from datetime import date, datetime
from typing import Any, Dict, NamedTuple, Optional

from gbi_common_py_utils.utils.environment import get_environment_tag
from gbi_common_py_utils.utils.event_logging import log_event

logger = logging.getLogger(__name__)


def json_serial(obj: Any) -> str:
    """JSON serializer for objects not serializable by default json code"""

    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    raise TypeError("Type %s not serializable" % type(obj))


class GPTRow(NamedTuple):
    model_id: str
    main_prompt: str
    sys_prompt: str
    response: str
    latency_seconds: float
    num_input_tokens: int
    num_output_tokens: int
    context: str  # json
    environment: str = get_environment_tag()


class GPTInputOutputLogger:
    @classmethod
    def log_request_response(
        cls,
        gpt_model_key: str,
        system_prompt: str,
        main_prompt: str,
        response: str,
        latency_seconds: float,
        num_input_tokens: int,
        num_output_tokens: int,
        context: Optional[Dict[str, str]] = None,
    ) -> None:
        if context is None:
            context = {}
        extra_params = {}
        try:
            extra_params["prompt_id"] = context.get("job_type", "") + context.get("task_type", "")
        except Exception:
            logger.info("Error parsing gpt context")
        row_to_insert = GPTRow(
            model_id=gpt_model_key,
            sys_prompt=system_prompt,
            main_prompt=main_prompt,
            response=response,
            latency_seconds=latency_seconds,
            num_input_tokens=num_input_tokens,
            num_output_tokens=num_output_tokens,
            context=json.dumps(context, default=json_serial),
        )._asdict()
        row_with_extra_params = {**row_to_insert, **extra_params}
        log_event(event_name="llm-inference", event_data=row_with_extra_params)
