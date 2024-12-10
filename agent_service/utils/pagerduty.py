import traceback
from typing import Any, Optional

from gbi_common_py_utils.utils.pagerduty import PD_WARNING, notify_agent_pg

from agent_service.types import PlanRunContext
from agent_service.utils import environment


def pager_wrapper(
    current_frame: Optional[Any],
    module_name: str,
    context: PlanRunContext,
    e: Exception,
    classt: str,
    summary: str,
) -> None:
    # some of the exception messages are long and contain unique info,
    # truncate it to try to prevent that from making too many unique pagers
    error_dedupe_str = str(type(e)) + " " + str(e)[:75]
    func_name = module_name
    if current_frame:
        # defined to be potentially null, in practice it never is
        func_name = current_frame.f_code.co_name
    group = f"{classt}-{func_name}-{error_dedupe_str}"

    notify_agent_pg(
        summary=f"{func_name}: {summary}: {error_dedupe_str}",
        severity=PD_WARNING,
        source=environment.get_environment_tag(),
        component="AgentError",
        classt=classt,
        group=group,
        custom_details={
            "_reminder": "This pager is deduped, check #oncall-info for more examples",
            "agent": context.agent_id,
            "plan_run": context.plan_run_id,
            "task": context.task_id,
            "error": "".join(traceback.TracebackException.from_exception(e).format()),
            "pagerduty_dedupe_key": group,
        },
    )
