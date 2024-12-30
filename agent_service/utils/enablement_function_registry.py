import inspect
import logging
from typing import Any, Callable, Dict, List, Optional

from gbi_common_py_utils.utils.environment import (
    PROD_TAG,
    get_environment_tag,
)

from agent_service.planner.planner_types import SamplePlan
from agent_service.tools.web_search.general_websearch import websearch_enabler_function
from agent_service.types import PlanRunContext
from agent_service.utils.feature_flags import get_ld_flag
from agent_service.utils.pagerduty import pager_wrapper
from agent_service.utils.postgres import get_psql

logger = logging.getLogger(__name__)


def is_dev(args: Optional[Dict[str, Any]] = None) -> bool:
    return not is_prod(args)


def is_prod(_: Optional[Dict[str, Any]] = None) -> bool:
    return get_environment_tag() == PROD_TAG


def websearch(args: Optional[Dict[str, Any]] = None) -> bool:
    if args is None or args.get("user_id", None) is None:
        return False
    user_settings = get_psql(read_only=True).get_user_agent_settings(args["user_id"])
    return websearch_enabler_function(user_id=args["user_id"], user_settings=user_settings)


# KEYS MUST BE LOWERCASE STRINGS FOR THE REGISTRY
ENABLEMENT_FUNCTION_REGISTRY: Dict[str, Callable[[Optional[Dict[Any, Any]]], bool]] = {
    "dev": is_dev,
    "prod": is_prod,
    "websearch": websearch,
}


def keyword_search_sample_plans(search_input: str, enabled_only: bool = True) -> List[SamplePlan]:
    plans = get_psql().keyword_search_sample_plans_helper(search_input)
    if not enabled_only:
        return plans
    return [plan for plan in plans if is_plan_enabled(plan)]


def is_plan_enabled(
    plan: SamplePlan,
    context: Optional[Dict[str, str]] = None,
    user_id: Optional[str] = None,
) -> bool:
    enabled_lowercase = plan.enabled.lower()
    # null converts to empty-string with ::TEXT
    if enabled_lowercase in ("1", "true", "yes", "y", "t", ""):
        return True
    elif enabled_lowercase in ("0", "false", "no", "n", "f"):
        return False
    elif enabled_lowercase in ENABLEMENT_FUNCTION_REGISTRY and ENABLEMENT_FUNCTION_REGISTRY[
        enabled_lowercase
    ]({"user_id": user_id}):
        return True
    else:
        ld_flag_result = get_ld_flag(enabled_lowercase, default=None, user_context=user_id)
        try:
            assert ld_flag_result is not None
            return ld_flag_result
        except AssertionError as e:
            if not context:
                context = {}
            plan_run_context = PlanRunContext(
                plan_id=context.get("plan_id", ""),
                plan_run_id=context.get("plan_run_id", ""),
                agent_id=context.get("agent_id", ""),
                user_id=context.get("user_id", ""),
            )
            pager_wrapper(
                current_frame=inspect.currentframe(),
                module_name=__name__,
                context=plan_run_context,
                e=e,
                classt="SamplePlanEnablementError",
                summary=f"LD flag {enabled_lowercase} does not exist!",
            )
            logger.exception(f"LD flag {enabled_lowercase} does not exist!")
            return False
