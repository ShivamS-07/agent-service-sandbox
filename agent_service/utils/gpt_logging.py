import inspect
from typing import Dict, Optional
from uuid import uuid4


class GptJobType:
    AGENT_PLANNER = "agent_planner"
    AGENT_CHATBOT = "agent_chatbot"
    AGENT_Q_AND_A = "agent_q_and_a"
    AGENT_TOOLS = "agent_tools"


class GptJobIdType:
    AGENT_ID = "agent_id"


GPT_TASK_TYPE = "task_type"

MAIN_PROMPT_NAME = "main_prompt_name"
SYS_PROMPT_NAME = "sys_prompt_name"
MAIN_PROMPT_TEMPLATE_NAME = "main_prompt_template"
SYS_PROMPT_TEMPLATE_NAME = "sys_prompt_template"
MAIN_PROMPT_TEMPLATE_ARGS = "main_prompt_template_args"
SYS_PROMPT_TEMPLATE_ARGS = "sys_prompt_template_args"


def create_gpt_context(
    job_type: str, job_id: str, job_id_type: str, run_id: Optional[str] = None
) -> Dict[str, str]:
    if not run_id:
        run_id = str(uuid4())
    return {"job_type": job_type, "job_id": job_id, "job_id_type": job_id_type, "run_id": run_id}


def plan_create_context(agent_id: str, user_id: str, plan_id: str) -> Dict[str, str]:
    context = create_gpt_context(GptJobType.AGENT_PLANNER, agent_id, GptJobIdType.AGENT_ID)
    context["user_id"] = user_id
    context["plan_id"] = plan_id
    context["agent_id"] = agent_id
    return context


def chatbot_context(agent_id: str, user_id: Optional[str] = None) -> Dict[str, str]:
    if not user_id:
        from agent_service.utils.postgres import get_psql

        user_id = get_psql().get_agent_owner(agent_id=agent_id)

    context = create_gpt_context(GptJobType.AGENT_CHATBOT, agent_id, GptJobIdType.AGENT_ID)
    context["agent_id"] = agent_id
    if user_id:
        context["user_id"] = user_id
    return context


def q_and_a_context(agent_id: str, user_id: Optional[str] = None) -> Dict[str, str]:
    if not user_id:
        from agent_service.utils.postgres import get_psql

        user_id = get_psql().get_agent_owner(agent_id=agent_id)

    context = create_gpt_context(GptJobType.AGENT_Q_AND_A, agent_id, GptJobIdType.AGENT_ID)
    context["agent_id"] = agent_id
    if user_id:
        context["user_id"] = user_id
    return context


def get_gpt_task_type() -> str:
    stack = inspect.stack()

    if len(stack) >= 3:
        # Not adding the immediate caller because it is same for all gpt calls
        caller = stack[2].function if hasattr(stack[2], "function") else None
        if caller is not None:
            return caller

    return "Unknown"
