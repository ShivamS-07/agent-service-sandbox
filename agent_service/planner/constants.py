import re
from enum import StrEnum

from agent_service.tool import ToolCategory

ASSIGNMENT_RE = re.compile(r"^([^=]+) = ([^\(]+)\((.*)\)  \# (.+)$")
ARGUMENT_RE = re.compile(r", [^=,/]+=")

RUN_EXECUTION_PLAN_FLOW_NAME = "run_execution_plan"
CREATE_EXECUTION_PLAN_FLOW_NAME = "create_execution_plan"

INITIAL_PLAN_TRIES = 3
EXECUTION_TRIES = 3
MIN_SUCCESSFUL_FOR_STOP = 2

WORKLOG_INTERVAL = 0.5  # every X seconds send a worklog event to FE


class FollowupAction(StrEnum):
    NONE = "NONE"
    RERUN = "RERUN"
    APPEND = "APPEND"
    REPLAN = "REPLAN"
    CREATE = "CREATE"
    LAYOUT = "LAYOUT"
    NOTIFICATION = "NOTIFICATION"


class FirstAction(StrEnum):
    # user asks for setting a notification
    NOTIFICATION = "NOTIFICATION"
    # user asks sth that should be refered to CS
    REFER = "REFER"
    # user asks for a task to be planned
    PLAN = "PLAN"
    # user asks irrelevant questions
    NONE = "NONE"


NO_CHANGE_MESSAGE = "Report updated, but no important differences found."
CHAT_DIFF_TEMPLATE = "Report updated with important changes found:\n{diff}"

MAX_SAMPLE_INPUT_MULTIPLER = 4

ALWAYS_AVAILABLE_TOOL_CATEGORIES = [ToolCategory.OUTPUT]
