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

PLAN_RUN_EMAIL_THRESHOLD_SECONDS = 600  # 10 minutes


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


NO_CHANGE_MESSAGE = "Agent analysis completed, but no important differences found."

REPORT_UPDATED_LINE = "Report updated"
CHAT_DIFF_TEMPLATE = (
    REPORT_UPDATED_LINE + " on ({date}) " + "with important changes found:" + "\n{diff}"
)  # if you changes this you need to update the add_unread_updates_message_to_chat_history
FOLLOW_UP_QUESTION = "Is there anything else I can assist you with?"

MAX_SAMPLE_INPUT_MULTIPLER = 4

ALWAYS_AVAILABLE_TOOL_CATEGORIES = [
    ToolCategory.OUTPUT,
    ToolCategory.LIST,
    ToolCategory.STOCK,
    ToolCategory.TEXT_RETRIEVAL,
]

PASS_CHECK_OUTPUT = "The plan fully covers the client request"

PARSING_FAIL_LINE = "The parsing failed on the following line: {line}. "
