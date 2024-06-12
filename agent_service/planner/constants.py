import re
from enum import Enum

ASSIGNMENT_RE = re.compile(r"^([^=]+) = ([^\(]+)\(([^)]*)\)  \# (.+)$")
ARGUMENT_RE = re.compile(r", [^=,]+=")

RUN_EXECUTION_PLAN_FLOW_NAME = "run_execution_plan"
CREATE_EXECUTION_PLAN_FLOW_NAME = "create_execution_plan"

INITIAL_PLAN_TRIES = 3
EXECUTION_TRIES = 3
MIN_SUCCESSFUL_FOR_STOP = 2

WORKLOG_INTERVAL = 0.5  # every X seconds send a worklog event to FE


class Action(str, Enum):
    NONE = "NONE"
    RERUN = "RERUN"
    APPEND = "APPEND"
    REPLAN = "REPLAN"
    CREATE = "CREATE"
    LAYOUT = "LAYOUT"
