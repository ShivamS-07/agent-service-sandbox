import re

ASSIGNMENT_RE = re.compile(r"^([^=]+) = ([^\(]+)\(([^)]*)\)  \# (.+)$")
ARGUMENT_RE = re.compile(r", [^=,]+=")

RUN_EXECUTION_PLAN_FLOW_NAME = "run_execution_plan"
CREATE_EXECUTION_PLAN_FLOW_NAME = "create_execution_plan"
