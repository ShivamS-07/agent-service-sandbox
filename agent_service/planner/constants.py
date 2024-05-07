import re

ASSIGNMENT_RE = re.compile(r"^([^=]+) = ([^\(]+)\(([^)]*)\)  \# (.+)$")
ARGUMENT_RE = re.compile(r", [^=,]+=")
