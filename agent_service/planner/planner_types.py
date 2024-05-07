from dataclasses import dataclass
from typing import Dict


@dataclass
class ParsedStep:
    output_var: str
    function: str
    arguments: Dict[str, str]
    description: str
