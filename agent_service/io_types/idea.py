from typing import Any

from agent_service.io_type_utils import ComplexIOBase, io_type
from agent_service.io_types.text import Text


@io_type
class Idea(ComplexIOBase):
    title: str
    description: Text

    def __hash__(self) -> int:
        return self.title.__hash__()

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, Idea):
            return self.title == other.title
        return False
