from agent_service.io_type_utils import ComplexIOBase, io_type
from agent_service.io_types.text import Text


@io_type
class Idea(ComplexIOBase):
    title: str
    description: Text
