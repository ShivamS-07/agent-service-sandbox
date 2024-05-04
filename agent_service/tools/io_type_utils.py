import enum
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Literal, Tuple, Type, Union, get_args

from pydantic import BaseModel, Field
from pydantic.config import ConfigDict
from pydantic.type_adapter import TypeAdapter
from typing_extensions import Annotated

PrimitiveType = Union[int, str, bool, float, List[str]]


def _get_all_subclasses(cls: Type) -> Tuple[Type]:
    all_subclasses = []

    for subclass in cls.__subclasses__():
        all_subclasses.append(subclass)
        all_subclasses.extend(_get_all_subclasses(subclass))

    return tuple(all_subclasses)


class IOTypeEnum(str, enum.Enum):
    """
    Enum of labels for ALL IOType's. When creating a new IOType, a new entry
    must be added here to be used in the `io_type` property. This will allow
    pydantic to serialize and deserialize to the correct subclass automatically.
    """

    INTEGER = "integer"
    STRING = "string"
    BOOL = "bool"
    FLOAT = "float"
    LIST = "list"


class IOBase(BaseModel, ABC):
    """
    Parent class of ALL types that may act as inputs or outputs to tools.
    """

    model_config = ConfigDict(extra="forbid")

    io_type: IOTypeEnum
    val: Any

    @abstractmethod
    def to_gpt_input(self) -> str:
        raise NotImplementedError()


#################################
# PRIMITIVE IO TYPES
#################################

# These IO Types should not be used directly. They are simply wrappers around
# some python primitives so that everything can be serialized in a consistent
# way.

class _IntIO(IOBase):
    io_type: Literal[IOTypeEnum.INTEGER] = IOTypeEnum.INTEGER
    val: int

    def to_gpt_input(self) -> str:
        return str(self.val)


class _StrIO(IOBase):
    io_type: Literal[IOTypeEnum.STRING] = IOTypeEnum.STRING
    val: str

    def to_gpt_input(self) -> str:
        return self.val


class _FloatIO(IOBase):
    io_type: Literal[IOTypeEnum.FLOAT] = IOTypeEnum.FLOAT
    val: float

    def to_gpt_input(self) -> str:
        return str(self.val)


class _BoolIO(IOBase):
    io_type: Literal[IOTypeEnum.BOOL] = IOTypeEnum.BOOL
    val: bool

    def to_gpt_input(self) -> str:
        return str(self.val)


class _ListIO(IOBase):
    """
    Stores list literals. For simplicity, all lists are represented as lists of
    strings. Tools that take lists can convert the types as needed.
    """

    io_type: Literal[IOTypeEnum.LIST] = IOTypeEnum.LIST
    val: List[str]

    def to_gpt_input(self) -> str:
        return str(self.val)

ComplexIOType = Union[_get_all_subclasses(IOBase)]  # type: ignore

IOType = Union[PrimitiveType, ComplexIOType]  # type: ignore


def type_is_primitive(typ: Type) -> bool:
    return typ in get_args(PrimitiveType)


def get_clean_type_name(typ: Type[IOType]) -> str:
    return typ.__name__


# We do a bit of fancy metaprogramming to be able to do this. Essentially this
# allows pydantic to automatically deserialize json into specific IO
# subclasses based on the 'io_type' property.
io_adapter = TypeAdapter(Annotated[IOType, Field(discriminator="io_type")])


def dump_io_type(val: IOType) -> Dict[str, Any]:
    if isinstance(val, int):
        val = _IntIO(val=val)
    elif isinstance(val, str):
        val = _StrIO(val=val)
    elif isinstance(val, bool):
        val = _BoolIO(val=val)
    elif isinstance(val, float):
        val = _FloatIO(val=val)
    elif isinstance(val, list):
        val = _ListIO(val=val)

    return io_adapter.dump_python(val)


def parse_io_type(val: Dict[str, Any]) -> IOType:
    io_type = io_adapter.validate_python(val)
    if issubclass(io_type, (_IntIO, _StrIO, _BoolIO, _FloatIO, _ListIO)):
        return io_type.val
    return io_type
