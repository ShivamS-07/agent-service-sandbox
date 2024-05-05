import enum
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Type, Union, get_args

from pydantic import BaseModel, Field
from pydantic.config import ConfigDict
from pydantic.type_adapter import TypeAdapter
from typing_extensions import Annotated

PrimitiveType = Union[int, str, bool, float, List["PrimitiveType"]]


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
    STOCK_TIMESERIES = "stock_timeseries"


class ComplexIOBase(BaseModel, ABC):
    """
    Parent class of ALL types that may act as inputs or outputs to tools.
    """

    model_config = ConfigDict(extra="forbid")

    io_type: IOTypeEnum
    val: Any

    @abstractmethod
    def to_gpt_input(self) -> str:
        raise NotImplementedError()


ComplexIOType = Union[_get_all_subclasses(ComplexIOBase)]  # type: ignore

IOType = Union[PrimitiveType, ComplexIOType]


def type_is_primitive(typ: Type) -> bool:
    return typ in get_args(PrimitiveType)


def get_clean_type_name(typ: Type[IOType]) -> str:
    return typ.__name__


# We do a bit of fancy metaprogramming to be able to do this. Essentially this
# allows pydantic to automatically deserialize json into specific IO
# subclasses based on the 'io_type' property.
io_adapter = TypeAdapter(Annotated[IOType, Field(discriminator="io_type")])


# Use these to dump and load supported primitive types, as well as more complex IO Types.
def dump_io_type(val: IOType) -> Union[PrimitiveType, Dict[str, Any]]:
    if type_is_primitive(type(val)):
        return val

    return io_adapter.dump_python(val)


def parse_io_type(val: Union[PrimitiveType, Dict[str, Any]]) -> IOType:
    if type_is_primitive(type(val)):
        return val
    return io_adapter.validate_python(val)
