import enum
from abc import ABC, abstractmethod
from typing import Generic, List, Literal, Optional, Tuple, Type, TypeVar, Union

from pydantic import BaseModel, Field, field_validator
from pydantic.type_adapter import TypeAdapter
from typing_extensions import Annotated

PrimitiveType = Union[int, str, bool, float]


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
    io_type: IOTypeEnum

    @abstractmethod
    def to_gpt_input(self) -> str:
        raise NotImplementedError()


#################################
# IO TYPES START HERE
#################################


class IntIO(IOBase):
    io_type: Literal[IOTypeEnum.INTEGER] = IOTypeEnum.INTEGER
    val: int

    def to_gpt_input(self) -> str:
        return str(self.val)


class StrIO(IOBase):
    io_type: Literal[IOTypeEnum.STRING] = IOTypeEnum.STRING
    val: str

    def to_gpt_input(self) -> str:
        return self.val


class FloatIO(IOBase):
    io_type: Literal[IOTypeEnum.FLOAT] = IOTypeEnum.FLOAT
    val: float

    def to_gpt_input(self) -> str:
        return str(self.val)


class BoolIO(IOBase):
    io_type: Literal[IOTypeEnum.BOOL] = IOTypeEnum.BOOL
    val: bool

    def to_gpt_input(self) -> str:
        return str(self.val)


T = TypeVar("T", bound=PrimitiveType)


class ListIO(IOBase, Generic[T]):
    io_type: Literal[IOTypeEnum.LIST] = IOTypeEnum.LIST
    vals: List[T]

    def to_gpt_input(self) -> str:
        return str(self.vals)


#################################
# IO TYPES END HERE
#################################

IOType = Union[_get_all_subclasses(IOBase)]  # type: ignore

# We do a bit of fancy metaprogramming to be able to do this. Essentially this
# allows pydantic to automatically deserialize json into specific IO
# subclasses based on the 'io_type' property.
io_adapter = TypeAdapter(Annotated[IOType, Field(discriminator="io_type")])
