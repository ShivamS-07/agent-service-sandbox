import enum
from abc import ABC, abstractmethod
from typing import List, Literal, Tuple, Type, Union

from pydantic import BaseModel, Field
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
    STRING_LIST = "string_list"
    INT_LIST = "int_list"


class IOBase(BaseModel, ABC):
    """
    Parent class of ALL types that may act as inputs or outputs to tools.
    """

    io_type: IOTypeEnum

    @abstractmethod
    def to_gpt_input(self) -> str:
        raise NotImplementedError()

    @classmethod
    def gpt_type_name(cls) -> str:
        """
        Returns the type for GPT function stub.
        """
        return cls.__name__

    def unwrap(self) -> "IOBase":
        return self


#################################
# PRIMITIVE IO TYPES START HERE
#################################

# Primitive IO types are a bit special, in that they are automatically converted
# from primitive python types when present in ToolArgs. This is done with
# pydantic validator ToolArgs.convert_to_io_types.


class IntIO(IOBase):
    io_type: Literal[IOTypeEnum.INTEGER] = IOTypeEnum.INTEGER
    val: int

    def to_gpt_input(self) -> str:
        return str(self.val)

    @classmethod
    def gpt_type_name(cls) -> str:
        return "int"

    def unwrap(self) -> int:
        return self.val


class StrIO(IOBase):
    io_type: Literal[IOTypeEnum.STRING] = IOTypeEnum.STRING
    val: str

    def to_gpt_input(self) -> str:
        return self.val

    @classmethod
    def gpt_type_name(cls) -> str:
        return "str"

    def unwrap(self) -> str:
        return self.val


class FloatIO(IOBase):
    io_type: Literal[IOTypeEnum.FLOAT] = IOTypeEnum.FLOAT
    val: float

    def to_gpt_input(self) -> str:
        return str(self.val)

    @classmethod
    def gpt_type_name(cls) -> str:
        return "float"

    def unwrap(self) -> float:
        return self.val


class BoolIO(IOBase):
    io_type: Literal[IOTypeEnum.BOOL] = IOTypeEnum.BOOL
    val: bool

    def to_gpt_input(self) -> str:
        return str(self.val)

    @classmethod
    def gpt_type_name(cls) -> str:
        return "bool"

    def unwrap(self) -> bool:
        return self.val


class StringList(IOBase):
    io_type: Literal[IOTypeEnum.STRING_LIST] = IOTypeEnum.STRING_LIST
    vals: List[str]

    def to_gpt_input(self) -> str:
        return str(self.vals)

    @classmethod
    def gpt_type_name(cls) -> str:
        return "List[str]"

    def unwrap(self) -> List[str]:
        return self.vals


class IntList(IOBase):
    io_type: Literal[IOTypeEnum.INT_LIST] = IOTypeEnum.INT_LIST
    vals: List[int]

    def to_gpt_input(self) -> str:
        return str(self.vals)

    @classmethod
    def gpt_type_name(cls) -> str:
        return "List[int]"

    def unwrap(self) -> List[int]:
        return self.vals


#################################
# PRIMITIVE IO TYPES END HERE
#################################


#################################
# IO TYPE DEFS END HERE
#################################

IOType = Union[_get_all_subclasses(IOBase)]  # type: ignore

# We do a bit of fancy metaprogramming to be able to do this. Essentially this
# allows pydantic to automatically deserialize json into specific IO
# subclasses based on the 'io_type' property.
io_adapter = TypeAdapter(Annotated[IOType, Field(discriminator="io_type")])
