import enum
from typing import List, Literal, Optional, Tuple, Type, Union

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
    INTEGER = "integer"
    STRING = "string"
    BOOL = "bool"
    FLOAT = "float"
    LIST = "list"


class IOBase(BaseModel):
    io_type: IOTypeEnum


#################################
# IO TYPES START HERE
#################################


class IntIO(IOBase):
    io_type: Literal[IOTypeEnum.INTEGER] = IOTypeEnum.INTEGER
    val: int


class ListIO(IOBase):
    io_type: Literal[IOTypeEnum.LIST] = IOTypeEnum.LIST
    vals: List[PrimitiveType]

    def get_list_type(self) -> Optional[Type]:
        return type(self.vals[0]) if self.vals else None


#################################
# IO TYPES END HERE
#################################

IOType = Union[_get_all_subclasses(IOBase)]  # type: ignore

# We do a bit of fancy metaprogramming to be able to do this. Essentially this
# allows pydantic to automatically deserialize json into specific IO
# subclasses based on the 'io_type' property.
io_adapter = TypeAdapter(Annotated[IOType, Field(discriminator="io_type")])
