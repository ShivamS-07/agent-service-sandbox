from abc import ABC
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Type,
    TypeVar,
    Union,
    cast,
    get_args,
)

import pandas as pd
from pydantic import BaseModel, field_validator
from pydantic.config import ConfigDict
from pydantic.functional_serializers import model_serializer

PrimitiveType = Union[int, str, bool, float, List[int], List[str], List[bool], List[float]]

_IO_TYPE_NAME_KEY = "_io_type"


class ComplexIOBase(BaseModel, ABC):
    """
    Parent class of ALL types that may act as inputs or outputs to tools.
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    val: Any

    def to_gpt_input(self) -> str:
        return str(self.val)

    @classmethod
    def name(cls) -> str:
        return cls.__name__

    @field_validator("val", mode="before")
    @classmethod
    def _deserializer(cls, val):
        val_field = cls.model_fields["val"]
        if isinstance(val, dict) and val_field.annotation is pd.DataFrame:
            val = pd.DataFrame.from_dict(val)
        return val

    @model_serializer(mode="wrap")
    def _serialize_io_base(self, dumper: Callable) -> Dict[str, Any]:
        if isinstance(self.val, pd.DataFrame):
            self.val = self.val.to_dict()

        # This calls the default pydantic serializer
        ser_dict = dumper(self)
        ser_dict[_IO_TYPE_NAME_KEY] = self.name()
        return ser_dict


IOType = Union[PrimitiveType, ComplexIOBase]


def type_is_primitive(typ: Optional[Type]) -> bool:
    return typ in get_args(PrimitiveType)


def get_clean_type_name(typ: Optional[Type]) -> str:
    try:
        return typ.__name__  # type: ignore
    except AttributeError:
        return str(typ)


class IOTypeSerializer:
    _COMPLEX_TYPE_DICT: Dict[str, Type[ComplexIOBase]] = {}

    # Use these to dump and load supported primitive types, as well as more complex IO Types.
    @classmethod
    def dump_io_type_dict(cls, val: IOType) -> Union[PrimitiveType, Dict[str, Any]]:
        if type_is_primitive(type(val)):
            val = cast(PrimitiveType, val)
            return val

        val = cast(ComplexIOBase, val)
        return val.model_dump()

    @classmethod
    def parse_io_type_dict(cls, val: Union[PrimitiveType, Dict[str, Any]]) -> IOType:
        if type_is_primitive(type(val)):
            val = cast(PrimitiveType, val)
            return val
        val = cast(Dict[str, Any], val)
        cls_name = val[_IO_TYPE_NAME_KEY]
        typ = cls._COMPLEX_TYPE_DICT.pop(cls_name)
        return typ(**val)


T = TypeVar("T", bound=ComplexIOBase)


def io_type(cls: Type[T]) -> Type[T]:
    """
    Simple decorator to store a mapping from class name to class.
    """
    IOTypeSerializer._COMPLEX_TYPE_DICT[cls.name()] = cls
    return cls
