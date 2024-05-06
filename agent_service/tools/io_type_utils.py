from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type, Union, cast, get_args

from pydantic import BaseModel, model_serializer
from pydantic.config import ConfigDict

# Recursive type defs don't work, so need to split over two lines.
_PrimitiveTypeBase = Union[int, str, bool, float]
PrimitiveType = Union[_PrimitiveTypeBase, List[_PrimitiveTypeBase]]


_IO_TYPE_NAME_KEY = "_io_type"


class ComplexIOBase(BaseModel, ABC):
    """
    Parent class of ALL types that may act as inputs or outputs to tools.
    """

    model_config = ConfigDict(extra="ignore")

    val: Any

    @abstractmethod
    def to_gpt_input(self) -> str:
        raise NotImplementedError()

    @classmethod
    def name(cls) -> str:
        return cls.__name__

    @model_serializer
    def serialize_model(self) -> Dict[str, Any]:
        ser_dict = self.model_dump()
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
        typ = cls._COMPLEX_TYPE_DICT[cls_name]
        return typ(**val)


def io_type(cls: Type[ComplexIOBase]) -> Type[ComplexIOBase]:
    """
    Simple decorator to store a mapping from class name to class.
    """
    IOTypeSerializer._COMPLEX_TYPE_DICT[cls.name()] = cls
    return cls
