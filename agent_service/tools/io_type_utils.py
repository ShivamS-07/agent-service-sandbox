import datetime
import json
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

SimpleType = Union[int, str, bool, float]

PrimitiveType = Union[
    int,
    str,
    bool,
    float,
    List[int],
    List[str],
    List[bool],
    List[float],
    List[Union[str, int, float, bool]],
    datetime.date,
    datetime.datetime,
]

IO_TYPE_NAME_KEY = "_type"


class ComplexIOBase(BaseModel, ABC):
    """
    Parent class of non-primitive types that may act as inputs or outputs to tools.
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    val: Any

    def to_gpt_input(self) -> str:
        return str(self.val)

    def __str__(self) -> str:
        return self.to_gpt_input()

    @classmethod
    def name(cls) -> str:
        return cls.__name__

    @field_validator("val", mode="before")
    @classmethod
    def _deserializer(cls, val: Any) -> Any:
        val_field = cls.model_fields["val"]
        if isinstance(val, dict) and val_field.annotation is pd.DataFrame:
            val = pd.DataFrame.from_dict(val)
        return val

    @model_serializer(mode="wrap")
    def _serialize_io_base(self, dumper: Callable) -> Dict[str, Any]:
        orig_val = self.val
        if isinstance(self.val, pd.DataFrame):
            self.val = self.val.to_dict()

        # This calls the default pydantic serializer
        ser_dict: Dict[str, Any] = dumper(self)
        # Don't overwrite in the class itself, only for serialization purposes
        self.val = orig_val
        ser_dict[IO_TYPE_NAME_KEY] = self.name()
        return ser_dict


IOType = Union[PrimitiveType, ComplexIOBase]


def type_is_primitive(typ: Optional[Type]) -> bool:
    return typ in get_args(PrimitiveType) or typ is list


def get_clean_type_name(typ: Optional[Type]) -> str:
    try:
        return typ.__name__  # type: ignore
    except AttributeError:
        return str(typ)


class IOTypeSerializer:
    _COMPLEX_TYPE_DICT: Dict[str, Type[ComplexIOBase]] = {}

    @classmethod
    def dump_io_type_dict(cls, val: IOType) -> Union[PrimitiveType, Dict[str, Any]]:
        if type_is_primitive(type(val)):
            if isinstance(val, datetime.datetime):
                val = _DatetimeIOType(val=val)
            elif isinstance(val, datetime.date):
                val = _DateIOType(val=val)
            else:
                val = cast(PrimitiveType, val)
                return val

        val = cast(ComplexIOBase, val)
        return val.model_dump()

    @classmethod
    def dump_io_type_json(cls, val: IOType) -> str:
        if type_is_primitive(type(val)):
            if isinstance(val, datetime.datetime):
                val = _DatetimeIOType(val=val)
            elif isinstance(val, datetime.date):
                val = _DateIOType(val=val)
            else:
                val = cast(PrimitiveType, val)
                return json.dumps(val)

        val = cast(ComplexIOBase, val)
        return val.model_dump_json()

    @classmethod
    def load_io_type_json(cls, val_str: str) -> IOType:
        val = json.loads(val_str)
        return cls.load_io_type_dict(val)

    @classmethod
    def load_io_type_dict(cls, serialized: Union[PrimitiveType, Dict[str, Any]]) -> IOType:
        if type_is_primitive(type(serialized)):
            serialized = cast(PrimitiveType, serialized)
            return serialized
        serialized = cast(Dict[str, Any], serialized)
        cls_name = serialized.pop(IO_TYPE_NAME_KEY)
        typ = cls._COMPLEX_TYPE_DICT[cls_name]
        val = serialized["val"]
        ret = typ(val=val)
        if isinstance(ret, (_DateIOType, _DatetimeIOType)):
            return ret.val
        return ret


T = TypeVar("T", bound=ComplexIOBase)


def io_type(cls: Type[T]) -> Type[T]:
    """
    Simple decorator to store a mapping from class name to class.
    """
    IOTypeSerializer._COMPLEX_TYPE_DICT[cls.name()] = cls
    return cls


def check_type_is_io_type(typ: Optional[Type]) -> bool:
    if not typ:
        return False
    if typ in get_args(IOType):
        return True
    try:
        if issubclass(typ, ComplexIOBase):
            return True
    except TypeError:
        return False
    return False


# Utility classes for wrapping dates. This will allow us to easily identify date
# strings without having to parse every string we encounter.


@io_type
class _DateIOType(ComplexIOBase):
    val: datetime.date

    @classmethod
    def name(cls) -> str:
        return "Date"


@io_type
class _DatetimeIOType(ComplexIOBase):
    val: datetime.datetime

    @classmethod
    def name(cls) -> str:
        return "DateTime"
