import datetime
import json
from abc import ABC
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union, get_args

import pandas as pd
from pydantic import BaseModel, field_validator
from pydantic.config import ConfigDict
from pydantic.functional_serializers import model_serializer
from pydantic.functional_validators import model_validator
from pydantic_core.core_schema import ValidationInfo, ValidatorFunctionWrapHandler

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

    @model_validator(mode="wrap")
    @classmethod
    def _model_deserializer(
        cls, data: Any, handler: ValidatorFunctionWrapHandler, _info: ValidationInfo
    ) -> Any:
        if isinstance(data, dict) and IO_TYPE_NAME_KEY in data:
            return cls.load(data)
        return handler(data)

    @classmethod
    def load(cls, data: Any) -> "ComplexIOBase":
        if isinstance(data, dict) and IO_TYPE_NAME_KEY in data:
            # If we get here, then we're looking at a ComplexIOType that's been
            # serialized. We want to return the actual type we want, so we check
            # the map.
            typ_name = data.pop(IO_TYPE_NAME_KEY)
            typ = _COMPLEX_TYPE_DICT[typ_name]
            return typ(**data)

        raise ValueError(f"Cannot load data without key {IO_TYPE_NAME_KEY}: {data}")

    @classmethod
    def load_json(cls, data: str) -> "ComplexIOBase":
        vals = json.loads(data)
        return cls.load(vals)


_COMPLEX_TYPE_DICT: Dict[str, Type[ComplexIOBase]] = {}

IOType = Union[PrimitiveType, ComplexIOBase]


def type_is_primitive(typ: Optional[Type]) -> bool:
    return typ in get_args(PrimitiveType) or typ is list


def get_clean_type_name(typ: Optional[Type]) -> str:
    try:
        return typ.__name__  # type: ignore
    except AttributeError:
        return str(typ)


T = TypeVar("T", bound=ComplexIOBase)


def io_type(cls: Type[T]) -> Type[T]:
    """
    Simple decorator to store a mapping from class name to class.
    """
    _COMPLEX_TYPE_DICT[cls.name()] = cls
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
