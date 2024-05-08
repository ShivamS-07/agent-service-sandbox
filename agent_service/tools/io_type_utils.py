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
    get_args,
    get_origin,
)

import pandas as pd
from pydantic import BaseModel, TypeAdapter, field_validator
from pydantic.config import ConfigDict
from pydantic.functional_serializers import field_serializer, model_serializer
from pydantic.functional_validators import model_validator
from pydantic_core.core_schema import ValidationInfo, ValidatorFunctionWrapHandler
from typing_extensions import TypeAliasType

SimpleType = Union[int, str, bool, float]

PrimitiveType = Union[
    int,
    str,
    bool,
    float,
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

    @field_serializer("val", mode="wrap")
    @classmethod
    def _field_serializer(cls, val: Any, dumper: Callable) -> Any:
        if isinstance(val, pd.DataFrame):
            val = val.to_dict()
        return dumper(val)

    @model_serializer(mode="wrap")
    def _serialize_io_base(self, dumper: Callable) -> Any:
        if not issubclass(type(self), ComplexIOBase):
            return self

        # This calls the default pydantic serializer
        ser_dict: Dict[str, Any] = dumper(self)
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

IOType = TypeAliasType(  # type: ignore[misc]
    "IOType",
    Union[PrimitiveType, List["IOType"], ComplexIOBase, "IOTypeDict"],  # type: ignore[misc]
)
# Need to do this due to limits of mypy and pydantic.
IOTypeDict = Union[  # type: ignore[misc]
    Dict[str, IOType],
    Dict[bool, IOType],
    Dict[float, IOType],
    Dict[int, IOType],
    Dict[SimpleType, IOType],
]

# A type adapter is a pydantic object used to dump and load objects that are not
# necessarily basemodels.
IOTypeAdapter = TypeAdapter(IOType)


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


def check_type_is_valid(actual: Optional[Type], expected: Optional[Type]) -> bool:
    if actual is None and expected is None:
        return True

    if not get_origin(expected) and not get_origin(actual):
        return actual is expected

    # Origin of generic types like List[str] -> list. For types like int, will
    # be None.
    orig_actual = get_origin(actual)
    orig_expected = get_origin(expected)

    # Args of generic types like List[str] -> str.
    params_actual = get_args(actual)
    params_expected = get_args(expected)

    if orig_actual is Union and orig_expected is Union:
        # This really should be "all" instead of "any", but that would
        # require doing some nonsense with generics. This is good enough for
        # now, other issues can be discovered at runtime.
        return any((val in params_expected for val in params_actual))
    elif orig_expected is Union and orig_actual is None:
        # int is valid if we expect Union[int, str]
        return actual in params_expected
    elif orig_actual is Union and orig_expected is None:
        # This technically also is always incorrect, but again without nasty
        # generic stuff we need to just handle it anyway. E.g. Union[str, int]
        # should not type check for just str, but it does now for simplicity.
        return expected in params_actual

    # In any case other than above, origin types must match
    if orig_actual is not orig_expected:
        return False

    # Now, we know that the origin types are the same, and they're not unions so
    # they're either Dict or List. We can recusrively check.
    for p1, p2 in zip(params_actual, params_expected):
        if not check_type_is_valid(p1, p2):
            return False

    return True


def check_type_is_io_type(typ: Optional[Type]) -> bool:
    if not typ:
        return False

    # Simple case
    primitive_types = set(get_args(PrimitiveType))
    if typ in primitive_types:
        return True

    if get_origin(typ) is Union:
        union_vals = get_args(typ)
        return all((val in primitive_types for val in union_vals))

    # Subclass case
    try:
        if issubclass(typ, ComplexIOBase):
            return True
    except TypeError:
        pass

    # List case, get_origin returns the base type without any type params
    # (e.g. List[int] -> list)
    if get_origin(typ) is list:
        elem_type_tup = get_args(typ)
        if not elem_type_tup:
            return False
        elem_type = elem_type_tup[0]
        # Using recursion here, TODO at some point maybe add a recursion limit?
        return check_type_is_io_type(elem_type)

    elif get_origin(typ) is dict:
        elem_type_tup = get_args(typ)
        if not elem_type_tup:
            return False
        key_type, elem_type = elem_type_tup
        if key_type not in get_args(SimpleType):
            return False
        return check_type_is_io_type(elem_type)

    return False
