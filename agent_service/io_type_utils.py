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

from pydantic import BaseModel, TypeAdapter
from pydantic.config import ConfigDict
from pydantic.functional_serializers import WrapSerializer, model_serializer
from pydantic.functional_validators import PlainValidator, model_validator
from pydantic_core.core_schema import ValidationInfo, ValidatorFunctionWrapHandler
from typing_extensions import Annotated, TypeAliasType

from agent_service.io_types.output import Output

SimpleType = Union[int, str, bool, float]

PrimitiveType = Union[
    datetime.date,
    datetime.datetime,
    int,
    str,
    bool,
    float,
]

IO_TYPE_NAME_KEY = "_type"


class ComplexIOBase(BaseModel, ABC):
    """
    Parent class of non-primitive types that may act as inputs or outputs to tools.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def to_gpt_input(self) -> str:
        return str(self.__class__)

    def to_rich_output(self) -> Output:
        """
        Converts a ComplexIOType to rich output that powers the frontend.
        """
        # By default, just return json text. This should be overriden by types
        # that are to be displayed to the frontend.
        raise NotImplementedError("This type does not have a rich output")

    @classmethod
    def type_name(cls) -> str:
        return cls.__name__

    @model_serializer(mode="wrap")
    def _serialize_io_base(self, dumper: Callable) -> Any:
        if not issubclass(type(self), ComplexIOBase):
            return self

        # This calls the default pydantic serializer
        ser_dict: Dict[str, Any] = dumper(self)
        ser_dict[IO_TYPE_NAME_KEY] = self.type_name()
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

IOTypeBase = TypeAliasType(  # type: ignore
    "IOTypeBase",
    Union[PrimitiveType, List["IOTypeBase"], ComplexIOBase, "IOTypeDict"],  # type: ignore
)


# Below functions are just used to match the required pydantic function schemas.
def _load_io_type_wrapper(v: Any) -> Any:
    return _load_io_type_helper(v)


def _dump_io_type_wrapper(v: Any, _: Any) -> Any:
    return _dump_io_type_helper(v)


IOType = TypeAliasType(  # type: ignore
    "IOType",
    Annotated[
        IOTypeBase,
        PlainValidator(_load_io_type_wrapper),
        WrapSerializer(_dump_io_type_wrapper),
    ],
)
# Need to do this due to limits of mypy and pydantic.
IOTypeDict = Union[  # type: ignore
    Dict[str, IOTypeBase],
    Dict[bool, IOTypeBase],
    Dict[float, IOTypeBase],
    Dict[int, IOTypeBase],
    Dict[SimpleType, IOTypeBase],
]

# A type adapter is a pydantic object used to dump and load objects that are not
# necessarily basemodels.
IOTypeAdapter = TypeAdapter(IOTypeBase)


def _dump_io_type_helper(val: IOTypeBase) -> Any:
    if isinstance(val, ComplexIOBase):
        return val.model_dump(mode="json")
    if isinstance(val, list):
        return [_dump_io_type_helper(elem) for elem in val]
    if isinstance(val, dict):
        return {k: _dump_io_type_helper(v) for k, v in val.items()}
    return IOTypeAdapter.dump_python(val, mode="json")


def _load_io_type_helper(val: Any) -> IOTypeBase:
    if isinstance(val, dict) and IO_TYPE_NAME_KEY in val:
        return ComplexIOBase.load(val)
    if isinstance(val, list):
        return [_load_io_type_helper(elem) for elem in val]
    if isinstance(val, dict):
        return {k: _load_io_type_helper(v) for k, v in val.items()}
    return IOTypeAdapter.validate_python(val)


def dump_io_type(val: IOType) -> str:
    return json.dumps(_dump_io_type_helper(val))


def load_io_type(val: str) -> IOType:
    loaded = json.loads(val)
    return _load_io_type_helper(loaded)


def get_clean_type_name(typ: Optional[Type]) -> str:
    if not typ:
        return str(typ)

    try:
        if issubclass(typ, ComplexIOBase):
            return typ.__name__
    except TypeError:
        pass
    name = str(typ)

    # Cleanup
    name = name.replace("agent_service.io_types.", "")
    name = name.replace("typing.", "")
    name = name.replace("IOType", "Any")
    return name


T = TypeVar("T", bound=ComplexIOBase)


def io_type(cls: Type[T]) -> Type[T]:
    """
    Simple decorator to store a mapping from class name to class.
    """
    _COMPLEX_TYPE_DICT[cls.type_name()] = cls
    return cls


def check_type_is_valid(actual: Optional[Type], expected: Optional[Type]) -> bool:
    if actual is None and expected is None:
        return True

    if expected in (IOType, Any):
        return True

    # TODO revisit this later, this will really help with preventing false
    # positives
    if actual and expected:
        try:
            if issubclass(actual, ComplexIOBase) and issubclass(expected, ComplexIOBase):
                return True
        except TypeError:
            pass

    if not get_origin(expected) and not get_origin(actual):
        return (
            actual is expected
            or expected is IOType
            or actual is IOType
            or (actual is not None and expected in actual.__bases__)
        )

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
    elif orig_expected is Union and orig_actual in (None, list, dict):
        # int is valid if we expect Union[int, str]
        return actual in params_expected or params_expected is IOType
    elif orig_actual is Union and orig_expected in (None, list, dict):
        # This technically also is always incorrect, but again without nasty
        # generic stuff we need to just handle it anyway. E.g. Union[str, int]
        # should not type check for just str, but it does now for simplicity.
        return expected in params_actual or params_expected is IOType

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

    if typ is IOType:
        return True

    primitive_types = set(get_args(PrimitiveType))
    if typ in primitive_types:
        return True

    if get_origin(typ) is Union:
        union_vals = get_args(typ)
        return all((check_type_is_io_type(val) for val in union_vals))

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
