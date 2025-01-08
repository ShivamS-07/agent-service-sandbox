import datetime
import enum
import hashlib
import json
import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from copy import deepcopy
from types import UnionType
from typing import (
    Annotated,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Self,
    Set,
    Type,
    TypeAliasType,
    TypeVar,
    Union,
    cast,
    get_args,
    get_origin,
)

from pydantic import BaseModel, TypeAdapter
from pydantic.config import ConfigDict
from pydantic.functional_serializers import WrapSerializer, model_serializer
from pydantic.functional_validators import PlainValidator, model_validator
from pydantic_core.core_schema import ValidationInfo, ValidatorFunctionWrapHandler

from agent_service.io_types.output import CitationOutput, Output
from agent_service.utils.async_utils import gather_with_concurrency
from agent_service.utils.boosted_pg import BoostedPG

logger = logging.getLogger(__name__)

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


class TableColumnType(enum.StrEnum):
    # Raw values
    INTEGER = "integer"
    STRING = "string"
    FLOAT = "float"
    BOOLEAN = "boolean"

    # Values with units
    INTEGER_WITH_UNIT = "integer_with_unit"
    FLOAT_WITH_UNIT = "float_with_unit"

    # A currency valued number
    CURRENCY = "currency"

    # date relevant
    DATE = "date"  # YYYY-MM-DD
    DATETIME = "datetime"  # yyyy-mm-dd + ISO timestamp
    QUARTER = "quarter"  # Year + quarter
    YEAR = "year"
    MONTH = "month"  # yyyy-mm

    # Float value where 1.0 = 100%
    PERCENT = "percent"

    # Values for showing changes, anything above zero = green, below zero = red
    DELTA = "delta"  # Raw float delta
    PCT_DELTA = "pct_delta"  # Float delta value where 1.0 = 100% change

    # Special type that stores StockID instances
    STOCK = "stock"

    # Special type that stores scores
    SCORE = "score"

    @staticmethod
    def get_type_explanations() -> str:
        """
        Get a string to explain to the LLM what each table column type means (if
        not obvious).
        """
        return (
            "- 'integer': A generic integer.\n"
            "- 'string': A generic string.\n"
            "- 'float': A generic float.\n"
            "- 'bool': A generic bool.\n"
            "- 'currency': A column containing a price or other float with a currency attached."
            "In this case the 'unit' is the currency ISO, please keep that consistent.\n"
            "- 'date': A column containing a python date object."
            "- 'datetime': A column containing a python datetime object."
            "- 'quarter': A column containing a year + financial quarter represented as a string (yyyyQq)."
            " For example, 2024Q2."
            "- 'year': A column containing a year represented as a string (yyyy). Note it is not an integer!"
            "- 'month': A column containing a year + month represented as a string (yyyy-mm)"
            "- 'percent': A column containing a percent value float. 100% is equal to 1.0, NOT 100. "
            "E.g. 25 percent is represented as 0.25. NEVER multiply by 100 to get this.\n"
            "- 'delta': A float value representing a raw change over time. E.g. price "
            "change day over day.\n"
            "- 'pct_delta': A float value representing a change over time as a percent. "
            "100% is equal to 1.0 NOT 100. E.g. percent change of price day over day. "
            "NEVER multiply by 100 to get this.\n"
            "- 'stock': A special column containing stock identifier information. "
            "In general, any stock representation (e.g. ticker, symbol, etc.) should use this type"
            "- 'score': This is a special column type, do not use this."
        )

    def is_date_type(self) -> bool:
        return self in (
            TableColumnType.DATE,
            TableColumnType.DATETIME,
            TableColumnType.QUARTER,
            TableColumnType.YEAR,
            TableColumnType.MONTH,
        )

    def is_float_type(self) -> bool:
        return self in (
            TableColumnType.FLOAT,
            TableColumnType.FLOAT_WITH_UNIT,
            TableColumnType.CURRENCY,
            TableColumnType.PERCENT,
            TableColumnType.DELTA,
            TableColumnType.PCT_DELTA,
        )

    def to_data_type(self) -> Type["IOType"]:
        if self in (TableColumnType.INTEGER, TableColumnType.INTEGER_WITH_UNIT):
            return int
        elif self in (
            TableColumnType.STRING,
            TableColumnType.QUARTER,
            TableColumnType.YEAR,
            TableColumnType.MONTH,
        ):
            return str
        elif self in (
            TableColumnType.FLOAT,
            TableColumnType.FLOAT_WITH_UNIT,
            TableColumnType.CURRENCY,
            TableColumnType.PERCENT,
            TableColumnType.DELTA,
            TableColumnType.PCT_DELTA,
        ):
            return float
        elif self in (TableColumnType.BOOLEAN):
            return bool
        elif self in (TableColumnType.DATE):
            return datetime.date
        elif self in (TableColumnType.DATETIME):
            return datetime.datetime
        elif self in (TableColumnType.STOCK):
            from agent_service.io_types.stock import StockID

            return StockID
        elif self in (TableColumnType.SCORE):
            # TODO
            return str
        raise NotImplementedError(f"{self} has no mapping to a data type!!")


class SerializeableBase(BaseModel, ABC):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # See https://docs.pydantic.dev/latest/concepts/serialization/#overriding-the-serialize_as_any-default-false
    # For reasoning behind 'serialize_as_any'. We want this behavior pretty much
    # everywhere, so we can override these methods to always use it.
    def model_dump(self, **kwargs: Any) -> Dict[str, Any]:
        kwargs.setdefault("serialize_as_any", True)
        return super().model_dump(**kwargs)

    def model_dump_json(self, **kwargs: Any) -> str:
        kwargs.setdefault("serialize_as_any", True)
        return super().model_dump_json(**kwargs)

    @classmethod
    def type_name(cls) -> str:
        return cls.__name__

    @classmethod
    def load(cls, data: Any) -> "SerializeableBase":
        if isinstance(data, dict) and IO_TYPE_NAME_KEY in data:
            # If we get here, then we're looking at a ComplexIOType that's been
            # serialized. We want to return the actual type we want, so we check
            # the map.
            typ_name = data.pop(IO_TYPE_NAME_KEY)
            typ = _COMPLEX_TYPE_DICT[typ_name]
            return typ(**data)

        raise ValueError(f"Cannot load data without key {IO_TYPE_NAME_KEY}: {data}")

    @model_serializer(mode="wrap")
    def _serialize_io_base(self, dumper: Callable) -> Any:
        if not issubclass(type(self), SerializeableBase):
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
    def load_json(cls, data: str) -> "SerializeableBase":
        vals = json.loads(data)
        return cls.load(vals)


_COMPLEX_TYPE_DICT: Dict[str, Type[SerializeableBase]] = {}

T = TypeVar("T", bound=SerializeableBase)


def io_type(cls: Type[T]) -> Type[T]:
    """
    Simple decorator to store a mapping from class name to class.
    """
    _COMPLEX_TYPE_DICT[cls.type_name()] = cls
    return cls


@io_type
class Citation(SerializeableBase, ABC):
    """
    Generic class for tracking citations internally. Any child classes should
    implement ways to resolve these citations into output citations that will be
    displayed to the user.
    """

    @abstractmethod
    def __hash__(self) -> int:
        pass

    @abstractmethod
    def __eq__(self, other: Any) -> bool:
        pass

    @classmethod
    @abstractmethod
    async def resolve_citations(
        cls, citations: List[Self], db: BoostedPG
    ) -> Dict[Self, List[CitationOutput]]:
        """
        Given a list of citations of the class's type, resolve them to an output
        list of citations.
        """
        pass

    @staticmethod
    async def resolve_all_citations_mapped(
        citations: List["Citation"], db: BoostedPG
    ) -> Dict["Citation", List[CitationOutput]]:
        """
        Given a list of ANY type of citation, resolve them all and put them in a
        list. NOTE: order is NOT preserved.
        """
        if not citations:
            return {}
        citation_type_map = defaultdict(list)
        for cit in citations:
            citation_type_map[type(cit)].append(cit)

        tasks = [
            typ.resolve_citations(citation_list, db)
            for typ, citation_list in citation_type_map.items()
        ]
        # List of lists, where each nested list has outputs for each type
        outputs_nested: List[Dict[Citation, List[CitationOutput]]] = await gather_with_concurrency(
            tasks
        )
        mapped_citations: Dict[Citation, List[CitationOutput]] = defaultdict(list)
        for output_dict in outputs_nested:
            for cit, output_cits in output_dict.items():
                mapped_citations[cit].extend(output_cits)

        return mapped_citations

    @staticmethod
    async def resolve_all_citations(
        citations: List["Citation"], db: BoostedPG
    ) -> List[CitationOutput]:
        mapped_citations = await Citation.resolve_all_citations_mapped(citations=citations, db=db)
        return list((cit for cit_list in mapped_citations.values() for cit in cit_list))


@io_type
class Score(SerializeableBase):
    # Generally between 0 and 1
    val: float

    def __lt__(self, other: Self) -> bool:
        return self.val < other.val

    def rescale(self, lb: float, ub: float) -> float:
        return self.val * (ub - lb) + lb

    @classmethod
    def scale_input(cls, val: float, lb: float, ub: float) -> Self:
        """
        Anything that is not in [0, 1] range should use this helper method to scale
        """
        return cls(val=(val - lb) / (ub - lb))

    @classmethod
    def average(cls, scores: List[Self]) -> Self:
        """
        Average any number of scores
        """
        return cls(val=sum([sub_score.val for sub_score in scores]) / len(scores) if scores else 0)


@io_type
class ScoreOutput(SerializeableBase):
    val: float
    # Source of the score
    source: Optional[str] = None
    # At the end of a workflow, we may aggregate scores into groups with
    # averaged values. This list stores the sub-components.
    sub_scores: List["SubScoreOutput"] = []

    def __lt__(self, other: Self) -> bool:
        return self.val < other.val

    @staticmethod
    def from_entry_list(entries: List["HistoryEntry"]) -> Optional["ScoreOutput"]:
        aggregate_score = 0.0
        num_scores = 0
        sub_scores = []
        for entry in entries:
            if entry.score:
                aggregate_score += entry.score.val
                num_scores += 1
                sub_scores.append(SubScoreOutput(val=entry.score.val, source=entry.title))
        if num_scores == 0:
            return None

        aggregate_score = aggregate_score / num_scores
        return ScoreOutput(val=aggregate_score, sub_scores=sub_scores)

    def to_gpt_input(self) -> str:
        if not self.sub_scores:
            return str(self.val)
        else:
            sub_scores = ", ".join((f"{score.source}: {score.val}" for score in self.sub_scores))
            return f"(Overall Score: {self.val}, {sub_scores})"


@io_type
class SubScoreOutput(ScoreOutput):
    pass


@io_type
class HistoryEntry(SerializeableBase):
    explanation: Optional["IOType"] = None  # type: ignore
    title: str = ""
    entry_type: TableColumnType = TableColumnType.STRING
    unit: Optional[str] = None
    # Citations for the explanation
    citations: List[Citation] = []
    # Some steps in the plan may insert scores to be aggregated at the end
    score: Optional[Score] = None
    # task id is useful for doing some diff stuff
    task_id: Optional[str] = None

    def __hash__(self) -> int:
        return hash((self.explanation, self.title, self.entry_type, self.unit))

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, HistoryEntry):
            return (self.explanation, self.title, self.entry_type, self.unit) == (
                other.explanation,
                other.title,
                other.entry_type,
                other.unit,
            )
        return NotImplemented


class ComplexIOBase(SerializeableBase, ABC):
    """
    Parent class of non-primitive types that may act as inputs or outputs to tools.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Tracks the history of the object across various filters, summarizations, etc.
    history: List[HistoryEntry] = []

    title: Optional[str] = None

    async def to_gpt_input(self, use_abbreviated_output: bool = True) -> str:
        return str(self.__class__)

    def history_to_str_with_text_objects(self) -> str:
        """
        Returns a string (markdown) representation of the history entries in this object.
        """
        from agent_service.io_types.text_objects import BasicTextObject

        strs = []
        for entry in self.history:
            entry_repr = entry.explanation
            if not entry.title or not entry.explanation:
                continue

            if entry.entry_type != TableColumnType.STRING and isinstance(
                entry.explanation, get_args(PrimitiveType)
            ):
                entry.explanation = cast(PrimitiveType, entry.explanation)
                # In this case, create a text object to represent tye type so
                # that it can be nicely displayed on the frontend.
                obj = BasicTextObject(
                    type=entry.entry_type, value=entry.explanation, index=0, unit=entry.unit
                )
                entry_repr = BasicTextObject.render_object_to_json(obj)

            strs.append(f"- **{entry.title}**: {entry_repr}")

        return "\n".join(strs)

    async def to_rich_output(self, pg: BoostedPG, title: str = "") -> Output:
        """
        Converts a ComplexIOType to rich output that powers the frontend.
        """
        raise NotImplementedError("This type does not have a rich output")

    async def split_into_components(self, main_title: Optional[str] = None) -> List["IOType"]:
        """
        Some types might want to split themselves to be displayed or stored
        separately. By default, just return a list with one item.
        """
        return [self]

    # Note on deepcopy below: mostly from paranoia about mutable data and
    # references being shared across many objects. Deepcopying ensures that
    # modifying the history of one object never updates the history of another
    # object. This allows us to work in a purely functional style without
    # worrying about mutability.

    def dedup_history(self) -> None:
        # Do a loop to make sure we preserve ordering
        new_history = []
        seen = set()
        for entry in self.history:
            if entry not in seen:
                new_history.append(entry)
                seen.add(entry)

        self.history = new_history

    def inject_history_entry(self, entry: HistoryEntry) -> Self:
        new = deepcopy(self)
        new.history.append(deepcopy(entry))
        # TODO this should be much more efficient
        self.dedup_history()
        return new

    def _extend_history_from(self, other: Self) -> Self:
        new = deepcopy(self)
        new.history.extend(deepcopy(other.history))
        return new

    def union_history_with(self, other: Self) -> Self:
        new = deepcopy(self)
        new.history.extend(deepcopy(other.history))
        new.dedup_history()
        return new

    def get_all_citations(self) -> List[Citation]:
        citations = []
        for entry in self.history:
            citations.extend(entry.citations)
        return citations

    def __hash__(self) -> int:
        serialized_obj = dump_io_type(self, sort_keys=True)

        # Hash the JSON string using SHA-256
        return int(hashlib.sha256(serialized_obj.encode("utf-8")).hexdigest(), 16)

    def __lt__(self, other: Any) -> bool:
        # This can be overridden by children, for now just do it randomly
        return hash(self) < hash(other)

    @classmethod
    def union_sets(cls, set1: Set[Self], set2: Set[Self]) -> Set[Self]:
        dict1 = {hash(val): val for val in set1}
        dict2 = {hash(val): val for val in set2}
        output = set()
        for val in set1.union(set2):
            key = hash(val)
            if key in dict1 and key in dict2:
                # If it's in both, merge the histories
                new_val = dict1[key]
                output.add(new_val._extend_history_from(dict2[key]))
            else:
                output.add(val)

        return output

    @classmethod
    def intersect_sets(cls, set1: Set[Self], set2: Set[Self]) -> Set[Self]:
        dict1 = {hash(val): val for val in set1}
        dict2 = {hash(val): val for val in set2}
        output = set()
        for val in set1.intersection(set2):
            key = hash(val)
            new_val = dict1[key]
            output.add(new_val._extend_history_from(dict2[key]))

        return output


IOTypeBase = TypeAliasType(  # type: ignore
    "IOTypeBase",
    Union[PrimitiveType, List["IOTypeBase"], ComplexIOBase, "IOTypeDict"],  # type: ignore
)


# Below functions are just used to match the required pydantic function schemas.
def _load_io_type_wrapper(v: Any) -> Any:
    return load_io_type_dict(v)


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
IOTypeAdapter: TypeAdapter = TypeAdapter(IOTypeBase)

TUPLE_IO_TYPE_NAME_KEY = "tuple"
TUPLE_IO_TYPE_VAL_KEY = "val"


def _dump_io_type_helper(val: IOTypeBase, sort_keys: bool = False) -> Any:
    if isinstance(val, SerializeableBase):
        return val.model_dump(mode="json")
    if isinstance(val, tuple):
        # TECHNICALLY tuples are not allowed, but for some reason mypy is dumb
        # and doesn't catch tuples in io types... so now we need to support
        # them. Json doesn't support tuples so we need to wrap them in an abject
        # to preserve type info.
        return _dump_io_type_helper(
            {IO_TYPE_NAME_KEY: TUPLE_IO_TYPE_NAME_KEY, TUPLE_IO_TYPE_VAL_KEY: list(val)}
        )
    if isinstance(val, list):
        return [_dump_io_type_helper(elem) for elem in val]
    if isinstance(val, dict):
        return {k: _dump_io_type_helper(v) for k, v in val.items()}
    return IOTypeAdapter.dump_python(val, mode="json")


def load_io_type_dict(val: Any) -> IOTypeBase:
    if isinstance(val, dict) and IO_TYPE_NAME_KEY in val:
        if val[IO_TYPE_NAME_KEY] == TUPLE_IO_TYPE_NAME_KEY:
            return tuple(load_io_type_dict(val["val"]))  # type: ignore
        else:
            return SerializeableBase.load(val)
    if isinstance(val, list):
        return [load_io_type_dict(elem) for elem in val]
    if isinstance(val, dict):
        return {k: load_io_type_dict(v) for k, v in val.items()}
    if val is None:
        return None  # type: ignore
    return IOTypeAdapter.validate_python(val)


async def split_io_type_into_components(
    val: IOType, main_title: Optional[str] = None
) -> List[IOType]:
    """
    Some IOType's are composed of other IOTypes that need to be split out
    sometimes. This function handles all IOTypes.
    """
    if isinstance(val, ComplexIOBase):
        return await val.split_into_components(main_title)
    return [val]


def dump_io_type(val: IOType, sort_keys: bool = False) -> str:
    return json.dumps(_dump_io_type_helper(val), sort_keys=sort_keys)


def safe_dump_io_type(val: IOType, errmsg: Optional[str] = None) -> Optional[str]:
    s = None
    try:
        s = dump_io_type(val)
    except Exception:
        if not errmsg:
            errmsg = "failed to serialize"
        logger.exception(errmsg)
    return s


def load_io_type(val: str) -> Optional[IOType]:
    loaded = json.loads(val)
    if loaded is None:
        return None
    return load_io_type_dict(loaded)


def get_clean_type_name(typ: Optional[Type]) -> str:
    if not typ:
        return str(typ)

    try:
        if issubclass(typ, ComplexIOBase):
            return typ.__name__
    except TypeError:
        pass

    origin = get_origin(typ)
    if origin:
        origin_name = origin
        type_args = get_args(typ)
        if origin is Union:
            if len(type_args) == 2 and type_args[1] is type(None):
                # Here we have an Optional special case
                origin_name = "Optional"
                type_args = (type_args[0],)
            else:
                origin_name = "Union"
        elif origin in (list, dict):
            origin_name = origin.__name__.title()
        clean_args = [get_clean_type_name(arg) for arg in type_args]
        if clean_args:
            args_str = ", ".join(clean_args)
            name = f"{origin_name}[{args_str}]"
        else:
            name = origin_name  # e.g. "List" not "list"
    else:
        try:
            name = typ.__name__
        except Exception:
            name = str(typ)

    # Cleanup
    name = name.replace("IOType", "Any")
    return name


def type_is_union(tp: Optional[Type]) -> bool:
    return tp in (Union, UnionType)


def check_type_is_valid(actual: Optional[Type], expected: Optional[Type]) -> bool:
    if actual is None and expected is None:
        return True

    if expected in (IOType, Any):
        return True

    if actual == expected:
        return True

    if not get_origin(expected) and not get_origin(actual):
        # This fails if "expected" is not a class, so just wrap it
        try:
            if actual is not None and issubclass(actual, expected):  # type: ignore
                return True
        except Exception:
            pass

        return actual is expected or expected is IOType or actual is IOType

    # Origin of generic types like List[str] -> list. For types like int, will
    # be None.
    orig_actual = get_origin(actual)
    orig_expected = get_origin(expected)

    # Args of generic types like List[str] -> str.
    params_actual = get_args(actual)
    params_expected = get_args(expected)

    if actual is list:
        return (
            expected is list
            or orig_expected is list
            or (type_is_union(orig_expected) and get_origin(params_expected[0]) is list)
        )
    if actual is dict:
        return (
            expected is dict
            or orig_expected is dict
            or (type_is_union(orig_expected) and get_origin(params_expected[0]) is dict)
        )

    if type_is_union(orig_actual) and type_is_union(orig_expected):
        # This really should be "all" instead of "any", but that would
        # require doing some nonsense with generics. This is good enough for
        # now, other issues can be discovered at runtime.
        return any(
            (
                check_type_is_valid(actual=actual_val, expected=expected_val)
                for actual_val in params_actual
                for expected_val in params_expected
            )
        )
    elif type_is_union(orig_expected) and orig_actual in (None, list, dict):
        # int is valid if we expect Union[int, str]
        return (
            any(
                (
                    check_type_is_valid(actual=actual, expected=expected_val)
                    for expected_val in params_expected
                )
            )
            or params_expected is IOType
        )

    elif type_is_union(orig_actual) and expected and orig_expected is None:
        # There's a very specific case that is correct here, namely
        # Union[Table, StockTable] is allowed to be passed in for Table
        try:
            if issubclass(expected, ComplexIOBase) and all(
                (issubclass(typ, expected) for typ in params_actual)
            ):
                return True
        except Exception:
            pass

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
