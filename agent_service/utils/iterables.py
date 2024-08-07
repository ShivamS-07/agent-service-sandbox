from itertools import chain, islice
from typing import Any, Callable, Iterable, List, Optional, Sequence, TypeVar

T = TypeVar("T")
R = TypeVar("R")


def flatten(lst: Iterable[Sequence[T]]) -> List[T]:
    """
    Flattens 1 level of an array.
    """
    return list(chain.from_iterable(lst))


def flat_map(f: Callable[[T], Sequence[R]], input: Iterable[T]) -> List[R]:
    return flatten(map(f, input))


def first(seq: Iterable[T]) -> Optional[T]:
    for item in seq:
        return item
    return None


def filter_nones(lst: Iterable[Optional[T]]) -> List[T]:
    return [x for x in lst if x is not None]


def nonnull(x: Optional[T]) -> T:
    if x is None:
        raise ValueError("Expected non null value via `nonnull`.")
    return x


def unique_by(inlist: Iterable[T], key: Callable[[T], Any]) -> List[T]:
    """
    Given a iterable input and a key function, return only the last
    item in the list for each key.
    """
    return list({key(item): item for item in inlist}.values())


def chunk(iterable: Iterable[T], n: int) -> Iterable[List[T]]:
    it = iter(iterable)
    group = list(islice(it, n))
    while group:
        yield group
        group = list(islice(it, n))
