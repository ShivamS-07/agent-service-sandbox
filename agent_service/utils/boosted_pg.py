from abc import ABC, abstractmethod
from typing import Any, Dict, List, NamedTuple, Optional, Union

from psycopg import AsyncCursor, Cursor


class InsertToTableArgs(NamedTuple):
    table_name: str
    rows: List[Dict[str, Any]]


CursorType = Union[Cursor, AsyncCursor]


class BoostedPG(ABC):
    @abstractmethod
    async def generic_read(self, sql: str, params: Optional[Any] = None) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    async def generic_write(self, sql: str, params: Optional[Any] = None) -> None:
        pass

    @abstractmethod
    async def delete_from_table_where(self, table_name: str, **kwargs: Any) -> None:
        pass

    @abstractmethod
    async def generic_update(self, table_name: str, where: Dict, values_to_update: Dict) -> None:
        pass

    @abstractmethod
    async def multi_row_insert(
        self,
        table_name: str,
        rows: List[Dict[str, Any]],
        ignore_conflicts: bool = False,
        conflict_suffix: str = "",
        insert_nulls: bool = False,
    ) -> None:
        pass

    @abstractmethod
    async def insert_atomic(
        self,
        to_insert: List[InsertToTableArgs],
        insert_nulls: bool = False,
    ) -> None:
        pass

    @abstractmethod
    async def generic_jsonb_update(
        self,
        table_name: str,
        jsonb_column: str,
        field_updates: Dict[str, Any],
        where: Dict[str, Any],
        cursor: Optional[CursorType] = None,
    ) -> None:
        pass
