# TODO
# Once we confirm this is working for agent service, we can move this into gbi common

import asyncio
import json
import logging
import traceback
from contextlib import asynccontextmanager
from datetime import timedelta
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple, cast

import backoff
import psycopg
import psycopg_pool
from gbi_common_py_utils.config import get_config
from psycopg.rows import dict_row
from psycopg_pool.pool_async import AsyncConnectionPool

from agent_service.utils.boosted_pg import BoostedPG, CursorType, InsertToTableArgs

logger = logging.getLogger(__name__)


def gen_delete_sql(tablename: str, **kwargs: Any) -> Tuple[str, List[Any]]:
    clauses = []
    params = []
    for key, value in kwargs.items():
        if value is None:
            continue
        if isinstance(value, list):
            # If the value is a list, query the individual list elements instead of the array
            clauses.append(f"{key} = ANY(%s)")
        else:
            clauses.append(f"{key} = %s")
        params.append(value)
    where_str = " AND ".join(clauses)
    return f"DELETE FROM {tablename} WHERE ({where_str})", params


class AsyncPostgresBase(BoostedPG):
    def __init__(
        self,
        environment: Optional[str] = None,
        min_pool_size: int = 1,
        max_pool_size: int = 8,
        connection_timeout: float = 60.0,
        max_idle_time: float = 600.0,
        read_only: bool = False,
    ):
        """
        min/max_pool_size: int
            The minimum and maximum number of connections in the connection
            pool. It is recommended to have a max size > 1 so that the async
            queries do not block each other.
        """
        self._environment = environment  # TODO use this
        self.min_pool_size = min_pool_size
        self.max_pool_size = max_pool_size
        self.connection_timeout = connection_timeout
        self.max_idle_time = max_idle_time
        self._pool: Optional[AsyncConnectionPool] = None
        self.read_only = read_only

    async def pool(self) -> AsyncConnectionPool:
        if not self._pool:
            await self.connect()
        return self._pool  # type: ignore

    @backoff.on_exception(
        backoff.expo,
        exception=(ValueError, psycopg.OperationalError, psycopg.InternalError),
        max_time=int(timedelta(seconds=120).total_seconds()),
        max_tries=10,
        logger=logger,
    )
    async def connect(self) -> None:
        if self._pool is not None:
            return
        try:
            db_config = get_config().app_db
            host = db_config.host if not self.read_only else db_config.read_only_host
            if not host:
                host = db_config.host
            connection_args = {
                "dbname": db_config.database,
                "user": db_config.username,
                "password": db_config.password,
                "host": db_config.host,
                "port": db_config.port,
                "row_factory": dict_row,
            }

            pool = psycopg_pool.AsyncConnectionPool(
                open=False,
                min_size=self.min_pool_size,
                max_size=self.max_pool_size,
                check=psycopg_pool.AsyncConnectionPool.check_connection,
                kwargs=connection_args,
                timeout=self.connection_timeout,
                max_idle=self.max_idle_time,
            )
            await pool.open()
            await pool.wait()
            self._pool = pool
        except Exception:
            raise ValueError(f"Unable to connect to Postgres DB: {traceback.format_exc()}")

    async def close(self) -> None:
        if self._pool:
            await self._pool.close()

    async def __aexit__(self, *excinfo: Any) -> None:
        await self.close()

    def __del__(self) -> None:
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(self.close())
            else:
                loop.run_until_complete(self.close())
        except Exception:
            pass

    # SQL METHODS
    async def generic_read(self, sql: str, params: Optional[Any] = None) -> List[Dict[str, Any]]:
        async with (await self.pool()).connection() as conn:
            async with conn.cursor() as cursor:
                await cursor.execute(sql, params)
                fetchall = await cursor.fetchall()
        return fetchall  # type: ignore

    async def generic_write(self, sql: str, params: Optional[Any] = None) -> None:
        async with (await self.pool()).connection() as conn:
            async with conn.cursor() as cursor:
                await cursor.execute(sql, params)

    @asynccontextmanager
    async def cursor(self) -> AsyncIterator[psycopg.AsyncCursor]:
        async with (await self.pool()).connection() as conn:
            async with conn.cursor() as cursor:
                yield cursor

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[psycopg.AsyncCursor]:
        async with self.cursor() as cursor:
            await cursor.execute("BEGIN")
            try:
                yield cursor
                await cursor.execute("COMMIT")
            except Exception as e:
                await cursor.execute("ROLLBACK")
                raise e

    async def delete_from_table_where(self, table_name: str, **kwargs: Any) -> None:
        """
        Delete rows from a table that match a condition
        Args:
            table_name: The name of the table in which to delete rows.
            kwargs: The values that determine which rows to delete

        Example: delete_from_table_where("users", username="tommy", email="tommy@t.com")
        would delete all rows matching the kwargs condition. Essentially, this will run:
        DELETE FROM users where username = 'tommy' and email = 'tommy@t.com'

        Note: this function assumes that all kwargs provided are valid column names. It uses
        prepared statements so the values for each kwarg need to be able to be interpreted
        by psycopg as prepared statement params.

        Returns: Nothing.

        """

        await self.generic_write(*gen_delete_sql(table_name, **kwargs))

    async def generic_update(self, table_name: str, where: Dict, values_to_update: Dict) -> None:
        """
        Update all rows matching 'where' with 'values_to_update'
        Args:
            table_name:
            where:
            values_to_update:

        This is best explained with an example.

        generic_update("users",
        where={"id": "tommy"},
        values_to_update={"email": "tommy2@boosted.ai", "favorite_color": "green"}
        )

        would get translated into the following SQL and executed:
        UPDATE users SET (email, favorite_color) = ('tommy2@boosted.ai', 'green') WHERE id = 'tommy'

        Returns:
            Nothing

        """
        to_update_keys = values_to_update.keys()
        to_update_key_string = ",".join(to_update_keys)
        to_update_placeholders = ",".join(["%s" for _ in range(len(to_update_keys))])
        if len(to_update_keys) > 1:
            to_update_key_string = "(" + to_update_key_string + ")"
            to_update_placeholders = "(" + to_update_placeholders + ")"
        clauses = []
        params = []
        for key in to_update_keys:
            params.append(values_to_update[key])
        for key, value in where.items():
            if value is None:
                continue
            clauses.append(f"{key} = %s")
            params.append(value)
        where_str = " AND ".join(clauses)

        await self.generic_write(
            f"UPDATE {table_name} SET {to_update_key_string} = {to_update_placeholders} WHERE ({where_str})",  # noqa
            params,
        )

    @staticmethod
    def _gen_multi_row_insert(
        table_name: str, values_to_insert: List[Dict[str, Any]], ignore_conficts: bool
    ) -> Tuple[str, list]:
        """
        Helper function for multi_row_insert() which generates the
        SQL statement and the parameters to write into SQL.
        :param table_name: The name of the table as a string.
        :param values_to_insert: A list of dictionaries with the values to insert into the table.
        :param ignore_conficts: If True, then do nothing on conflicts with existing rows.
        :return: A tuple containing the SQL statement and the argument parameters
        """
        non_null_keys = []
        params = []
        keys_done = False
        values_strs = []
        for row in values_to_insert:
            non_null_values = []
            for key, value in row.items():
                if value is not None:
                    if not keys_done:
                        non_null_keys.append(key)
                    non_null_values.append("%s")
                    params.append(value)
            keys_done = True
            values_strs.append(f'({",".join(non_null_values)})')
        keys_to_insert_str = ",".join(non_null_keys)
        prefix = f"INSERT INTO {table_name} ({keys_to_insert_str}) VALUES {','.join(values_strs)} "
        suffix = "ON CONFLICT DO NOTHING" if ignore_conficts else ""
        sql = prefix + suffix
        return (
            sql,
            params,
        )

    async def multi_row_insert(
        self, table_name: str, rows: List[Dict[str, Any]], ignore_conflicts: bool = False
    ) -> None:
        """
        Insert many rows into a table
        Args:
            table_name: The name of the table in which to insert.
            rows: List of dicts, where each dict contains keys that are column names
            and values to insert
            ignore_conflicts: If True, will do nothing on conflicts with existing rows.

        Note: this function assumes that all dict keys provided are valid column names. It also
        assumes that each dict will have the same set of keys.

        It uses prepared statements so the values for each kwarg need to be able to be
        interpreted by psycopg as prepared statement params.

        Returns: Nothing.

        """

        if len(rows) == 0:
            return
        await self.generic_write(*self._gen_multi_row_insert(table_name, rows, ignore_conflicts))

    async def insert_atomic(self, to_insert: List[InsertToTableArgs]) -> None:
        async with (await self.pool()).connection() as conn:
            async with conn.cursor() as cursor:
                for arg in to_insert:
                    sql, params = self._gen_multi_row_insert(
                        table_name=arg.table_name, values_to_insert=arg.rows, ignore_conficts=False
                    )
                    await cursor.execute(sql, params)

    async def generic_jsonb_update(
        self,
        table_name: str,
        jsonb_column: str,
        field_updates: Dict[str, Any],
        where: Dict[str, Any],
        cursor: Optional[CursorType] = None,
    ) -> None:
        """
        Update specific fields within a JSONB column for rows matching the where condition
        Args:
            table_name: Name of the table to update
            jsonb_column: Name of the JSONB column to update
            field_updates: Dictionary mapping paths to their new values.
                         Supports both dot notation and array indexing:
                         - Dot notation: "settings.theme"
                         - Array indexing: "items[0].name"
            where: Dictionary of column-value pairs to identify rows to update

        Example:
            # Update array element
            generic_jsonb_update(
                table_name="users",
                jsonb_column="metadata",
                field_updates={
                    "items[0].name": "new_name",
                    "settings.theme": "dark"
                },
                where={"id": "ted"}
            )

            Would translate to:
            UPDATE users
            SET metadata = jsonb_set(
                jsonb_set(
                    CAST(metadata AS jsonb),
                    '{items,0,name}',
                    '"new_name"',
                    true
                ),
                '{settings,theme}',
                '"dark"',
                true
            )
            WHERE (id = 'ted')
        """
        if not field_updates:
            return

        # Build the nested jsonb_set calls
        jsonb_sets = f"CAST({jsonb_column} AS jsonb)"
        params = []
        for path, value in field_updates.items():
            # Convert path to PostgreSQL JSONB path format
            # Handle array indexing: convert "items[0].name" to "{items,0,name}"
            pg_path = path.replace("[", ",").replace("]", "")
            pg_path = "{" + pg_path.replace(".", ",") + "}"

            jsonb_sets = f"jsonb_set({jsonb_sets}, %s, %s, true)"
            json_value = json.dumps(value)
            params.extend([pg_path, json_value])

        # Build WHERE clause
        where_clauses = []
        for key, value in where.items():
            if value is None:
                continue
            where_clauses.append(f"{key} = %s")
            params.append(value)
        where_str = " AND ".join(where_clauses)

        # Create SQL using nested jsonb_set calls
        sql = f"UPDATE {table_name} SET {jsonb_column} = {jsonb_sets} WHERE ({where_str})"

        if cursor:
            await cast(psycopg.AsyncCursor, cursor).execute(sql, params)
        else:
            await self.generic_write(sql, params)
