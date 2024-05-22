from typing import Any, Dict, List, Optional

from gbi_common_py_utils.utils.postgres import PostgresBase

from agent_service.utils.boosted_pg import BoostedPG, InsertToTableArgs


class SkipCommitBoostedPG(BoostedPG):
    def __init__(self):
        self.pg = PostgresBase(environment="DEV", skip_commit=True)

    async def generic_read(self, sql: str, params: Optional[Any] = None) -> List[Dict[str, Any]]:
        return self.pg.generic_read(sql, params)

    async def generic_write(self, sql: str, params: Optional[Any] = None) -> None:
        self.pg.generic_write(sql, params)

    async def delete_from_table_where(self, table_name, **kwargs):
        self.pg.delete_from_table_where(table_name, **kwargs)

    async def generic_update(self, table_name, where: Dict, values_to_update: Dict):
        self.pg.generic_update(table_name, where, values_to_update)

    async def multi_row_insert(
        self, table_name, rows: List[Dict[str, Any]], ignore_conflicts: bool = False
    ) -> None:
        self.pg.multi_row_insert(table_name, rows, ignore_conflicts)

    async def insert_atomic(self, to_insert: List[InsertToTableArgs]) -> None:
        # This is not actually atomic, since these inserts might happen in different transactions.
        # However, this class is only for testing purposes, so this is completely fine.
        for arg in to_insert:
            self.pg.multi_row_insert(table_name=arg.table_name, rows=arg.rows)
