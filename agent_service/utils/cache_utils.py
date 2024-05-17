import logging
from abc import ABC, abstractmethod
from typing import Optional

from gbi_common_py_utils.utils.redis import RedisCache

from agent_service.io_type_utils import IOType, dump_io_type, load_io_type
from agent_service.utils.postgres import get_psql

DEFAULT_CACHE_TTL = 6 * 60 * 60  # six hours

logger = logging.getLogger(__name__)


class CacheBackend(ABC):
    @abstractmethod
    async def get(self, key: str) -> Optional[IOType]:
        pass

    @abstractmethod
    async def set(self, key: str, val: IOType, ttl: Optional[int] = None) -> None:
        pass


class RedisCacheBackend(CacheBackend):
    def __init__(self) -> None:
        # TODO use async redis client
        self.client = RedisCache(
            namespace="agent-tool-cache", serialize_func=dump_io_type, deserialize_func=load_io_type
        )

    async def get(self, key: str) -> Optional[IOType]:
        return self.client.get(key=key)

    async def set(self, key: str, val: IOType, ttl: Optional[int] = None) -> None:
        self.client.set(key=key, val=val, ttl=ttl)


class PostgresCacheBackend(CacheBackend):
    def __init__(self) -> None:
        # TODO use async client
        self.db = get_psql()

    async def get(self, key: str) -> Optional[IOType]:
        rows = self.db.select_where(table_name="agent.task_cache", where={"cache_key": key})
        if not rows:
            return None
        val_str = rows[0]["cache_value"]
        return load_io_type(val_str)

    async def set(self, key: str, val: IOType, ttl: Optional[int] = None) -> None:
        sql = """
        INSERT INTO agent.task_cache (cache_key, cache_value) VALUES
          (%(key)s, %(val)s)
        ON CONFLICT (cache_key) DO UPDATE SET
          cache_value = %(val)s,
          last_updated = NOW()
        """
        self.db.generic_write(sql, {"key": key, "val": dump_io_type(val)})