import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from gbi_common_py_utils.utils.redis import RedisCache

from agent_service.io_type_utils import IOType, dump_io_type, load_io_type
from agent_service.utils.postgres import get_psql

DEFAULT_CACHE_TTL = 6 * 60 * 60  # six hours

logger = logging.getLogger(__name__)


class CacheBackend(ABC):
    @abstractmethod
    async def get(self, key: str, ttl: Optional[int] = None) -> Optional[IOType]:
        pass

    @abstractmethod
    async def set(self, key: str, val: IOType, ttl: Optional[int] = None) -> None:
        pass


class RedisCacheBackend(CacheBackend):
    def __init__(self, namespace: Any, serialize_func: Any, deserialize_func: Any) -> None:
        # TODO use async redis client
        self.client = RedisCache(
            namespace=namespace, serialize_func=serialize_func, deserialize_func=deserialize_func
        )

    async def get(self, key: str, ttl: Optional[int] = None) -> Optional[IOType]:
        # redis implements ttl on insert, so ignored here
        val = await asyncio.to_thread(self.client.get, key=key)
        return val

    async def set(self, key: str, val: IOType, ttl: Optional[int] = None) -> None:
        await asyncio.to_thread(self.client.set, key=key, val=val, ttl=ttl)


class PostgresCacheBackend(CacheBackend):
    def __init__(self) -> None:
        # TODO use async client
        self.db = get_psql()

    async def get(self, key: str, ttl: Optional[int] = None) -> Optional[IOType]:
        sql = """SELECT * FROM agent.task_cache  WHERE cache_key = %(cache_key)s
        """
        params: Dict[str, Any] = {"cache_key": key}

        if ttl:
            # only consider rows recent enough to satisfy ttl requirements
            params["ttl"] = ttl
            sql += " AND last_updated >= NOW() - %(ttl)s *  interval '1 second'"

        rows = self.db.generic_read(sql, params)
        if not rows:
            return None
        val_str = rows[0]["cache_value"]
        return load_io_type(val_str)

    async def set(self, key: str, val: IOType, ttl: Optional[int] = None) -> None:
        # postgres implements ttl on read, so ignored here
        sql = """
        INSERT INTO agent.task_cache (cache_key, cache_value) VALUES
          (%(key)s, %(val)s)
        ON CONFLICT (cache_key) DO UPDATE SET
          cache_value = %(val)s,
          last_updated = NOW()
        """
        self.db.generic_write(sql, {"key": key, "val": dump_io_type(val)})
