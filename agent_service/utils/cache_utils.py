import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, Optional, Union

from pydantic import BaseModel, Field

from agent_service.io_type_utils import IOType, dump_io_type, load_io_type
from agent_service.io_types.graph import GraphOutput
from agent_service.io_types.table import TableOutput
from agent_service.io_types.text import TextOutput
from agent_service.utils.postgres import get_psql
from agent_service.utils.redis_async import AsyncRedisCache

DEFAULT_CACHE_TTL = 6 * 60 * 60  # six hours

logger = logging.getLogger(__name__)

REDIS_OUTPUT_CACHE_NAMESPACE = "agent-output-cache-new"


class CacheBackend(ABC):
    @abstractmethod
    async def get(self, key: str, ttl: Optional[int] = None) -> Optional[IOType]:
        pass

    @abstractmethod
    async def multiget(
        self, keys: Iterable[str], ttl: Optional[int] = None
    ) -> Optional[Dict[str, IOType]]:
        pass

    @abstractmethod
    async def set(self, key: str, val: IOType, ttl: Optional[int] = None) -> None:
        pass

    @abstractmethod
    async def multiset(self, key_val_map: Dict[str, IOType], ttl: Optional[int] = None) -> None:
        pass


class RedisCacheBackend(CacheBackend):
    def __init__(
        self,
        namespace: Any,
        serialize_func: Any,
        deserialize_func: Any,
        max_connections: Optional[int] = None,
        auto_close_connection: bool = False,
    ) -> None:
        # TODO use async redis client
        self.client = AsyncRedisCache(
            namespace=namespace,
            serialize_func=serialize_func,
            deserialize_func=deserialize_func,
            max_connections=max_connections,
            auto_close_connection=auto_close_connection,
        )

    async def get(self, key: str, ttl: Optional[int] = None) -> Optional[IOType]:
        # redis implements ttl on insert, so ignored here
        val = await self.client.get(key=key)
        return val

    async def multiget(
        self, keys: Iterable[str], ttl: Optional[int] = None
    ) -> Optional[Dict[str, IOType]]:
        return await self.client.multiget(keys=keys)

    async def set(self, key: str, val: IOType, ttl: Optional[int] = None) -> None:
        await self.client.set(key=key, val=val, ttl=ttl)

    async def multiset(self, key_val_map: Dict[str, IOType], ttl: Optional[int] = None) -> None:
        await self.client.multiset(key_val_map=key_val_map, ttl=ttl)

    async def invalidate(self, key: str) -> None:
        await self.client.delete(key)


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

    async def multiget(
        self, keys: Iterable[str], ttl: Optional[int] = None
    ) -> Optional[Dict[str, IOType]]:
        raise NotImplementedError("multiget not implemented for PostgresCacheBackend")

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

    async def multiset(self, key_val_map: Dict[str, IOType], ttl: Optional[int] = None) -> None:
        raise NotImplementedError("multiset not implemented for PostgresCacheBackend")


class OutputWrapper(BaseModel):
    output: Union[TextOutput, GraphOutput, TableOutput] = Field(discriminator="output_type")

    @classmethod
    def serialize_func(cls, raw_output: Any) -> Any:
        to_write = cls(output=raw_output)
        res = to_write.model_dump_json(serialize_as_any=True)
        return res

    @classmethod
    def deserialize_func(cls, raw_input: Any) -> Any:
        agent_output = cls.model_validate_json(raw_input)
        return agent_output.output


def get_redis_cache_backend_for_output(auto_close_connection: bool = False) -> RedisCacheBackend:
    return RedisCacheBackend(
        namespace=REDIS_OUTPUT_CACHE_NAMESPACE,
        serialize_func=OutputWrapper.serialize_func,
        deserialize_func=OutputWrapper.deserialize_func,
        auto_close_connection=auto_close_connection,
    )
