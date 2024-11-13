import json
import logging
from typing import Any, Callable, Optional

import redis.asyncio as redis
from gbi_common_py_utils.utils.redis import (
    _REDIS_CONNECT_TIMEOUT,
    _REDIS_HOST,
    _REDIS_OPERATION_TIMEOUT,
    _REDIS_PORT,
    RedisInvalidationKey,
    _to_string,
    is_redis_available,
)

logger = logging.getLogger(__name__)


_REDIS_ASYNC = None


def get_redis_async_client(max_connections: Optional[int] = None) -> redis.Redis:
    """
    Returns a global Redis instance if one is available, otherwise returns None.
    """
    global _REDIS_ASYNC

    # is_redis_available checks these, but needed for mypy
    if _REDIS_PORT and _REDIS_HOST and is_redis_available():
        logger.info(f"Initializing ASYNC redis connection: {_REDIS_HOST}:{_REDIS_PORT}")
        _REDIS_ASYNC = redis.Redis(
            host=_REDIS_HOST,
            port=int(_REDIS_PORT),
            decode_responses=False,
            socket_timeout=_REDIS_OPERATION_TIMEOUT,
            socket_connect_timeout=_REDIS_CONNECT_TIMEOUT,
            max_connections=max_connections,
        )
    else:
        raise RuntimeError("No Redis client available")

    return _REDIS_ASYNC


class AsyncRedisCache:
    """
    Wrapper around a Redis client instance. Handles errors without crashing, as
    well as serialization/deserialization and automatic namespacing.
    NOTE: If not Redis client is able to be initialized, will raise a RuntimeError.
    """

    def _make_redis_key(self, key: str, key_prefix: str = "") -> str:
        parts = []
        if self.namespace:
            parts.append(self.namespace)
        if key_prefix:
            parts.append(key_prefix)
        parts.append(key)
        return ":".join(parts)

    @staticmethod
    def make_prefix_key(args: Any) -> str:
        if not args:
            return ""
        if isinstance(args, tuple):
            return ":".join((_to_string(a) for a in args))
        else:
            return _to_string(args)

    def __init__(
        self,
        client: Optional[redis.Redis] = None,
        namespace: Optional[str] = None,
        serialize_func: Callable = json.dumps,
        deserialize_func: Callable = json.loads,
        max_connections: Optional[int] = None,
        auto_close_connection: bool = False,
    ) -> None:
        self.client = client or get_redis_async_client(max_connections=max_connections)
        self.namespace = namespace
        self.serialize_func = serialize_func
        self.deserialize_func = deserialize_func
        self.auto_close_connection = auto_close_connection

    async def get(self, key: str, key_prefix: str = "") -> Optional[Any]:
        key = self._make_redis_key(key, key_prefix)
        # first, try to get the value from Redis
        try:
            val = await self.client.get(key)
        except redis.RedisError as e:
            logger.error("Encountered redis error in RedisCache get operation")
            logger.exception(e)
            return None
        finally:
            if self.auto_close_connection:
                await self.client.close()

        # if the value isn't present, return None
        if val is None:
            return None

        # If the value is present, try to deserialize.
        try:
            return self.deserialize_func(val)
        except Exception as e:
            logger.error(f"Encountered exception while deserializing: {val}")
            logger.exception(e)

        return None

    async def set(
        self, key: str, val: Any, ttl: Optional[int] = None, key_prefix: str = ""
    ) -> None:
        key = self._make_redis_key(key, key_prefix)
        # Try to serialize the value
        try:
            val = self.serialize_func(val)
        except Exception as e:
            logger.error(f"Encountered exception while serializing: {val}")
            logger.exception(e)
            return

        # Set the value in redis
        try:
            await self.client.set(name=key, value=val, ex=ttl)
        except redis.RedisError as e:
            logger.error("Encountered redis error in RedisCache set operation")
            logger.exception(e)
        finally:
            if self.auto_close_connection:
                await self.client.close()

    async def invalidate(self, key: RedisInvalidationKey, scan_count: int = 100) -> None:
        """
        Invalidate the keys represented by the given RedisInvalidationKey.
        scan_count - Refers to the number of keys scanned at once. A higher
        number will delete the keys more quickly, but is at risk of blocking
        other redis operations.
        """
        invalidation_key = key.to_key_str()
        # Illegal characters will cause invalidation to not work, so error if they are present
        for char in ("*", "[", "]"):
            if char in invalidation_key:
                raise ValueError(f"Invalid char '{char}' in key '{invalidation_key}'")
        key_pattern = self._make_redis_key("*", key_prefix=invalidation_key)

        try:
            keys = [key async for key in self.client.scan_iter(key_pattern, count=scan_count)]
            await self.client.delete(*keys)
        except redis.RedisError as e:
            logger.error("Encountered redis error in RedisCache delete operation")
            logger.exception(e)
        finally:
            if self.auto_close_connection:
                await self.client.close()

    # Retrieves val of key from redis and resets TTL of the key
    async def getex(self, key: str, ttl: int, key_prefix: str = "") -> None:
        key = self._make_redis_key(key, key_prefix)
        # first, try to get the value from Redis
        try:
            val = self.client.getex(name=key, ex=ttl)
        except redis.RedisError as e:
            logger.error("Encountered redis error in RedisCache getex operation")
            logger.exception(e)
            return None
        finally:
            if self.auto_close_connection:
                await self.client.close()

        # If the value isn't present, return None
        if val is None:
            return None

        # If the value is present, try to deserialize.
        try:
            return self.deserialize_func(val)
        except Exception as e:
            logger.error(f"Encountered exception while deserializing: {val}")
            logger.exception(e)

        return None

    # Increments the value at key in redis
    async def incr(self, key: str, key_prefix: str = "", ttl: int = 60 * 60 * 24) -> None:
        key = self._make_redis_key(key, key_prefix)
        try:
            await self.client.incr(name=key)
            await self.client.expire(name=key, time=ttl)
        except redis.RedisError as e:
            logger.error("Encountered redis error in RedisCache incr operation")
            logger.exception(e)
        finally:
            if self.auto_close_connection:
                await self.client.close()

    async def delete(self, key: str, key_prefix: str = "") -> None:
        key = self._make_redis_key(key, key_prefix)
        try:
            await self.client.delete(key)
        except redis.RedisError as e:
            logger.error("Encountered redis error in RedisCache delete operation")
            logger.exception(e)
        finally:
            if self.auto_close_connection:
                await self.client.close()
