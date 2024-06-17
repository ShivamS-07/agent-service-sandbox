import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

import redis.asyncio as redis
from gbi_common_py_utils.utils.environment import get_environment_tag
from redis.asyncio.client import PubSub

# TODO eventually move async redis into gbi common


_REDIS_QUEUE_HOST = os.getenv("REDIS_QUEUE_HOST")
_REDIS_PORT = os.getenv("REDIS_PORT", "6379")

_REDIS_OPERATION_TIMEOUT = 1.0  # 1s
_REDIS_CONNECT_TIMEOUT = 5.0  # 5s

_REDIS_QUEUE = None

logger = logging.getLogger(__name__)


def is_redis_queue_available() -> bool:
    if _REDIS_QUEUE_HOST and _REDIS_PORT:
        return True
    return False


def get_redis_queue_client() -> Optional[redis.Redis]:
    """
    Returns a global Redis instance if one is available, otherwise returns None.
    """
    global _REDIS_QUEUE
    # is_redis_available checks these, but needed for mypy
    if _REDIS_QUEUE is None and _REDIS_PORT and _REDIS_QUEUE_HOST and is_redis_queue_available():
        logger.info(f"Initializing redis connection: {_REDIS_QUEUE_HOST}:{_REDIS_PORT}")
        _REDIS_QUEUE = redis.Redis(
            host=_REDIS_QUEUE_HOST,
            port=int(_REDIS_PORT),
            decode_responses=False,
            socket_timeout=_REDIS_OPERATION_TIMEOUT,
            socket_connect_timeout=_REDIS_CONNECT_TIMEOUT,
        )

    return _REDIS_QUEUE


CHANNEL_TEMPLATE = "agent-svc:events:{env}:{agent_id}"
NOTIFICATION_CHANNEL_TEMPLATE = "agent-svc:notifications:{env}:{user_id}"


@asynccontextmanager
async def get_agent_event_channel(agent_id: str) -> AsyncGenerator[PubSub, None]:
    env = get_environment_tag().lower()
    redis = get_redis_queue_client()
    if not redis:
        raise RuntimeError("Redis instance not found! Please specify REDIS_QUEUE_HOST")
    async with redis.pubsub() as pubsub:
        await pubsub.subscribe(CHANNEL_TEMPLATE.format(env=env, agent_id=agent_id))
        yield pubsub


@asynccontextmanager
async def get_notification_event_channel(user_id: str) -> AsyncGenerator[PubSub, None]:
    env = get_environment_tag().lower()
    redis = get_redis_queue_client()
    if not redis:
        raise RuntimeError("Redis instance not found! Please specify REDIS_QUEUE_HOST")
    async with redis.pubsub() as pubsub:
        await pubsub.subscribe(NOTIFICATION_CHANNEL_TEMPLATE.format(env=env, user_id=user_id))
        yield pubsub


async def wait_for_messages(channel: PubSub) -> AsyncGenerator[str, None]:
    while True:
        message = await channel.get_message(ignore_subscribe_messages=True)
        if message is None:
            continue
        data = message["data"].decode()
        yield data


async def publish_agent_event(agent_id: str, serialized_event: str) -> None:
    redis = get_redis_queue_client()
    if not redis:
        logger.warning("Skipping message publish on local")
        return

    env = get_environment_tag().lower()
    channel = CHANNEL_TEMPLATE.format(env=env, agent_id=agent_id)
    await redis.publish(channel, serialized_event)


async def publish_notification_event(user_id: str, serialized_event: str) -> None:
    redis = get_redis_queue_client()
    if not redis:
        logger.warning("Skipping message publish on local")
        return

    env = get_environment_tag().lower()
    channel = NOTIFICATION_CHANNEL_TEMPLATE.format(env=env, user_id=user_id)
    await redis.publish(channel, serialized_event)
