from functools import lru_cache

import redis
import redis.asyncio as aredis

from backend.config import REDIS_URL


@lru_cache
def get_async_redis() -> aredis.Redis:
    return aredis.from_url(REDIS_URL, decode_responses=True)


@lru_cache
def get_sync_redis() -> redis.Redis:
    # Bytes client: the embedding cache stores raw serialized vectors
    return redis.from_url(REDIS_URL)
