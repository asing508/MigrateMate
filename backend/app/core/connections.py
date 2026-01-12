"""External service connections."""

from contextlib import asynccontextmanager
from typing import AsyncGenerator
import redis.asyncio as redis
from neo4j import AsyncGraphDatabase, AsyncDriver
from qdrant_client import AsyncQdrantClient
from app.core.config import settings


class QdrantConnection:
    _client: AsyncQdrantClient | None = None
    
    @classmethod
    async def get_client(cls) -> AsyncQdrantClient:
        if cls._client is None:
            cls._client = AsyncQdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_HTTP_PORT, prefer_grpc=False)
        return cls._client
    
    @classmethod
    async def close(cls) -> None:
        if cls._client is not None:
            await cls._client.close()
            cls._client = None
    
    @classmethod
    async def health_check(cls) -> bool:
        try:
            client = await cls.get_client()
            await client.get_collections()
            return True
        except Exception:
            return False


class Neo4jConnection:
    _driver: AsyncDriver | None = None
    
    @classmethod
    async def get_driver(cls) -> AsyncDriver:
        if cls._driver is None:
            cls._driver = AsyncGraphDatabase.driver(settings.NEO4J_URI, auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD))
        return cls._driver
    
    @classmethod
    async def close(cls) -> None:
        if cls._driver is not None:
            await cls._driver.close()
            cls._driver = None
    
    @classmethod
    async def health_check(cls) -> bool:
        try:
            driver = await cls.get_driver()
            async with driver.session() as session:
                result = await session.run("RETURN 1 AS num")
                record = await result.single()
                return record["num"] == 1
        except Exception:
            return False
    
    @classmethod
    @asynccontextmanager
    async def get_session(cls) -> AsyncGenerator:
        driver = await cls.get_driver()
        session = driver.session()
        try:
            yield session
        finally:
            await session.close()


class RedisConnection:
    _client: redis.Redis | None = None
    
    @classmethod
    async def get_client(cls) -> redis.Redis:
        if cls._client is None:
            cls._client = redis.Redis.from_url(settings.REDIS_URL, encoding="utf-8", decode_responses=True)
        return cls._client
    
    @classmethod
    async def close(cls) -> None:
        if cls._client is not None:
            await cls._client.close()
            cls._client = None
    
    @classmethod
    async def health_check(cls) -> bool:
        try:
            client = await cls.get_client()
            return await client.ping()
        except Exception:
            return False


async def check_all_connections() -> dict[str, bool]:
    from app.core.database import engine
    from sqlalchemy import text
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
            postgres_ok = True
    except Exception:
        postgres_ok = False
    
    return {
        "postgres": postgres_ok,
        "qdrant": await QdrantConnection.health_check(),
        "neo4j": await Neo4jConnection.health_check(),
        "redis": await RedisConnection.health_check(),
    }


async def close_all_connections() -> None:
    from app.core.database import close_db
    await close_db()
    await QdrantConnection.close()
    await Neo4jConnection.close()
    await RedisConnection.close()
