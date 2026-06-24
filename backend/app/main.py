"""
MigrateMate FastAPI Application - Full Backend
"""

import sys
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.core.config import settings
from app.core.connections import check_all_connections, close_all_connections, QdrantConnection, Neo4jConnection, RedisConnection
from app.core.database import init_db
from app.services import initialize_embedding_service, get_qdrant_service, get_neo4j_service
from app.api.v1 import projects_router, migrations_router, ai_router, batch_router

# Windows consoles often default to cp1252, which can't encode the emoji used in
# the startup banner — that raised UnicodeEncodeError and crashed startup. Force
# UTF-8 (best-effort) so the app boots on any console.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):  # not a reconfigurable stream
        pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown.

    The core migration path only needs a Gemini API key. The Postgres-backed
    project store and the RAG stack (embeddings/Qdrant/Neo4j) are optional and
    gated behind settings, so the app starts in ~1s without Docker by default.
    """
    print(f"🚀 Starting {settings.APP_NAME} in {settings.APP_ENV} mode...")

    if settings.ENABLE_DATABASE:
        try:
            from app.models import Project, MigrationJob, CodeChunk  # noqa: F401
            await init_db()
            print("✅ Database tables initialized")
        except Exception as e:
            print(f"⚠️  Database init warning: {e}")

    if settings.ENABLE_VECTOR_SERVICES:
        # Check connections
        health = await check_all_connections()
        for service, status in health.items():
            emoji = "✅" if status else "❌"
            print(f"{emoji} {service.capitalize()}: {'Connected' if status else 'FAILED'}")

        print("🧠 Loading embedding model...")
        try:
            emb = await initialize_embedding_service()
            print(f"✅ Embedding model loaded (dimension: {emb.dimension}, device: {emb.device})")
        except Exception as e:
            print(f"⚠️  Embedding model warning: {e}")

        try:
            qdrant = get_qdrant_service()
            await qdrant.create_collection()
            print("✅ Qdrant collection ready")
        except Exception as e:
            print(f"⚠️  Qdrant warning: {e}")

        try:
            neo4j = get_neo4j_service()
            await neo4j.setup_schema()
            print("✅ Neo4j schema ready")
        except Exception as e:
            print(f"⚠️  Neo4j warning: {e}")
    else:
        print("ℹ️  RAG services disabled (set ENABLE_VECTOR_SERVICES=true to enable).")

    if not settings.GEMINI_API_KEY:
        print("⚠️  No GEMINI_API_KEY set — migrations will use the pattern-based fallback.")

    print(f"🎉 {settings.APP_NAME} is ready!")

    yield

    print(f"👋 Shutting down {settings.APP_NAME}...")
    if settings.ENABLE_DATABASE:
        await close_all_connections()
    if settings.ENABLE_VECTOR_SERVICES:
        try:
            await get_qdrant_service().close()
            await get_neo4j_service().close()
        except Exception:
            pass
    print("✅ All connections closed")


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.APP_NAME,
        description="🚀 MigrateMate - Agentic Code Migration Platform",
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/docs" if settings.DEBUG else None,
        redoc_url="/redoc" if settings.DEBUG else None,
    )
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include routers
    app.include_router(projects_router, prefix="/api/v1/projects", tags=["Projects"])
    app.include_router(migrations_router, prefix="/api/v1/migrations", tags=["Migrations"])
    app.include_router(ai_router, prefix="/api/v1/ai", tags=["AI"])
    app.include_router(batch_router, prefix="/api/v1/batch", tags=["Batch Migration"])
    
    return app


app = create_app()


# ==================== Response Models ====================

class HealthResponse(BaseModel):
    status: str
    environment: str
    services: dict[str, bool]


class MessageResponse(BaseModel):
    message: str
    data: dict[str, Any] | None = None


# ==================== Root Endpoints ====================

@app.get("/", response_model=MessageResponse)
async def root():
    return MessageResponse(
        message=f"Welcome to {settings.APP_NAME}!",
        data={"version": "0.1.0", "docs": "/docs", "health": "/health"}
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    # When the optional infra is disabled, don't try to dial Postgres/Qdrant/
    # Neo4j/Redis — those connection attempts would just time out.
    if not settings.ENABLE_DATABASE and not settings.ENABLE_VECTOR_SERVICES:
        return HealthResponse(status="healthy", environment=settings.APP_ENV, services={})

    services = await check_all_connections()
    return HealthResponse(
        status="healthy" if all(services.values()) else "degraded",
        environment=settings.APP_ENV,
        services=services,
    )


@app.get("/health/postgres")
async def postgres_health():
    from app.core.database import engine
    from sqlalchemy import text
    try:
        async with engine.connect() as conn:
            result = await conn.execute(text("SELECT version()"))
            return {"status": "healthy", "version": result.scalar()}
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.get("/health/qdrant")
async def qdrant_health():
    try:
        client = await QdrantConnection.get_client()
        collections = await client.get_collections()
        return {"status": "healthy", "collections_count": len(collections.collections)}
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.get("/health/neo4j")
async def neo4j_health():
    try:
        async with Neo4jConnection.get_session() as session:
            result = await session.run("CALL dbms.components() YIELD name, versions")
            record = await result.single()
            return {"status": "healthy", "name": record["name"], "version": record["versions"][0] if record["versions"] else "unknown"}
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.get("/health/redis")
async def redis_health():
    try:
        client = await RedisConnection.get_client()
        info = await client.info("server")
        return {"status": "healthy", "version": info.get("redis_version", "unknown")}
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))
