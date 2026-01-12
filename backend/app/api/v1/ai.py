"""AI/ML API endpoints for embedding, search, and code analysis."""

from typing import List, Optional
from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel, Field

from app.core.config import settings
from app.services import (
    get_embedding_service,
    get_qdrant_service,
    get_neo4j_service,
    get_hybrid_retrieval_service,
    parse_python_file,
    detect_flask_routes,
)
from app.agents import get_migration_agent

router = APIRouter()


# ==================== Schemas ====================

class EmbedRequest(BaseModel):
    text: str = Field(..., description="Text to embed")
    preprocess: bool = Field(default=True)
    language: str = Field(default="python")


class EmbedResponse(BaseModel):
    embedding: List[float]
    dimension: int
    model: str


class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query")
    project_id: Optional[int] = None
    chunk_types: Optional[List[str]] = None
    limit: int = Field(default=10, ge=1, le=100)


class SearchResult(BaseModel):
    id: str
    score: float
    name: Optional[str]
    file_path: Optional[str]
    chunk_type: Optional[str]
    content_preview: Optional[str]


class SearchResponse(BaseModel):
    results: List[SearchResult]
    total: int
    query: str


class HybridSearchRequest(BaseModel):
    query: str
    project_id: int
    max_results: int = Field(default=10, ge=1, le=50)
    include_dependencies: bool = True


class HybridSearchResponse(BaseModel):
    results: List[dict]
    prompt_context: str
    total_tokens: int


class ParseCodeRequest(BaseModel):
    content: str
    file_path: str = "unknown.py"


class ParsedChunkResponse(BaseModel):
    name: str
    chunk_type: str
    start_line: int
    end_line: int
    dependencies: List[str]
    content_preview: str
    content: str


class MigrateChunkRequest(BaseModel):
    source_framework: str = "flask"
    target_framework: str = "fastapi"
    chunk_content: str
    chunk_name: str = "unknown"
    chunk_type: str = "function"
    context: str = ""
    max_iterations: int = Field(default=3, ge=1, le=5)


class MigrateChunkResponse(BaseModel):
    migrated_code: Optional[str]
    confidence_score: float
    status: str
    errors: List[str]
    iterations: int


# ==================== Endpoints ====================

@router.post("/embed", response_model=EmbedResponse, tags=["Embedding"])
async def embed_text(request: EmbedRequest):
    """Generate embedding for text using local model."""
    try:
        service = get_embedding_service()
        text = request.text
        if request.preprocess:
            text = service.preprocess_code(text, request.language)
        embedding = await service.embed_text(text)
        return EmbedResponse(embedding=embedding, dimension=len(embedding), model=settings.EMBEDDING_MODEL)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/embed/batch", tags=["Embedding"])
async def embed_batch(texts: List[str]):
    """Generate embeddings for multiple texts."""
    try:
        service = get_embedding_service()
        embeddings = await service.embed_batch(texts)
        return {"embeddings": embeddings, "count": len(embeddings), "dimension": len(embeddings[0]) if embeddings else 0}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/embedding/info", tags=["Embedding"])
async def embedding_info():
    """Get embedding model information."""
    try:
        service = get_embedding_service()
        return {"model": settings.EMBEDDING_MODEL, "dimension": service.dimension, "device": service.device}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/search", response_model=SearchResponse, tags=["Search"])
async def search_code(request: SearchRequest):
    """Semantic search for code chunks."""
    try:
        qdrant = get_qdrant_service()
        results = await qdrant.search_by_text(
            query_text=request.query,
            project_id=request.project_id,
            chunk_types=request.chunk_types,
            limit=request.limit,
        )
        return SearchResponse(
            results=[
                SearchResult(
                    id=r["id"],
                    score=r["score"],
                    name=r["payload"].get("name"),
                    file_path=r["payload"].get("file_path"),
                    chunk_type=r["payload"].get("chunk_type"),
                    content_preview=r["payload"].get("content", "")[:200],
                )
                for r in results
            ],
            total=len(results),
            query=request.query,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/search/hybrid", response_model=HybridSearchResponse, tags=["Search"])
async def hybrid_search(request: HybridSearchRequest):
    """Hybrid search combining vector similarity and graph relationships."""
    try:
        service = get_hybrid_retrieval_service()
        context = await service.retrieve(
            query=request.query,
            project_id=request.project_id,
            max_results=request.max_results,
            include_deps=request.include_dependencies,
        )
        return HybridSearchResponse(
            results=[
                {
                    "chunk_id": r.chunk_id,
                    "name": r.name,
                    "file_path": r.file_path,
                    "chunk_type": r.chunk_type,
                    "combined_score": r.combined_score,
                    "dependencies": r.dependencies,
                    "dependents": r.dependents,
                }
                for r in context.results
            ],
            prompt_context=context.to_prompt_context(),
            total_tokens=context.total_tokens,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/parse", response_model=List[ParsedChunkResponse], tags=["Analysis"])
async def parse_code(request: ParseCodeRequest):
    """Parse Python code into chunks with dependencies."""
    try:
        chunks = parse_python_file(request.content, request.file_path)
        return [
            ParsedChunkResponse(
                name=c.name,
                chunk_type=c.chunk_type,
                start_line=c.start_line,
                end_line=c.end_line,
                dependencies=c.dependencies,
                content_preview=c.content[:200],
                content=c.content,
            )
            for c in chunks
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze/routes", tags=["Analysis"])
async def analyze_routes(content: str, framework: str = "flask"):
    """Detect API routes in code."""
    try:
        if framework == "flask":
            routes = detect_flask_routes(content)
        else:
            from app.services import detect_fastapi_routes
            routes = detect_fastapi_routes(content)
        return {"routes": routes, "count": len(routes), "framework": framework}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/migrate/chunk", response_model=MigrateChunkResponse, tags=["Migration"])
async def migrate_chunk(request: MigrateChunkRequest):
    """Migrate a single code chunk using LangGraph agent."""
    try:
        agent = get_migration_agent()
        result = await agent.migrate_chunk(
            project_id=0,
            job_id=0,
            source_framework=request.source_framework,
            target_framework=request.target_framework,
            chunk_id=0,
            chunk_content=request.chunk_content,
            chunk_name=request.chunk_name,
            chunk_type=request.chunk_type,
            retrieval_context=request.context,
            max_iterations=request.max_iterations,
        )
        return MigrateChunkResponse(
            migrated_code=result.get("migrated_code"),
            confidence_score=result.get("confidence_score", 0.0),
            status=result.get("status", "unknown"),
            errors=result.get("errors", []),
            iterations=result.get("iterations", 0),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/qdrant/info", tags=["Infrastructure"])
async def qdrant_info():
    """Get Qdrant collection information."""
    try:
        qdrant = get_qdrant_service()
        return await qdrant.get_collection_info()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/neo4j/stats", tags=["Infrastructure"])
async def neo4j_stats():
    """Get Neo4j graph statistics."""
    try:
        neo4j = get_neo4j_service()
        return await neo4j.get_stats()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
