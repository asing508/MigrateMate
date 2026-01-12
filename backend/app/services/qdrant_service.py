"""Qdrant vector service for semantic code search."""

from typing import List, Dict, Any, Optional
from uuid import uuid4
import logging
from qdrant_client import AsyncQdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse
from app.core.config import settings
from app.services.embedding_service import get_embedding_service

logger = logging.getLogger(__name__)


class QdrantService:
    """Service for managing code vectors in Qdrant."""
    
    DEFAULT_COLLECTION = "migratemate_code"
    
    def __init__(self):
        self._client: Optional[AsyncQdrantClient] = None
    
    async def get_client(self) -> AsyncQdrantClient:
        if self._client is None:
            self._client = AsyncQdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_HTTP_PORT)
        return self._client
    
    async def close(self) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None
    
    async def create_collection(self, collection_name: str = None, dimension: int = None) -> bool:
        client = await self.get_client()
        collection_name = collection_name or self.DEFAULT_COLLECTION
        
        if dimension is None:
            embedding_service = get_embedding_service()
            dimension = embedding_service.dimension
        
        try:
            collections = await client.get_collections()
            if collection_name in [c.name for c in collections.collections]:
                logger.info(f"Collection '{collection_name}' already exists")
                return False
            
            await client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(size=dimension, distance=models.Distance.COSINE),
                optimizers_config=models.OptimizersConfigDiff(indexing_threshold=10000),
                on_disk_payload=True,
            )
            
            for field in ["project_id", "chunk_type", "file_path"]:
                schema = models.PayloadSchemaType.INTEGER if field == "project_id" else models.PayloadSchemaType.KEYWORD
                await client.create_payload_index(collection_name=collection_name, field_name=field, field_schema=schema)
            
            logger.info(f"Created collection '{collection_name}'")
            return True
        except Exception as e:
            logger.error(f"Failed to create collection: {e}")
            raise
    
    async def upsert_vectors(self, vectors: List[List[float]], payloads: List[Dict[str, Any]], ids: List[str] = None, collection_name: str = None) -> List[str]:
        client = await self.get_client()
        collection_name = collection_name or self.DEFAULT_COLLECTION
        
        if ids is None:
            ids = [str(uuid4()) for _ in vectors]
        
        await self.create_collection(collection_name)
        
        points = [models.PointStruct(id=pid, vector=vec, payload=pay) for pid, vec, pay in zip(ids, vectors, payloads)]
        
        for i in range(0, len(points), 100):
            await client.upsert(collection_name=collection_name, points=points[i:i+100])
        
        logger.info(f"Upserted {len(points)} vectors")
        return ids
    
    async def search(self, query_vector: List[float], project_id: int = None, chunk_types: List[str] = None, limit: int = 10, score_threshold: float = 0.0, collection_name: str = None) -> List[Dict[str, Any]]:
        client = await self.get_client()
        collection_name = collection_name or self.DEFAULT_COLLECTION
        
        filter_conditions = []
        if project_id is not None:
            filter_conditions.append(models.FieldCondition(key="project_id", match=models.MatchValue(value=project_id)))
        if chunk_types:
            filter_conditions.append(models.FieldCondition(key="chunk_type", match=models.MatchAny(any=chunk_types)))
        
        query_filter = models.Filter(must=filter_conditions) if filter_conditions else None
        
        try:
            results = await client.search(collection_name=collection_name, query_vector=query_vector, query_filter=query_filter, limit=limit, score_threshold=score_threshold, with_payload=True)
            return [{"id": str(r.id), "score": r.score, "payload": r.payload} for r in results]
        except UnexpectedResponse as e:
            if "doesn't exist" in str(e):
                return []
            raise
    
    async def search_by_text(self, query_text: str, project_id: int = None, chunk_types: List[str] = None, limit: int = 10, collection_name: str = None) -> List[Dict[str, Any]]:
        embedding_service = get_embedding_service()
        query_vector = await embedding_service.embed_text(query_text)
        return await self.search(query_vector=query_vector, project_id=project_id, chunk_types=chunk_types, limit=limit, collection_name=collection_name)
    
    async def delete_by_project(self, project_id: int, collection_name: str = None) -> int:
        client = await self.get_client()
        collection_name = collection_name or self.DEFAULT_COLLECTION
        try:
            result = await client.delete(collection_name=collection_name, points_selector=models.FilterSelector(filter=models.Filter(must=[models.FieldCondition(key="project_id", match=models.MatchValue(value=project_id))])))
            return result.operation_id or 0
        except Exception as e:
            logger.error(f"Failed to delete: {e}")
            raise
    
    async def get_collection_info(self, collection_name: str = None) -> Dict[str, Any]:
        client = await self.get_client()
        collection_name = collection_name or self.DEFAULT_COLLECTION
        try:
            info = await client.get_collection(collection_name)
            return {"name": collection_name, "vectors_count": info.vectors_count, "points_count": info.points_count, "status": info.status.name}
        except Exception:
            return {"name": collection_name, "error": "Collection not found"}


_qdrant_service: Optional[QdrantService] = None

def get_qdrant_service() -> QdrantService:
    global _qdrant_service
    if _qdrant_service is None:
        _qdrant_service = QdrantService()
    return _qdrant_service
